"""Check matched training stages on synthetic windows with the public backbone."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time

import psutil

from run import (ROOT, SOURCE, RESIDUAL_METHODS, capture_prefix, encoder_tail,
                 load_backbone, read_event, resource_check, to_device, torch, train_stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    device = args.device
    if os.name == 'nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    resource_check(device)
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'))
    config.update(device=device, event_epochs=1, effective_batch=1)
    torch.manual_seed(config['seed'])
    started = time.perf_counter()
    model, head = load_backbone(SOURCE / 'tools/models/chronos-2', device)
    scratch = ROOT / 'artifacts' / 'verification'
    scratch.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=scratch, prefix='matched-stages-') as target:
        root = Path(target)
        # Use two observed channels with a full six-channel evidence packet.
        keep = torch.tensor([True, True, False, False, False, False])
        family = torch.tensor([[0, 1, 2, 2, 2, 3]])
        source = SimpleNamespace(family=family)
        items = {'train': [], 'validation': []}
        for index, split in enumerate(items):
            context = torch.randn(2, 2112)
            packed = dict(context=context, observed=torch.ones_like(context, dtype=torch.bool),
                          hour_offsets=torch.arange(-2112, 0)[None].expand(2, -1),
                          group_ids=torch.zeros(2, dtype=torch.long))
            saved = capture_prefix(model, to_device(packed, device))
            prefix_path = root / f'{split}.pt'
            torch.save(saved, prefix_path)
            node_valid = keep[:, None].expand(-1, 21)
            edge_valid = (keep[:, None] & keep[None, :] & ~torch.eye(6, dtype=torch.bool))
            evidence = dict(node=torch.randn(6, 21, 9), node_valid=node_valid,
                            edge=torch.randn(21, 6, 6, 4), edge_valid=edge_valid[None].expand(21, -1, -1),
                            summary=torch.zeros(12))
            items[split].append(dict(person=split, origin=89 + index, label=float(index),
                                     prefix_path=prefix_path, evidence=evidence,
                                     state=torch.ones(1, 2, dtype=torch.bool, device=device),
                                     family=family[:, keep].to(device), keep=keep.to(device), source=source))

        def log(event, **values):
            print(json.dumps({'event': event, **values}), flush=True)

        _, _, evaluate, _ = train_stage('baseline', 0, .001, .0001, config, root, items, model, head, log)
        original = evaluate(items['validation'])
        model.requires_grad_(False)
        model.zero_grad(set_to_none=True)
        head.requires_grad_(False)
        head.zero_grad(set_to_none=True)
        with torch.no_grad():
            for rows in items.values():
                for item in rows:
                    cached = to_device(torch.load(item['prefix_path'], weights_only=True), device)
                    logit, feature = read_event(head, encoder_tail(model, cached), cached,
                                                item['state'], item['family'], config['dataset'], True)
                    item['baseline'] = dict(logit=logit[0], feature=feature[0])
        anchor = evaluate(items['validation'], True)
        torch.testing.assert_close(torch.tensor(original[0]['logit']), torch.tensor(anchor[0]['logit']),
                                   atol=2e-6, rtol=2e-5)
        checked = {}
        for mode, name in RESIDUAL_METHODS.items():
            _, result, evaluate, restore = train_stage(mode, 0, .001, .0001, config, root, items, model, head, log)
            selected = evaluate(items['validation'])
            assert evaluate(items['validation'], True) == anchor
            restore(result['trained_state'])
            always_on = evaluate(items['validation'])
            _, reloaded, evaluate, restore = train_stage(mode, 0, .001, .0001, config, root, items, model, head, log)
            assert evaluate(items['validation']) == selected
            restore(reloaded['trained_state'])
            assert evaluate(items['validation']) == always_on
            assert all(p.grad is None for module in (model, head) for p in module.parameters())
            checked[name] = dict(off_exact=True, checkpoint_reload_exact=True,
                                 inspected_epochs=result['report']['epochs_completed'],
                                 absolute_logit_change=abs(always_on[0]['logit'] - anchor[0]['logit']))
        print(json.dumps(dict(status='passed', device=device, seconds=time.perf_counter() - started,
                              synthetic_only=True, checks=checked,
                              reforward_abs_error=abs(original[0]['logit'] - anchor[0]['logit']))), flush=True)


if __name__ == '__main__':
    main()
