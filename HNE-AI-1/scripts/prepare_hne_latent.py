"""Cache the frozen ten-block Chronos prefix; retain two blocks for latent feedback."""
from pathlib import Path
import numpy as np
import torch
from chronos import Chronos2Pipeline

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)
    model = Chronos2Pipeline.from_pretrained(str(ROOT/'tools/models/chronos-2'), device_map='cuda').model.eval()
    config = model.config.to_dict()
    torch.save(dict(config=config, blocks=[b.cpu().state_dict() for b in model.encoder.block[-2:]],
                    norm=model.encoder.final_layer_norm.cpu().state_dict()), ROOT/'tools/models/chronos-2/latent_core.pt')
    model.cuda()
    captured = {}
    def capture(module, args, kwargs):
        captured['x'] = args[0].detach()
    handle = model.encoder.block[-2].register_forward_pre_hook(capture, with_kwargs=True)
    for dataset in ['RESILIENT', 'BrisT1D-Open']:
        source = ROOT/'data/hne_routed'/dataset
        dest = ROOT/'data/hne_latent'/dataset
        dest.mkdir(parents=True, exist_ok=True)
        arrays = np.load(source/'arrays.npz')
        for name in ['history', 'reference']:
            path = dest/f'{name}_prefix.npy'
            if path.exists():
                continue
            raw = arrays[name]; length = raw.shape[-1]; channels = raw.shape[1]
            tokens = length//16+2
            output = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16,
                                              shape=(len(raw), channels, tokens, 768))
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                for start in range(0, len(raw), 16):
                    x = torch.as_tensor(raw[start:start+16].reshape(-1, length), device='cuda')
                    groups = torch.arange(len(x)//channels, device='cuda').repeat_interleave(channels)
                    model.encode(x, group_ids=groups)
                    output[start:start+16] = captured['x'].float().cpu().numpy().reshape(-1, channels, tokens, 768)
            output.flush()
            print(dataset, name, output.shape, 'frozen ten-block prefix saved', flush=True)
    handle.remove()


if __name__ == '__main__':
    main()
