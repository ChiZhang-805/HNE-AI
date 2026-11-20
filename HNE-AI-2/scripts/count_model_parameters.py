"""Count the actual research modules and original weights; no training or GPU use."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pathlib import Path
import json, struct, math, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from precursor_guidance import ConditionalPredictor, EvidenceBias, UtilityGate
from event_readout import FamilyEventHead
from fusion.evidence import FullEvidenceFusion, InputEvidenceResidual
from study_inputs import SOURCE

def count_parameters(module):
    return sum(p.numel() for p in module.parameters())

def count_safetensors_header(path):
    # Inspect shapes without loading the large tensor payload or allocating a GPU.
    with path.open('rb') as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        header = json.loads(handle.read(length))
    return sum(math.prod(value['shape']) for key,value in header.items() if key!='__metadata__')

def calculate_budget():
    cfg=json.loads((ROOT/'configs/study.json').read_text(encoding='utf8'))
    small_cfg=cfg['small']; guide_cfg=cfg['guidance']; backbone=cfg['backbone']
    modules={
        'small_predictor':ConditionalPredictor(patch_size=small_cfg['patch_hours'],width=small_cfg['width'],dilations=small_cfg['dilations']),
        'guidance_heads':EvidenceBias(hidden=guide_cfg['time_head_dimensions'][1],bound=guide_cfg['bound_after_centering']),
        'event_head':FamilyEventHead(),
        'utility_gate':UtilityGate(cfg['utility_gate']['input_dim']),
    }
    counts={name:count_parameters(module) for name,module in modules.items()}
    lora=len(backbone['lora_blocks_zero_based'])*len(backbone['lora_attention_types'])*len(backbone['lora_projections'])*backbone['lora_rank']*(768+768)
    path=SOURCE/'tools/models/chronos-2/model.safetensors'
    return {
        'status':'instantiated_module_budget_not_a_trained_model',
        'parameters':counts,
        'control_heads':{'full_evidence_fusion':count_parameters(FullEvidenceFusion()),
                         'full_S_evidence_only':count_parameters(FullEvidenceFusion('evidence')),
                         'full_evidence_input_embedding_residual':count_parameters(InputEvidenceResidual()),
                         'full_evidence_logit_fusion':count_parameters(FullEvidenceFusion('logit')),
                         'summary_hidden_fusion':721,'summary_logit_residual':856},
        'planned_control_heads':{
            'status':'design_arithmetic_not_instantiated',
            'retained_node_only_input_ablation':1546},
        'chronos2_local_tensor_parameters':count_safetensors_header(path),
        'planned_lora_parameters':lora,
        'single_deployment_path_added_parameters':sum(counts.values())+lora,
        'personal_parameters_per_channel':cfg['personal_calibration']['personal_parameters_per_channel'],
        'cross_fit_small_copies':cfg['cross_fitting']['small_inner_person_groups'],
        'small_components':{key:count_parameters(value) for key,value in modules['small_predictor'].named_children()},
        'guidance_components':{key:count_parameters(value) for key,value in modules['guidance_heads'].named_children()},
        'weight_metadata_source':str(path),
        'gpu_used':False,
    }

if __name__=='__main__':
    result=calculate_budget()
    (ROOT/'configs/parameter_budget.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))
