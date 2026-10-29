"""Record real-LLM ambiguity behavior; success is measured, never supplied by a rule fallback."""
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from vca.sesd.llm import LocalLlamaCppLLM,LocalTransformersLLM
from vca.scoring.clip_reward import CLIPReward
from vca.runtime.logging import write_json
from deep_experiment import llm_diagnostic

def main():
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,default=ROOT)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(42)
    manifest=json.loads((a.model/'vca_manifest.json').read_text(encoding='utf-8'))
    scorer=CLIPReward(a.project_root/'models/clip',device='cpu')
    llm=LocalLlamaCppLLM(a.model) if manifest.get('quantization') else LocalTransformersLLM(a.model)
    try:
        cases=llm_diagnostic(llm,scorer)
        result={'model':manifest,'cases':cases,'lexical_proxy_passes':sum(x['lexical_proxy_pass'] for x in cases),
            'total':len(cases),'real_model':True,'rule_fallback':False,
            'scope':'paper-example functional diagnosis, not held-out dataset experiment'}
        write_json(a.out/'summary.json',result);print('LLM_PROBE '+str(a.out),flush=True)
    finally:write_json(a.out/'llm_events.json',llm.events)

if __name__=='__main__':main()
