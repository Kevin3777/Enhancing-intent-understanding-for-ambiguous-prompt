from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import argparse,datetime,json
import torch
from vca.editing.pipeline_adapter import PairDiffusion
from vca.scoring.clip_reward import CLIPReward
from vca.runtime.pipeline import VCARuntime,RuntimeConfig
from vca.runtime.feedback import Feedback
from vca.runtime.logging import write_json,environment

def main():
    p=argparse.ArgumentParser(description='Audited VCA: clarification, frozen diffusion edits, CLIP reward, on-policy PPO')
    p.add_argument('--prompt',default='a mouse sitting on a wooden desk');p.add_argument('--out',type=Path)
    p.add_argument('--resume',type=Path);p.add_argument('--fixture',action='store_true')
    p.add_argument('--size',type=int,default=384);p.add_argument('--steps',type=int,default=20)
    p.add_argument('--max-rounds',type=int,default=8);p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--action-authority',choices=['user','policy'],default='user');p.add_argument('--no-ppo',action='store_true')
    p.add_argument('--provider',action='store_true');p.add_argument('--local-llm',type=Path)
    p.add_argument('--mapper-alpha',choices=['legacy','fixed','learned'],default='legacy')
    p.add_argument('--no-rag',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    folder=a.resume or a.out or ROOT/'outputs'/('session-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    if not a.resume and folder.exists():raise FileExistsError('Choose a new output folder, or use --resume')
    prompt=None if a.resume else a.prompt
    cfg=RuntimeConfig(seed=a.seed,size=a.size,step_choices=(a.steps,a.steps),initial_steps=a.steps,
                      max_rounds=3 if a.fixture else a.max_rounds,ppo=not a.no_ppo,action_authority=a.action_authority,
                      mapper_alpha_mode=a.mapper_alpha,rag_enabled=not a.no_rag,strict_llm=bool(a.local_llm or a.provider))
    generator=PairDiffusion(ROOT/'models/sd15');scorer=CLIPReward(ROOT/'models/clip',device='cpu')
    llm=None
    if a.provider and a.local_llm:raise ValueError('Choose either provider or local LLM')
    if a.local_llm:
        from vca.sesd.llm import LocalTransformersLLM,LocalLlamaCppLLM
        info=json.loads((a.local_llm/'vca_manifest.json').read_text(encoding='utf-8'))
        llm=LocalLlamaCppLLM(a.local_llm) if info.get('quantization') else LocalTransformersLLM(a.local_llm)
    if a.provider:
        from vca.sesd.llm import HTTPJSONClient
        llm=HTTPJSONClient()
    runtime=VCARuntime(folder,prompt,generator,scorer,cfg,llm=llm)
    write_json(folder/'environment.json',environment(ROOT));runtime.initialize()
    if a.resume and runtime.session.data['stop_reason']:runtime.continue_session()
    if a.fixture:
        feedbacks=[Feedback(t) for t in ('computer device','add a red mouse pad','emphasize computer mouse')]
        for feedback in feedbacks:
            result=runtime.step(feedback)
            print(json.dumps({'status':result['status'],'new_prompt':result.get('new_prompt'),
                'executed_operation':result.get('executed_operation'),'reward':result.get('reward'),
                'training':result.get('training'),'question':result.get('question')}),flush=True)
    else:
        print('Commands: /quit, /undo, /rate 1..5, /satisfied; ordinary text supplies actual human feedback.',flush=True)
        rating=None
        while not runtime.session.data['stop_reason']:
            text=input('Feedback> ').strip()
            if text=='/quit':break
            if text=='/undo':runtime.undo();print('Restored prompt, image and policy checkpoint.');continue
            if text.startswith('/rate '):
                rating=float(text.split()[1])
                if not 1<=rating<=5:raise ValueError('Rating must be between 1 and 5')
                continue
            satisfied=text=='/satisfied'
            if satisfied:
                runtime.session.data['stop_reason']='human_satisfied';runtime.session.save();break
            result=runtime.step(Feedback(text,rating,source='human'));rating=None
            print(json.dumps({'status':result['status'],'question':result.get('question'),'image':result.get('image'),
                              'prompt':result.get('new_prompt'),'reward':result.get('reward')},ensure_ascii=False),flush=True)
    print(json.dumps(runtime.finalize(),ensure_ascii=False),flush=True)
    if hasattr(llm,'events'):write_json(folder/'real_llm_events.json',llm.events)

if __name__=='__main__':main()
