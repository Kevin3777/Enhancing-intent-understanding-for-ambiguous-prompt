from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import argparse,datetime,json
import torch
from vca.rl.toy import train_toy
from vca.rl.ppo import PPO,PPOConfig
from vca.runtime.logging import write_json,environment

def main():
    p=argparse.ArgumentParser();p.add_argument('--env',choices=['toy','diffusion'],default='toy')
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);p.add_argument('--iterations',type=int,default=60)
    p.add_argument('--episodes',type=int,default=1);p.add_argument('--out',type=Path)
    p.add_argument('--size',type=int,default=256);p.add_argument('--steps',type=int,default=12)
    p.add_argument('--lr',type=float,default=5e-5);p.add_argument('--resume',type=Path)
    p.add_argument('--catalog',type=Path);p.add_argument('--record-id');p.add_argument('--provider',action='store_true')
    p.add_argument('--rollout-episodes',type=int,default=1);p.add_argument('--minibatch',type=int,default=64)
    p.add_argument('--local-llm',type=Path);p.add_argument('--mapper-alpha',choices=['legacy','fixed','learned'],default='legacy')
    p.add_argument('--no-rag',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    out=a.out or ROOT/'outputs'/('ppo-'+a.env+'-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True,exist_ok=False);write_json(out/'environment.json',environment(ROOT))
    write_json(out/'arguments.json',vars(a));results=[];llm=None
    if a.env=='toy':
        for seed in a.seeds:
            trainer,result=train_toy(seed,a.iterations)
            trainer.save(out/f'checkpoint_seed_{seed}.pt');results.append(result)
            print(json.dumps(result),flush=True)
        assert all(r['after']>r['before']+.4 for r in results)
    else:
        from vca.editing.pipeline_adapter import PairDiffusion
        from vca.scoring.clip_reward import CLIPReward
        from vca.runtime.pipeline import VCARuntime,RuntimeConfig
        from vca.runtime.feedback import Feedback
        generator=PairDiffusion(ROOT/'models/sd15');scorer=CLIPReward(ROOT/'models/clip',device='cpu')
        llm=None
        if a.provider and a.local_llm:raise ValueError('Choose either provider or local LLM')
        if a.local_llm:
            from vca.sesd.llm import LocalTransformersLLM,LocalLlamaCppLLM
            info=json.loads((a.local_llm/'vca_manifest.json').read_text(encoding='utf-8'))
            llm=LocalLlamaCppLLM(a.local_llm) if info.get('quantization') else LocalTransformersLLM(a.local_llm)
        elif a.provider:
            from vca.sesd.llm import HTTPJSONClient
            llm=HTTPJSONClient()
        episodes=[]
        if a.catalog:
            from vca.dataset import load_catalog,episode_from_record
            from vca.sesd.dialogue import Dialogue
            from vca.editing.token_alignment import tokenize
            skipped=[]
            for entry in load_catalog(a.catalog,'train',min_turns=3):
                if a.record_id and entry['record_id']!=a.record_id:continue
                episode=episode_from_record(entry)
                try:
                    tokenize(generator.pipe.tokenizer,episode['prompt'])
                    trial=Dialogue(episode['prompt'],llm=llm,retrieval_embedder=scorer.retrieval_embeddings if llm else None,
                                   rag_enabled=not a.no_rag,strict_llm=bool(llm))
                    for feedback in episode['feedbacks']:
                        proposal=trial.refine(feedback)
                        if proposal.question:raise ValueError('Needs further human clarification')
                        tokenize(generator.pipe.tokenizer,proposal.new_prompt)
                    episodes.append(episode)
                except ValueError as e:skipped.append({'record_id':entry['record_id'],'reason':str(e)})
                if len(episodes)>=a.episodes:break
            write_json(out/'dataset_selection.json',{'catalog':str(a.catalog),'selected':episodes,'skipped':skipped,
                'filter':'no prompt truncation; >=3 non-placeholder actual answers; mock can execute without unanswered clarification',
                'classification':'implementation-assumption; not paper original split or trajectory equivalence'})
            if not episodes:raise RuntimeError('No executable verified dataset episodes; inspect dataset_selection.json')
        else:
            episodes=[{'prompt':'a mouse sitting on a wooden desk','feedbacks':['computer device','add a red mouse pad','emphasize computer mouse'],
                       'source':'authored scripted fixture'} for _ in range(a.episodes)]
        for seed in a.seeds:
            torch.manual_seed(seed)
            adapter=None
            if a.mapper_alpha=='learned':
                from vca.rl.mapper_alpha import MapperAlphaPolicy
                adapter=MapperAlphaPolicy()
            trainer=PPO.load(a.resume) if a.resume else PPO(config=PPOConfig(lr=a.lr,minibatch=a.minibatch,
                            ignore_strength_head=a.mapper_alpha!='legacy'),adapter=adapter)
            if a.rollout_episodes>1:
                from vca.rl.diffusion_batch import train_batched_episodes
                results.extend(train_batched_episodes(episodes,generator,scorer,trainer,out/f'seed_{seed}_batches',
                    seed=seed,size=a.size,steps=a.steps,rollout_episodes=a.rollout_episodes,dataset=bool(a.catalog),llm=llm,
                    mapper_alpha_mode=a.mapper_alpha,rag_enabled=not a.no_rag,strict_llm=bool(llm)))
                trainer.save(out/f'checkpoint_seed_{seed}.pt')
                continue
            for episode_index,episode in enumerate(episodes):
                cfg=RuntimeConfig(seed=seed,size=a.size,step_choices=(a.steps,a.steps),initial_steps=a.steps,
                                  max_rounds=3,action_authority='policy',mapper_alpha_mode=a.mapper_alpha,
                                  rag_enabled=not a.no_rag,strict_llm=bool(llm))
                runtime=VCARuntime(out/f'seed_{seed}_episode_{episode_index}',episode['prompt'],generator,scorer,cfg,trainer,llm=llm)
                write_json(runtime.session.folder/'data_record.json',episode)
                for text in episode['feedbacks']:
                    row=runtime.step(Feedback(text,source='dataset_annotation' if a.catalog else 'scripted_fixture'))
                    if row['status']!='generated':raise RuntimeError('Dataset needs further clarification: '+row.get('question',''))
                    print(json.dumps({'seed':seed,'episode':episode_index,'status':row['status'],
                        'reward':row.get('reward'),'training':row.get('training')}),flush=True)
                results.append(runtime.finalize());trainer=runtime.trainer
            trainer.save(out/f'checkpoint_seed_{seed}.pt')
    write_json(out/'summary.json',{'environment':a.env,'results':results,'paper_experiment_replicated':False})
    if hasattr(llm,'events'):write_json(out/'real_llm_events.json',llm.events)
    print('PPO COMPLETE '+str(out),flush=True)

if __name__=='__main__':main()
