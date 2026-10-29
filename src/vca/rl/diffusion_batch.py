"""Collect multiple real trajectories under ONE behavior policy, then apply Eq19/20.

No off-policy correction or new objective: current-version PPO rollout batching.
"""
from pathlib import Path
from dataclasses import asdict
import copy
import torch
from .rollout import RolloutBatch
from .replay import DiagnosticReplay
from ..runtime.pipeline import VCARuntime,RuntimeConfig
from ..runtime.feedback import Feedback
from ..runtime.logging import write_json

def train_batched_episodes(episodes,generator,scorer,trainer,out,*,seed=42,size=256,steps=12,
                          rollout_episodes=2,dataset=True,llm=None,mapper_alpha_mode='legacy',
                          rag_enabled=True,strict_llm=False,save_attention_samples=True,allow_clarifications=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);pool=DiagnosticReplay();clock=0;updates=[]
    if rollout_episodes<2:raise ValueError('Multi-trajectory collector requires >=2 episodes per update')
    for start in range(0,len(episodes),rollout_episodes):
        version=trainer.version;runtimes=[];metadata=[];pool_indices=[]
        expected=min(rollout_episodes,len(episodes)-start)*3
        if expected>pool.capacity:raise ValueError('Requested on-policy rollout exceeds replay capacity')
        keep=pool.capacity-expected
        pool.rows=pool.rows[-keep:] if keep else []  # do not shift current-batch indices during append
        for offset,episode in enumerate(episodes[start:start+rollout_episodes]):
            episode_index=start+offset
            # ppo=False defers automatic per-round updates. The initialized actor,
            # action path and loss are unchanged; an actual batched update follows.
            cfg=RuntimeConfig(seed=seed+episode_index,size=size,step_choices=(steps,steps),initial_steps=steps,
                              max_rounds=3,ppo=False,action_authority='policy',mapper_alpha_mode=mapper_alpha_mode,
                              rag_enabled=rag_enabled,strict_llm=strict_llm,save_attention_samples=save_attention_samples)
            runtime=VCARuntime(out/f'episode_{episode_index:03d}',episode['prompt'],generator,scorer,cfg,trainer,llm=llm)
            runtime.session.data['ppo_collection_mode']='deferred_same_policy_version_batch'
            write_json(runtime.session.folder/'data_record.json',episode)
            for feedback_index,feedback in enumerate(episode['feedbacks']):
                try:
                    row=runtime.step(Feedback(feedback,source='dataset_annotation' if dataset else 'scripted_fixture'))
                except Exception as exc:
                    if not allow_clarifications:raise
                    write_json(runtime.session.folder/f'feedback_failure_{feedback_index:03d}.json',
                               {'feedback_index':feedback_index,'feedback':feedback,'error':type(exc).__name__+': '+str(exc),
                                'handling':'episode stopped; no fallback or synthetic transition'})
                    break
                if row['status']!='generated':
                    if allow_clarifications:continue
                    raise RuntimeError('Unresolved dataset clarification: '+row.get('question',''))
                import json
                path=runtime.session.folder/f"round_{row['round_id']:03d}"/'transition.json'
                transition=json.loads(path.read_text(encoding='utf-8'))
                if transition['policy_version']!=version:raise ValueError('Behavior policy changed during rollout collection')
                clock+=1;transition['age_clock']=clock;transition['trajectory']=str(runtime.session.folder)
                transition['transition_path']=str(path)
                td=transition['reward']['reward']+trainer.config.gamma*(not transition['done'])*transition['next_value']-transition['value']
                pool.append(transition,td,clock);pool_indices.append(len(pool.rows)-1);metadata.append(transition)
            runtimes.append(runtime)
        if len(metadata)<2:raise RuntimeError('Fewer than two generated transitions; no prioritized batch update')
        batch=RolloutBatch(
            torch.tensor([m['state'] for m in metadata]),torch.tensor([m['action'] for m in metadata]),
            torch.tensor([m['old_logprob'] for m in metadata]),torch.tensor([m['value'] for m in metadata]),
            torch.tensor([m['reward']['reward'] for m in metadata]),torch.tensor([m['next_value'] for m in metadata]),
            torch.tensor([m['done'] for m in metadata]),torch.tensor([m['policy_version'] for m in metadata]),
            torch.tensor([m['type_mask'] for m in metadata]),metadata)
        log=trainer.update(batch,replay_pool=pool,replay_indices=pool_indices,age_clock=clock)
        batch_id=len(updates);folder=out/f'update_{batch_id:03d}';folder.mkdir(exist_ok=False)
        checkpoint=folder/'checkpoint.pt';trainer.save(checkpoint)
        write_json(folder/'rollout.json',metadata);write_json(folder/'training.json',log)
        write_json(folder/'replay_diagnostic.json',pool.sample(min(32,len(pool.rows)),clock,beta=1.,seed=seed))
        torch.save({'states':batch.states,'actions':batch.actions,'old_logprob':batch.old_logprob,
            'values':batch.values,'rewards':batch.rewards,'next_values':batch.next_values,'dones':batch.dones,
            'versions':batch.versions,'type_masks':batch.type_masks},folder/'rollout.pt')
        first=log['replay_sampling'][0]
        entry={'batch_id':batch_id,'behavior_version':version,'new_version':trainer.version,'transitions':len(metadata),
            'episodes':len(runtimes),'probabilities_first_epoch':first['probabilities'],
            'importance_first_epoch':first['raw_importance_weights'],'ages':first['ages'],
            'actor_delta':log['actor_parameter_delta_l2'],'critic_delta':log['critic_parameter_delta_l2'],
            'type_head_delta':log['actor_type_head_delta_l2'],'optimizer_steps':log['optimizer_steps'],
            'mapper_delta':log['mapper_parameter_delta_l2'],'alpha_delta':log['alpha_parameter_delta_l2'],
            'mapper_grad_max':max(x['mapper_grad_norm'] for x in log['losses']),
            'alpha_grad_max':max(x['alpha_grad_norm'] for x in log['losses']),
            'checkpoint':str(checkpoint),'classification':'paper-faithful Eq19/20; rollout grouping/age clock are implementation-assumptions'}
        updates.append(entry)
        for runtime in runtimes:
            runtime.session.data.update(checkpoint=str(checkpoint),replay_rows=copy.deepcopy(pool.rows),
                external_batch_update={'batch_id':batch_id,'training_log':str(folder/'training.json'),'policy_version':trainer.version})
            runtime.session.save();runtime.finalize()
        write_json(out/'summary.json',{'updates':updates,'total_transitions':sum(u['transitions'] for u in updates),
            'strict_same_behavior_version':True,'replay_loss_used':True,'paper_absolute_metrics_reproduced':False})
        print(f'BATCH PPO update={batch_id} transitions={len(metadata)} policy={version}->{trainer.version}',flush=True)
    return updates
