"""Held-out real-data/real-LLM/real-diffusion comparison; no simulated model output."""
from pathlib import Path
import argparse,hashlib,json,sys,traceback
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from PIL import Image
from vca.dataset import load_catalog,episode_from_record
from vca.grouped_split import grouped_split,verify_no_leak
from vca.editing.pipeline_adapter import PairDiffusion,PairPrompt
from vca.editing.token_alignment import align,tokenize,TokenAlignment
from vca.editing.controllers import AttentionController,normalize
from vca.scoring.clip_reward import CLIPReward
from vca.scoring.metrics import ssim,lpips,bootstrap_ci
from vca.sesd.dialogue import Dialogue
from vca.sesd.llm import LocalLlamaCppLLM,LocalTransformersLLM
from vca.rl.mapper_alpha import MapperAlphaPolicy
from vca.rl.ppo import PPO,PPOConfig
from vca.rl.diffusion_batch import train_batched_episodes
from vca.runtime.pipeline import VCARuntime,RuntimeConfig
from vca.runtime.feedback import Feedback
from vca.runtime.logging import write_json,environment

VARIANTS=('full','fixed_editing','no_ppo','no_rag','no_mapper_alpha')

def file_sha(path):return hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()

def preselect(rows,ids,tokenizer,n):
    selected=[]
    for e in rows:
        if e['record_sha256'] not in ids or len(e.get('qa_pairs',[]))<3:continue
        episode=episode_from_record(e);prompt=episode['prompt'];answers=episode['feedbacks']
        counts=[len(tokenizer(x,add_special_tokens=False,truncation=False)['input_ids']) for x in [prompt,*answers]]
        if counts[0]<=48 and max(counts[1:])<=24 and sum(counts)<=68:
            selected.append({**episode,'input_token_counts':counts,'split_group':'preassigned'})
            if len(selected)>=n:break
    return selected

def llm_diagnostic(llm,scorer):
    cases=[('mouse','a mouse on a wooden desk','computer device',('computer','mouse')),
           ('bat','a bat on the grass','baseball tool',('baseball','bat')),
           ('bank','a bank in the landscape','river edge',('river','bank')),
           ('spring','a spring in a forest','water source',('water','spring'))]
    rows=[]
    for word,prompt,feedback,expected in cases:
        d=Dialogue(prompt,llm=llm,strict_llm=True,retrieval_embedder=scorer.retrieval_embeddings)
        try:
            question=d.refine('');answer=d.refine(feedback)
            rows.append({'word':word,'question':question.question,'rewritten_prompt':answer.new_prompt,
                'operation':answer.type,'followup_question':answer.question,
                'expected_terms_present':all(x in answer.new_prompt.lower() for x in expected),
                'lexical_proxy_pass':answer.question is None and all(x in answer.new_prompt.lower() for x in expected),
                'knowledge_alias_match':word in d.resolved and answer.question is None,
                'question_mentions_ambiguous_term':bool(question.question and word in question.question.lower()),
                'question_origin':'deterministic_guard' if any('deterministic ambiguity guard' in x for x in question.assumptions) else 'LLM',
                'quality_note':'lexical checks are not a human semantic correctness judgment',
                'backend':answer.backend,'fallback':answer.schema_fallback,
                'evidence_ids':[x['id'] for x in answer.evidence]})
        except Exception as e:rows.append({'word':word,'error':type(e).__name__+': '+str(e),'lexical_proxy_pass':False})
        print('LLM_CASE '+word+' '+json.dumps(rows[-1],ensure_ascii=True),flush=True)
    return rows

def compare_real_attention(generator,trained,initial,case,folder,seed,steps,size):
    """Use identical actual source attention and same inputs, separate mapper/beta effects."""
    folder.mkdir(parents=True,exist_ok=False)
    before=case['previous_prompt'];after=case['new_prompt']
    alignment=align(generator.pipe.tokenizer,before,after)
    map0,beta0=initial.adapter.mean_controls(alignment);map1,beta1=trained.adapter.mean_controls(alignment)
    reports={};latents=generator.initial_latents(seed,size,size);maps={}
    for name,mapper,beta in [('before',map0,beta0),('baseline_repeat',map0,beta0),('mapper_only',map1,beta0),('alpha_only',map0,beta1),('after',map1,beta1)]:
        a=TokenAlignment(alignment.source,alignment.target,mapper,alignment.mapped_target)
        c=AttentionController(a,'refine',steps=steps,strength=beta,cross_fraction=.7,record_target_samples=name=='before')
        result=generator.generate(PairPrompt(before,after),c,seed=seed,steps=steps,height=size,width=size,latents=latents)
        result['image'].save(folder/(name+'.png'))
        maps[name]=result['latents'];reports[name]={'alpha':beta,'trace_calls':len(c.capture.trace),
            'injections':sum(t['delta_l1']>0 for t in c.capture.trace)}
        if name=='before':
            if not c.capture.samples:raise ValueError('No source attention samples recorded')
            sample_key=next(iter(c.capture.samples));source_half=c.capture.samples[sample_key]
            target_half=c.target_samples[sample_key];source=source_half.float();target=target_half.float()
            torch.save({'source':source_half,'target':target_half,'layer_step':sample_key},folder/'observed_network_attention.pt')
            first=torch.einsum('bhqk,kj->bhqj',source,map0)
            changed=torch.einsum('bhqk,kj->bhqj',source,map1)
            reports['actual_source_map_projection_delta_l1']=float((changed-first).abs().mean())
            reports['actual_source_map_shape']=list(source.shape)
            mask=alignment.mapped_target.view(1,1,1,-1)
            orig=normalize(beta0*torch.where(mask,first,target)+(1-beta0)*target)
            map_changed=normalize(beta0*torch.where(mask,changed,target)+(1-beta0)*target)
            alpha_changed=normalize(beta1*torch.where(mask,first,target)+(1-beta1)*target)
            reports['actual_injected_attention_delta']={
                'mapper_fp32_l1':float((map_changed-orig).abs().mean()),
                'alpha_fp32_l1':float((alpha_changed-orig).abs().mean()),
                'mapper_after_fp16_l1':float((map_changed.to(target_half.dtype).float()-orig.to(target_half.dtype).float()).abs().mean()),
                'alpha_after_fp16_l1':float((alpha_changed.to(target_half.dtype).float()-orig.to(target_half.dtype).float()).abs().mean())}
    for name in ('baseline_repeat','mapper_only','alpha_only','after'):
        reports[name]['edited_latent_l2_vs_before']=float((maps[name][1].float()-maps['before'][1].float()).norm())
        reports[name]['source_latents_unchanged']=torch.equal(maps[name][0],maps['before'][0])
    reports['mapper_matrix_delta_l2']=float((map1-map0).norm());reports['alpha_mean_delta']=beta1-beta0
    reports['interpretation']='Compare changed latent distances against baseline_repeat to assess numerical nondeterminism; attention deltas are isolated on identical recorded tensors.'
    reports['classification']='same-seed real frozen SD attention; score-function-trained mapper/beta, not CLIP backprop'
    write_json(folder/'effect.json',reports);return reports

def run():
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,default=ROOT)
    p.add_argument('--llm-model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--train-cases',type=int,default=2);p.add_argument('--eval-cases',type=int,default=4)
    p.add_argument('--seeds',type=int,nargs='+',default=[41,42]);p.add_argument('--steps',type=int,default=8)
    p.add_argument('--size',type=int,default=256);p.add_argument('--lr',type=float,default=5e-5);a=p.parse_args()
    root=a.project_root;out=a.out;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    config={'method_basis':'paper/main.pdf only','llm_model':str(a.llm_model),'steps':a.steps,'size':a.size,
        'seeds':a.seeds,'train_cases':a.train_cases,'eval_cases':a.eval_cases,'train_lr':a.lr,
        'sampling':'DDIM, fixed latent seeds and max 3 feedback/image rounds; each completed round uses fixed steps',
        'metrics':['history raw CLIP cosine vs original prompt','new request raw CLIP cosine vs last user answer',
                   'SSIM and LPIPS vs initial generated image'],
        'variants':list(VARIANTS),'inference':'all policy/mapper weights frozen; no held-out optimizer updates',
        'success_policy':'LLM clarification/error or token overflow is recorded; no rule fallback or fabricated image',
        'scope':'low-budget held-out real-data pilot, not paper-scale metric reproduction'}
    write_json(out/'config.json',config);write_json(out/'environment.json',environment(root))
    llm=None;report={'status':'started','stages':{}}
    try:
        rows=load_catalog(root/'data/catalog.jsonl',min_turns=0)
        split=grouped_split(rows);proof=verify_no_leak(rows,split)
        split.update(proof=proof,catalog_sha256=file_sha(root/'data/catalog.jsonl'))
        write_json(out/'grouped_split.json',split)
        generator=PairDiffusion(root/'models/sd15');scorer=CLIPReward(root/'models/clip',device='cpu')
        train=preselect(rows,set(split['train']),generator.pipe.tokenizer,a.train_cases)
        validation=preselect(rows,set(split['validation']),generator.pipe.tokenizer,a.eval_cases)
        if len(train)!=a.train_cases or len(validation)!=a.eval_cases:raise RuntimeError('Insufficient eligible actual Q/A records')
        write_json(out/'selected_data.json',{'train':train,'validation':validation,'leakage_proof':proof,
            'selection':'first hash-sorted entries meeting predeclared Q/A/token-budget filters; no validation outcome selection'})
        info=json.loads((a.llm_model/'vca_manifest.json').read_text(encoding='utf-8'))
        llm=LocalLlamaCppLLM(a.llm_model) if info.get('quantization') else LocalTransformersLLM(a.llm_model)
        diag=llm_diagnostic(llm,scorer);write_json(out/'llm_diagnostic.json',diag);write_json(out/'llm_events.json',llm.events)
        report['stages']['llm']={'backend':info['id'],'lexical_proxy_passes':sum(x['lexical_proxy_pass'] for x in diag),'cases':len(diag),
                                'formal_rule_fallbacks':0}
        checkpoints={};training={}
        for mode in ('learned','fixed'):
            torch.manual_seed(20261027)
            adapter=MapperAlphaPolicy() if mode=='learned' else None
            trainer=PPO(config=PPOConfig(lr=a.lr,minibatch=6,epochs=4,ignore_strength_head=True),adapter=adapter)
            path=out/f'train_{mode}';path.mkdir()
            initial=path/'initial.pt';trainer.save(initial)
            updates=train_batched_episodes(train,generator,scorer,trainer,path/'trajectories',seed=20261027,size=a.size,
                steps=a.steps,rollout_episodes=a.train_cases,dataset=True,llm=llm,mapper_alpha_mode=mode,
                strict_llm=True,save_attention_samples=False,allow_clarifications=True)
            trainer.save(path/'trained.pt');checkpoints[mode]={'initial':str(initial),'trained':str(path/'trained.pt')}
            training[mode]=updates;write_json(out/'llm_events.json',llm.events)
            print('TRAINED '+mode+' '+json.dumps(updates),flush=True)
        write_json(out/'training_summary.json',{'training':training,'checkpoints':checkpoints})
        report['stages']['training']=training
        # Isolate real-map vs beta changes on the same stored training prompt pair.
        actual=next((f for f in (out/'train_learned/trajectories').glob('episode_*/round_*/result.json')),None)
        if actual:
            case=json.loads(actual.read_text(encoding='utf-8'))
            initial=PPO.load(checkpoints['learned']['initial'],restore_rng=False)
            trained=PPO.load(checkpoints['learned']['trained'],restore_rng=False)
            effect=compare_real_attention(generator,trained,initial,case,out/'mapper_alpha_effect',a.seeds[0],a.steps,a.size)
            report['stages']['real_attention_effect']=effect
        measurements=[]
        for episode in validation:
            for seed in a.seeds:
                for variant in VARIANTS:
                    # Weights are reset identically for every held-out episode and
                    # never optimized on validation data.
                    if variant in ('full','no_rag'):ckpt=checkpoints['learned']['trained'];mode='learned'
                    elif variant=='no_ppo':ckpt=checkpoints['learned']['initial'];mode='learned'
                    elif variant=='no_mapper_alpha':ckpt=checkpoints['fixed']['trained'];mode='fixed'
                    else:ckpt=checkpoints['fixed']['initial'];mode='fixed'
                    trainer=PPO.load(ckpt,restore_rng=False);torch.manual_seed(seed)
                    cfg=RuntimeConfig(seed=seed,size=a.size,step_choices=(a.steps,a.steps),initial_steps=a.steps,max_rounds=3,
                        ppo=False,action_authority='user' if variant=='fixed_editing' else 'policy',mapper_alpha_mode=mode,
                        rag_enabled=variant!='no_rag',strict_llm=True,policy_mode='fixed' if variant=='fixed_editing' else 'stochastic',
                        save_attention_samples=False)
                    folder=out/'eval'/episode['record_id']/str(seed)/variant
                    row={'record_id':episode['record_id'],'record_sha256':episode['record_sha256'],'seed':seed,
                         'variant':variant,'model_checkpoint':ckpt,'status':'started','real_llm':info['id'],
                         'training_fold':'train','evaluation_fold':'validation','generation_budget_steps':a.steps}
                    event_start=len(llm.events)
                    try:
                        runtime=VCARuntime(folder,episode['prompt'],generator,scorer,cfg,trainer,llm=llm)
                        write_json(folder/'data_record.json',episode)
                        for feedback in episode['feedbacks']:
                            result=runtime.step(Feedback(feedback,source='dataset_annotation'))
                            if result['status']!='generated':raise RuntimeError('LLM asked clarification: '+str(result['question']))
                            if result['proposal']['schema_fallback'] is not None or result['proposal']['backend'] not in ('LocalLlamaCppLLM','LocalTransformersLLM'):
                                raise ValueError('Formal evaluation must use a real LLM without mock fallback')
                        last=Image.open(runtime.session.data['current_image']).convert('RGB')
                        initial_image=Image.open(runtime.session.data['initial_image']).convert('RGB')
                        scores=scorer.score(last,episode['prompt'],episode['feedbacks'][-1]);l=lpips(initial_image,last)
                        if l['status']!='measured':raise RuntimeError('LPIPS unavailable: '+str(l))
                        row.update(status='measured',history_raw_cosine=scores['raw_cosine_previous'],
                            latest_request_raw_cosine=scores['raw_cosine_new'],ssim_initial_final=ssim(initial_image,last),
                            lpips_initial_final=l['value'],rounds=3,images_generated=4,final_image=runtime.session.data['current_image'],
                            score_scale='raw cosine, not paper calibrated score',policy_version=trainer.version,
                            optimizer_updates_on_validation=0,feedback_source='dataset_annotation')
                        assert all(not any(r['reward'].get('text_truncated',[])) for r in runtime.session.data['rounds'])
                        assert not trainer.adapter or all(p.grad is None for p in trainer.adapter.parameters())
                        if variant=='no_rag':assert all(not r['proposal']['evidence'] for r in runtime.session.data['rounds'])
                        runtime.finalize()
                    except Exception as e:
                        row.update(status='FAILED',error=type(e).__name__+': '+str(e))
                        folder.mkdir(parents=True,exist_ok=True);write_json(folder/'failure.json',row)
                    row['llm_events']=llm.events[event_start:];measurements.append(row)
                    write_json(out/'measurements.json',measurements);write_json(out/'llm_events.json',llm.events)
                    print('EVALUATED '+episode['record_id']+' '+str(seed)+' '+variant+' '+row['status'],flush=True)
        paired={}
        full={(x['record_id'],x['seed']):x for x in measurements if x['variant']=='full' and x['status']=='measured'}
        for variant in VARIANTS[1:]:
            pairs=[(full[(x['record_id'],x['seed'])],x) for x in measurements
                   if x['variant']==variant and x['status']=='measured' and (x['record_id'],x['seed']) in full]
            metrics={}
            for metric in ('latest_request_raw_cosine','history_raw_cosine','ssim_initial_final','lpips_initial_final'):
                per_case={}
                for f,x in pairs:per_case.setdefault(x['record_id'],[]).append(f[metric]-x[metric])
                deltas=[sum(values)/len(values) for values in per_case.values()]
                metrics[metric]=bootstrap_ci(deltas) if deltas else None
                if metrics[metric] and len(deltas)>1:
                    metrics[metric]['method']='percentile bootstrap across independent records; seeds averaged within each record'
            paired[variant]={'paired_record_count':len({x['record_id'] for _,x in pairs}),'paired_seed_count':len(pairs),
                'full_minus_ablation_by_metric':metrics,
                'success_count':sum(x['variant']==variant and x['status']=='measured' for x in measurements),
                'attempted_count':sum(x['variant']==variant for x in measurements)}
        report.update(status='completed',measurements=measurements,paired=paired,
            training_validation_leakage=proof,selected_validation_records=len(validation),
            model_pretraining_overlap='unknown',paper_metric_replication=False,
            inference='No conclusion of superiority unless held-out paired evidence supports it; 4 cases are a pilot.')
        write_json(out/'summary.json',report);print('DEEP_EXPERIMENT_COMPLETE '+str(out),flush=True)
    except Exception as e:
        report.update(status='FAILED',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
        write_json(out/'summary.json',report)
        if llm is not None:write_json(out/'llm_events.json',llm.events)
        print('DEEP_EXPERIMENT_FAILED '+report['error'],flush=True);raise

if __name__=='__main__':run()
