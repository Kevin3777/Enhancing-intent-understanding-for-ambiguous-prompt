"""PDF mechanism ablations on transparent synthetic feedback fixtures, not paper data."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import argparse,datetime,json
from dataclasses import replace,asdict
import torch
from PIL import Image
from vca.editing.pipeline_adapter import PairDiffusion
from vca.scoring.clip_reward import CLIPReward
from vca.scoring.metrics import bootstrap_ci,ssim,lpips
from vca.runtime.pipeline import VCARuntime,RuntimeConfig
from vca.runtime.feedback import Feedback
from vca.runtime.logging import write_json,environment

VARIANTS={
    'full':{},'A_no_sesd_rag':{'sesd':False},'B_no_attention':{'attention':False},
    'C_no_ppo':{'ppo':False},'D_no_swap':{'disabled_operations':('Swap',)},
    'E_no_add':{'disabled_operations':('Add',)},'F_no_reweight':{'disabled_operations':('Reweight',)},
    'G_new_priority_reward':{'reward_variant':'new_priority'}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--seeds',type=int,nargs='+',default=[41,42,43])
    p.add_argument('--size',type=int,default=384);p.add_argument('--steps',type=int,default=20)
    p.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS));p.add_argument('--lpips',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);out=a.out or ROOT/'outputs'/('ablation-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True,exist_ok=False);write_json(out/'environment.json',environment(ROOT))
    generator=PairDiffusion(ROOT/'models/sd15');scorer=CLIPReward(ROOT/'models/clip',device='cpu')
    prompt='a mouse sitting on a wooden desk';target='a computer mouse sitting on a wooden desk with a red mouse pad'
    feedbacks=('computer device','add a red mouse pad','emphasize computer mouse');rows=[]
    config={'seeds':a.seeds,'size':a.size,'steps':a.steps,'variants':a.variants,'initial_prompt':prompt,'common_target':target,
        'feedbacks':feedbacks,'data_source':'authored deterministic fixture; no real dialogue dataset used',
        'split':'fixed engineering fixture, no train/validation result claim','confidence_unit':'prompt-seed pair; one prompt, multiple noise seeds',
        'method_basis':'main.pdf only','comparison':'fixed model/revision/DDIM/seed/steps; user semantic types, policy control parameters',
        'C_no_ppo':'same initialized actor with optimizer disabled; no replacement control heuristic',
        'G_new_priority_reward':'explicit implementation variant of Eq14 weights, not paper-faithful'}
    write_json(out/'config.json',config)
    for seed in a.seeds:
        for name in a.variants:
            torch.manual_seed(seed)
            cfg=RuntimeConfig(seed=seed,size=a.size,step_choices=(a.steps,a.steps),initial_steps=a.steps,max_rounds=3,**VARIANTS[name])
            folder=out/name/f'seed_{seed}';runtime=VCARuntime(folder,prompt,generator,scorer,cfg)
            try:
                for feedback in feedbacks:
                    result=runtime.step(Feedback(feedback))
                    if result['status']!='generated':raise RuntimeError(result.get('question'))
                # Evaluate last generated image against one common target across all
                # variants, not each variant's potentially different rewritten text.
                final=Image.open(runtime.session.data['current_image']).convert('RGB')
                initial=Image.open(runtime.session.data['initial_image']).convert('RGB')
                score=scorer.score(final,prompt,target)
                row={'variant':name,'seed':seed,'raw_cosine_target':score['raw_cosine_new'],
                     'raw_cosine_original':score['raw_cosine_previous'],'ssim_initial_final':ssim(initial,final),
                     'lpips_initial_final':lpips(initial,final) if a.lpips else {'value':None,'status':'not_requested'},
                     'rounds':3,'satisfaction':None,'fid':None,'final_image':runtime.session.data['current_image'],
                     'status':'measured','policy_updates':runtime.trainer.version}
                runtime.finalize()
            except Exception as e:
                row={'variant':name,'seed':seed,'status':'FAILED','error':type(e).__name__+': '+str(e)}
                write_json(folder/'error.json',row)
            rows.append(row);write_json(out/'measurements.json',rows)
            print(json.dumps(row,ensure_ascii=False),flush=True)
    summary={}
    for name in a.variants:
        good=[r for r in rows if r['variant']==name and r['status']=='measured']
        summary[name]={metric:bootstrap_ci([r[metric] for r in good],seed=0) for metric in
                       ('raw_cosine_target','raw_cosine_original','ssim_initial_final') if good}
        vals=[r['lpips_initial_final']['value'] for r in good if r['lpips_initial_final'].get('value') is not None]
        if vals:summary[name]['lpips_initial_final']=bootstrap_ci(vals)
    paired={}
    full={r['seed']:r for r in rows if r['variant']=='full' and r['status']=='measured'}
    for name in a.variants:
        deltas=[r['raw_cosine_target']-full[r['seed']]['raw_cosine_target'] for r in rows
                if r['variant']==name and r['status']=='measured' and r['seed'] in full]
        if deltas:paired[name]=bootstrap_ci(deltas)
    report={'summary':summary,'paired_delta_vs_full_raw_cosine':paired,'paper_metrics_reproduced':False,
        'limitations':['one authored prompt, scripted feedback, tiny local PPO updates',
            'CIs describe noise-seed variability of this fixture, not population/user-study uncertainty',
            'no ground-truth target image or adequate reference set: FID not computed',
            'no real human survey: satisfaction null','no claim that 3 rounds improves the real-image policy']}
    write_json(out/'summary.json',report)
    lines=['# 局部消融实测\n\n固定模型、DDIM、步数和seed配对；完整配置见config.json。只使用一个明确标记的脚本样例，不是论文对话数据。\n\n|组|目标CLIP raw cosine (mean)|95% CI|N|SSIM初末|\n|---|---:|---|---:|---:|\n']
    for name,s in summary.items():
        if not s:continue
        clip=s['raw_cosine_target'];lines.append(f"|{name}|{clip['mean']:.6f}|{clip['ci95']}|{clip['n']}|{s['ssim_initial_final']['mean']:.6f}|\n")
    lines.append('\n区间仅表示该样例跨噪声seed的波动。没有真实用户调查，不报告满意度；没有足量参考图，不报告FID。不同于论文绝对指标或全量训练。\n')
    (out/'report.md').write_text(''.join(lines),encoding='utf-8');print('ABLATION COMPLETE '+str(out),flush=True)

if __name__=='__main__':main()
