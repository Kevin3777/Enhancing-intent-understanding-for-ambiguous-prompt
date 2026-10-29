"""Repair missing evaluation metrics from saved images; never rerun generation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import argparse,datetime,hashlib,json
import torch
from PIL import Image
from vca.scoring.metrics import lpips,bootstrap_ci
from vca.runtime.logging import write_json

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=ROOT/'outputs/m5_ablation');a=p.parse_args()
    torch.set_num_threads(4)
    asset=ROOT/'work/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'
    digest=hashlib.file_digest(asset.open('rb'),'sha256').hexdigest()
    if not digest.startswith('7be5be79'):raise ValueError('Cached pretrained metric weights hash invalid')
    rows=read(a.run/'measurements.json');summary=read(a.run/'summary.json')
    backup=ROOT/'work/audit_history'/('metric-repair-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    backup.mkdir(parents=True)
    for name in ('measurements.json','summary.json','report.md'):(backup/name).write_bytes((a.run/name).read_bytes())
    repaired=[];image_hashes={}
    for row in rows:
        if row['status']!='measured':raise ValueError('Cannot metric-repair a failed generation')
        image=Path(row['final_image']);initial=image.parents[1]/'initial.png'
        before=row['lpips_initial_final']
        value=lpips(Image.open(initial).convert('RGB'),Image.open(image).convert('RGB'))
        if value['status']!='measured':raise RuntimeError('Cached metric calculation failed: '+str(value))
        if before.get('status')=='measured' and abs(before['value']-value['value'])>1e-6:
            raise ValueError('Previously measured LPIPS changed unexpectedly')
        if before.get('status')!='measured':repaired.append({'variant':row['variant'],'seed':row['seed'],'previous':before,'new':value})
        row['lpips_initial_final']={**value,'evaluated_from_saved_images':True,'weights_sha256':digest}
        for path in (image,initial):image_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name,stats in summary['summary'].items():
        vals=[r['lpips_initial_final']['value'] for r in rows if r['variant']==name]
        stats['lpips_initial_final']=bootstrap_ci(vals)
    summary['metric_recalculation']={'source':'saved PNGs','weights_sha256':digest,'repaired_missing':len(repaired),'all_lpips_n':3}
    write_json(a.run/'measurements.json',rows);write_json(a.run/'summary.json',summary)
    write_json(a.run/'metric_repair.json',{'weights_sha256':digest,'repaired':repaired,'backup':str(backup),
        'saved_image_sha256':image_hashes,'generation_repeated':False,'invalid_downloaded_weights_never_used':True})
    lines=['# 局部消融实测\n\n固定SD1.5 revision、DDIM、256px、12步及配对seed；8组×3 seed，共24次生成。一个脚本样例，只验证局部机制，不是论文全部实验或真实用户研究。\n\n|组|目标CLIP raw cosine mean|95% CI|SSIM初末 mean|LPIPS初末 mean|N|\n|---|---:|---|---:|---:|---:|\n']
    for name,stats in summary['summary'].items():
        c=stats['raw_cosine_target'];lines.append(f"|{name}|{c['mean']:.6f}|{c['ci95']}|{stats['ssim_initial_final']['mean']:.6f}|{stats['lpips_initial_final']['mean']:.6f}|3|\n")
    lines.append('\n全部指标由保存图像实际计算。早期缺失的LPIPS已用通过哈希验证的本地AlexNet权重补算；原始失败记录保留，见metric_repair.json。未重新生成图像，CLIP/SSIM原值未改动。\n\n区间只描述该样例的噪声seed波动，不证明总体PPO质量提升；满意度没有真实调查而为null，FID缺少足量明确参考数据而未计算。\n')
    (a.run/'report.md').write_text(''.join(lines),encoding='utf-8')
    print(json.dumps({'repaired_lpips':len(repaired),'all_measurements':len(rows),'generation_repeated':False}),flush=True)

if __name__=='__main__':main()
