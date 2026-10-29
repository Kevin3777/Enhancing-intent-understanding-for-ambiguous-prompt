from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import argparse,datetime,json
import torch
from vca.editing.pipeline_adapter import PairDiffusion,PairPrompt
from vca.runtime.logging import write_json,environment

def main():
    p=argparse.ArgumentParser();p.add_argument('--prompt',default='a photo of a cat sitting on a wooden table')
    p.add_argument('--size',type=int,default=512);p.add_argument('--steps',type=int,default=20);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',type=Path);a=p.parse_args();torch.set_num_threads(4)
    out=a.out or ROOT/'outputs'/('smoke-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'));out.mkdir(parents=True,exist_ok=False)
    generator=PairDiffusion(ROOT/'models/sd15');torch.cuda.reset_peak_memory_stats()
    result=generator.generate(PairPrompt(a.prompt,a.prompt),seed=a.seed,steps=a.steps,height=a.size,width=a.size)
    result['image'].save(out/'image.png');torch.save(result['initial_latents'],out/'initial_latents.pt')
    write_json(out/'run.json',{'config':vars(a),'environment':environment(ROOT),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()})
    print('SMOKE '+str(out),flush=True)

if __name__=='__main__':main()
