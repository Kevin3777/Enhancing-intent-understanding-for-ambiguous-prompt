"""M1: real same-target/no-edit controls, source cache proof, identical latents."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import argparse,json,datetime,hashlib
import numpy as np
import torch
from PIL import Image,ImageDraw
from vca.editing.pipeline_adapter import PairDiffusion,PairPrompt
from vca.editing.controllers import AttentionController
from vca.editing.token_alignment import align,target_tokens
from vca.scoring.clip_reward import CLIPReward

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=12);p.add_argument('--size',type=int,default=256)
    p.add_argument('--seed',type=int,default=20261027);p.add_argument('--out',type=Path);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.out or ROOT/'outputs'/('attention-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    config={**vars(a),'out':str(out),'model':json.loads((ROOT/'models/sd15/vca_manifest.json').read_text()),
            'steps':a.steps,'size':a.size,'seed':a.seed,'classification':'partial_experiment_replication',
            'mode':'safe_attention','equations':[21,23,26,27,28],'scheduler':'DDIM'}
    (out/'config.json').write_text(json.dumps(config,indent=2,default=str))
    try:
        pipe=PairDiffusion(ROOT/'models/sd15')
        scorer=CLIPReward(ROOT/'models/clip',device='cpu')
        torch.cuda.reset_peak_memory_stats()
        latents=pipe.initial_latents(a.seed,a.size,a.size);torch.save(latents,out/'initial_latents.pt')
        base='a photo of a cat sitting on a wooden table'
        cases=[('replace','a photo of a dog sitting on a wooden table','dog'),
               ('refine','a photo of a cat with a red hat sitting on a wooden table','red hat'),
               ('reweight',base,'cat')]
        report=[];tiles=[]
        for op,new,phrase in cases:
            pair=PairPrompt(base,new);alignment=align(pipe.pipe.tokenizer,base,new)
            official=pipe.generate(pair,seed=a.seed,steps=a.steps,height=a.size,width=a.size,latents=latents)
            noop_c=AttentionController(alignment,'none',steps=a.steps)
            noop=pipe.generate(pair,noop_c,seed=a.seed,steps=a.steps,height=a.size,width=a.size,latents=latents)
            noop_err=(official['latents']-noop['latents']).abs().max().item()
            if noop_err>5e-3:raise AssertionError(f'No-op deviates from official AttnProcessor: {noop_err}')
            controller=AttentionController(alignment,op,steps=a.steps,target_indices=target_tokens(alignment.target,phrase),
                            cross_fraction=.7,strength=.7,reweight_c=1.2)
            edited=pipe.generate(pair,controller,seed=a.seed,steps=a.steps,height=a.size,width=a.size,latents=latents)
            folder=out/op;folder.mkdir()
            for name,result in [('no_edit',official),('edited',edited)]:result['image'].save(folder/(name+'.png'))
            edited['base_image'].save(folder/'source.png')
            torch.save({'no_edit':official['latents'],'edited':edited['latents']},folder/'final_latents.pt')
            trace=controller.capture.save(folder)
            (folder/'alignment.json').write_text(json.dumps(alignment.as_dict(),indent=2))
            latent_delta=float((edited['latents'][1]-official['latents'][1]).norm())
            base_unchanged=torch.equal(edited['latents'][0],official['latents'][0])
            assert base_unchanged and latent_delta>1e-4 and trace['injections']>0
            row={'operator':op,'same_target_prompt_control':True,'no_op_max_latent_error':noop_err,
                 'source_latents_unchanged':base_unchanged,'attention_only_latent_delta_l2':latent_delta,
                 'pixel_mae':float(np.abs(np.asarray(edited['image']).astype(float)-np.asarray(official['image'])).mean()),
                 'trace':trace,'reward_no_edit':scorer.score(official['image'],base,new),
                 'reward_edited':scorer.score(edited['image'],base,new)}
            report.append(row);tiles.append((op,official['image'],edited['image']))
            print(json.dumps(row),flush=True)
        # Identical prompt and identical branch latents: replace/refine become identity.
        alignment=align(pipe.pipe.tokenizer,base,base)
        identity_c=AttentionController(alignment,'refine',steps=a.steps)
        identity=pipe.generate(PairPrompt(base,base),identity_c,seed=a.seed,steps=a.steps,height=a.size,width=a.size,latents=latents)
        identity_error=float((identity['latents'][0]-identity['latents'][1]).abs().max())
        assert identity_error<5e-3, f'Identical prompt latent error: {identity_error}'
        sheet=Image.new('RGB',(a.size*2,3*(a.size+30)),(248,248,248));draw=ImageDraw.Draw(sheet)
        for i,(op,raw,img) in enumerate(tiles):
            y=i*(a.size+30);draw.text((8,y+7),f'{op}: same new prompt, left=no edit / right=attention edit',fill='black')
            sheet.paste(raw,(0,y+30));sheet.paste(img,(a.size,y+30))
        sheet.save(out/'comparison.png')
        result={'status':'passed','operators':report,'identical_prompt_max_error':identity_error,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                'claim':'Three operators affect real frozen SD1.5 sampling; quality superiority not inferred.'}
        (out/'summary.json').write_text(json.dumps(result,indent=2));print('M1 PASS '+str(out),flush=True)
    except Exception as e:
        (out/'error.json').write_text(json.dumps({'status':'FAILED','type':type(e).__name__,'message':str(e)},indent=2));raise

if __name__=='__main__':main()
