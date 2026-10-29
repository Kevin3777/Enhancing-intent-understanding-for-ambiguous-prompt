from pathlib import Path
import os
import numpy as np
from skimage.metrics import structural_similarity

def ssim(reference,generated):
    a=np.asarray(reference.convert('RGB'));b=np.asarray(generated.convert('RGB'))
    if a.shape!=b.shape:raise ValueError('SSIM requires equal image dimensions')
    return float(structural_similarity(a,b,channel_axis=-1,data_range=255))

_LPIPS_MODEL=None

def lpips(reference,generated):
    global _LPIPS_MODEL
    import torch
    try:import lpips as package
    except ImportError:return {'value':None,'status':'BLOCKED','reason':'Optional lpips/torchvision dependencies and pretrained AlexNet weights unavailable'}
    def tensor(image):return torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1).float()[None]/127.5-1
    os.environ.setdefault('TORCH_HOME',str(Path(__file__).resolve().parents[3]/'work/cache/torch'))
    if _LPIPS_MODEL is None:
        try:_LPIPS_MODEL=package.LPIPS(net='alex',verbose=False).eval().requires_grad_(False)
        except (OSError,RuntimeError) as e:return {'value':None,'status':'BLOCKED','reason':'Metric weights unavailable: '+str(e)}
    with torch.inference_mode():value=_LPIPS_MODEL(tensor(reference),tensor(generated)).item()
    return {'value':value,'status':'measured','backbone':'pretrained AlexNet','scale':'LPIPS v0.1'}

def fid(generated,reference,min_samples=1000):
    if not isinstance(generated,(str,Path)) or not isinstance(reference,(str,Path)):
        raise ValueError(f'FID requires explicit generated/reference directories and >= {min_samples} images (implementation-assumption guard)')
    generated,reference=Path(generated),Path(reference)
    if not generated.is_dir() or not reference.is_dir():raise ValueError('FID dataset directory missing')
    count=lambda p:sum(x.suffix.lower() in ('.png','.jpg','.jpeg') for x in p.iterdir())
    if min(count(generated),count(reference))<min_samples:raise ValueError('Insufficient FID images')
    try:import torch_fidelity
    except ImportError:raise RuntimeError('FID dependency/feature weights unavailable')
    import torch
    os.environ.setdefault('TORCH_HOME',str(Path(__file__).resolve().parents[3]/'work/cache/torch'))
    return torch_fidelity.calculate_metrics(input1=str(generated),input2=str(reference),cuda=torch.cuda.is_available(),
        fid=True,isc=False,kid=False,verbose=False)['frechet_inception_distance']

def bootstrap_ci(values,seed=0,repeats=2000):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or len(x)==0 or not np.isfinite(x).all():raise ValueError('Finite nonempty scalar measurements required')
    if len(x)<2:return {'mean':float(x.mean()),'n':len(x),'ci95':None,'reason':'One independent unit has no estimated CI'}
    means=np.random.default_rng(seed).choice(x,(repeats,len(x)),replace=True).mean(1)
    return {'mean':float(x.mean()),'n':len(x),'ci95':np.quantile(means,[.025,.975]).tolist(),
            'method':'percentile bootstrap across independent prompt-seed units'}

def human_satisfaction(feedback):
    ratings=[x['rating'] for x in feedback if x.get('source')=='human' and x.get('rating') is not None]
    if not ratings:return {'mean':None,'n':0,'status':'NOT_REPRODUCED: no real human survey'}
    return {'mean':float(np.mean(ratings)),'n':len(ratings),'status':'actual entered ratings, not controlled user study'}
