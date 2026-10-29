import pytest
from PIL import Image
from vca.scoring.metrics import ssim,bootstrap_ci,fid,human_satisfaction

def test_metrics():
    image=Image.new('RGB',(32,32),'red')
    assert ssim(image,image)==1
    assert bootstrap_ci([1])['ci95'] is None
    ci=bootstrap_ci([1,2,3]);assert ci['ci95'][0]<=2<=ci['ci95'][1]
    with pytest.raises(ValueError):fid([image],[])
    assert human_satisfaction([])['mean'] is None
    assert human_satisfaction([{'source':'human','rating':4}])['mean']==4

def test_pretrained_lpips_identity_when_weights_present():
    from pathlib import Path
    from vca.scoring.metrics import lpips
    root=Path(__file__).resolve().parents[1]
    if not (root/'work/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth').exists():
        pytest.skip('Optional pretrained metric weights not cached')
    red=Image.new('RGB',(64,64),'red');blue=Image.new('RGB',(64,64),'blue')
    same=lpips(red,red);changed=lpips(red,blue)
    assert same['status']=='measured' and abs(same['value'])<1e-6
    assert changed['value']>same['value']
