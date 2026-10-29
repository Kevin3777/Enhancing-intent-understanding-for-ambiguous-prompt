import pytest,torch
from vca.scoring.mi import update,rank_one_proxy,gaussian_mi

def test_direction():
    assert update(1.,2.,.1,'paper_literal')==.8
    assert update(1.,2.,.1,'mi_ascent')==1.2

def test_gaussian_domain():
    rho=.3
    mi=gaussian_mi(torch.eye(1),torch.eye(1),torch.tensor([[rho]]))
    assert torch.allclose(mi,rank_one_proxy(rho))
    with pytest.raises(ValueError):rank_one_proxy(1.)
    with pytest.raises(torch.linalg.LinAlgError):gaussian_mi(torch.eye(1),torch.eye(1),torch.tensor([[2.]]))

def test_not_clip_mi():
    assert torch.allclose(rank_one_proxy(.4),rank_one_proxy(-.4))
    assert rank_one_proxy(.4)!=torch.tensor(.4)
