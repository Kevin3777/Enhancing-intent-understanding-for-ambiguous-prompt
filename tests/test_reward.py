import pytest,torch
from vca.scoring.clip_reward import cosine,dual_reward

def test_cosine():
    assert torch.allclose(cosine(torch.tensor([[2.,0.]]),torch.tensor([[1.,0.],[-1.,0.]])),torch.tensor([[1.,-1.]]))
    with pytest.raises(ValueError):cosine(torch.zeros(1,2),torch.ones(1,2))

def test_weights():
    assert dual_reward(.8,.3)==pytest.approx(.86)
    assert dual_reward(.8,.3,'new_priority')==pytest.approx(.46)
