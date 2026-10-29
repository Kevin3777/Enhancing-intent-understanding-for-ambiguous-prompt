import pytest,torch
from vca.rl.policy import ActorCritic
from vca.rl.state_action import decode_action,encode_state

def test_sample_logprob():
    p=ActorCritic();s=torch.randn(9,64);actions,old,values=p.sample(s)
    new,entropy,v=p.evaluate(s,actions)
    assert actions.shape==(9,5) and values.shape==(9,)
    assert torch.allclose(old,new) and torch.allclose(values,v)
    assert torch.all(entropy>0)

def test_mask_persisted():
    p=ActorCritic();s=torch.randn(4,64);mask=torch.tensor([[False,True,False]]*4)
    actions,old,_=p.sample(s,mask)
    assert (actions[:,0]==1).all()
    assert torch.allclose(p.evaluate(s,actions,mask)[0],old)
    with pytest.raises(ValueError):p.sample(s,torch.zeros(4,3,dtype=torch.bool))

def test_controls_discrete():
    a=decode_action([2,1,2,1,0],(8,12))
    assert a['type']=='Reweight' and a['steps']==8 and isinstance(a['steps'],int)
    assert a['self_strength']==.25
    with pytest.raises(ValueError):decode_action([3,0,0,0,0])

def test_state():
    s=encode_state(torch.ones(512),torch.ones(512),torch.ones(512),'a cat',2,.4)
    assert s.shape==(64,) and torch.isfinite(s).all()
