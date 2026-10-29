import pytest,torch
from vca.editing.token_alignment import align
from vca.editing.controllers import AttentionController
from vca.rl.mapper_alpha import MapperAlphaPolicy
from vca.rl.ppo import PPO,PPOConfig
from vca.rl.rollout import RolloutBatch

def test_mapping_and_alpha_sample_contract(tokenizer):
    a=align(tokenizer,'a cat on a desk','a dog on a desk');p=MapperAlphaPolicy()
    torch.manual_seed(8);sample=p.sample(a)
    assert sample['mapper'].shape==(24,24)
    assert torch.equal(sample['mapper'].sum(1),torch.ones(24))
    assert 0<sample['alpha']<1
    saved={k:v for k,v in sample.items() if k!='mapper'}
    assert p.evaluate([saved]).item()==pytest.approx(saved['old_logprob'],abs=1e-5)
    assert not saved['clip_gradient_through_diffusion']

def test_joint_ppo_updates_both_parameters(tokenizer,tmp_path):
    torch.manual_seed(3);alignment=align(tokenizer,'a cat on a desk','a dog on a desk')
    trainer=PPO(config=PPOConfig(lr=.001,epochs=2,minibatch=8,target_kl=0),adapter=MapperAlphaPolicy())
    states=torch.randn(8,64);masks=torch.ones(8,3,dtype=torch.bool)
    with torch.no_grad():
        actions,actor_lp,values=trainer.policy.sample(states,masks,(1,))
        samples=[trainer.adapter.sample(alignment) for _ in range(8)]
    adaptive=[{k:v for k,v in s.items() if k!='mapper'} for s in samples]
    old=actor_lp+torch.tensor([s['old_logprob'] for s in samples])
    batch=RolloutBatch(states,actions,old,values,torch.linspace(.1,1,8),torch.zeros(8),
        torch.ones(8,dtype=torch.bool),torch.zeros(8,dtype=torch.long),masks,[{'adaptive':m} for m in adaptive])
    result=trainer.update(batch)
    assert result['mapper_parameter_delta_l2']>0 and result['alpha_parameter_delta_l2']>0
    assert any(x['mapper_grad_norm']>0 and x['alpha_grad_norm']>0 for x in result['losses'])
    path=tmp_path/'adaptive.pt';trainer.save(path);restored=PPO.load(path)
    assert restored.version==1 and restored.config.ignore_strength_head
    assert torch.equal(restored.adapter.mapper_logits,trainer.adapter.mapper_logits)
    assert torch.equal(restored.adapter.alpha_unconstrained,trainer.adapter.alpha_unconstrained)

def test_joint_logprob_does_not_use_unused_strength_head():
    trainer=PPO(config=PPOConfig(ignore_strength_head=True))
    states=torch.randn(3,64);mask=torch.tensor([[True,False,False],[False,True,False],[False,False,True]])
    a,old,_=trainer.policy.sample(states,mask,(1,))
    with torch.no_grad():trainer.policy.heads[1].bias[0]+=1.
    new,_,_=trainer.policy.evaluate(states,a,mask,(1,))
    assert torch.allclose(old[:2],new[:2])
    assert not torch.isclose(old[2],new[2])

def test_unchanged_version_but_modified_policy_is_rejected():
    p=PPO();states=torch.randn(4,64)
    with torch.no_grad():actions,old,values=p.policy.sample(states)
    batch=RolloutBatch(states,actions,old,values,torch.ones(4),torch.zeros(4),torch.ones(4,dtype=torch.bool),torch.zeros(4,dtype=torch.long))
    with torch.no_grad():p.policy.heads[0].bias[0]+=1
    with pytest.raises(ValueError,match='Behavior log-prob'):p.update(batch)

def test_mapper_changes_captured_network_attention(tokenizer):
    alignment=align(tokenizer,'a cat on a desk','a dog on a desk');a=MapperAlphaPolicy()
    prior=a.prior(alignment);source=torch.rand(1,2,7,24).softmax(-1)
    with torch.no_grad():
        initial=(prior.log()+a.mapper_logits[:24,:24]).softmax(-1)
        a.mapper_logits[1,2]+=1
        changed=(prior.log()+a.mapper_logits[:24,:24]).softmax(-1)
    x=torch.einsum('bhqk,kj->bhqj',source,initial)
    y=torch.einsum('bhqk,kj->bhqj',source,changed)
    assert (x-y).abs().max()>0 and torch.allclose(y.sum(-1),torch.ones_like(y[...,0]),atol=1e-6)

def test_end_to_end_fixture_updates_adaptive_controls(tokenizer,tmp_path):
    from test_runtime import FixtureGenerator,FixtureScorer
    from vca.runtime.pipeline import VCARuntime,RuntimeConfig
    from vca.runtime.feedback import Feedback
    cfg=RuntimeConfig(size=32,step_choices=(2,2),initial_steps=2,max_rounds=2,mapper_alpha_mode='learned')
    r=VCARuntime(tmp_path/'adaptive','a cat on a desk',FixtureGenerator(tokenizer),FixtureScorer(),cfg)
    row=r.step(Feedback('replace cat with dog'))
    assert row['training']['mapper_parameter_delta_l2']>0
    assert row['training']['alpha_parameter_delta_l2']>0
    assert row['proposal']['type']=='Swap'
    assert (tmp_path/'adaptive/round_001/mapper_alpha_action.json').exists()
    restored=PPO.load(tmp_path/'adaptive/round_001/checkpoint.pt')
    assert restored.adapter is not None and restored.version==1
