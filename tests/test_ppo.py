import pytest,torch
from vca.rl.ppo import PPO,PPOConfig,surrogate
from vca.rl.rollout import RolloutBatch,td_targets
from vca.rl.toy import train_toy

def batch_for(p,n=48):
    s=torch.randn(n,64)
    with torch.no_grad():a,lp,v=p.policy.sample(s)
    return RolloutBatch(s,a,lp,v,torch.linspace(0,1,n),torch.ones(n)*.5,
                        torch.arange(n)%2==0,torch.full((n,),p.version))

def test_ratio_and_clipping():
    old=torch.zeros(4,requires_grad=True);new=torch.tensor([2.,2.,-2.,-2.]).log().nan_to_num()
    # Direct ratios [2, 2, .5, .5], with positive/negative advantages.
    new=torch.tensor([2.,2.,.5,.5]).log().requires_grad_();adv=torch.tensor([1.,-1.,1.,-1.])
    obj,ratio=surrogate(new,old,adv)
    assert torch.allclose(ratio,torch.tensor([2.,2.,.5,.5]))
    assert torch.allclose(obj,torch.tensor([1.2,-2.,.5,-.8]))
    (-obj.mean()).backward();assert old.grad is None and new.grad.abs().sum()>0

def test_done_mask():
    target=td_targets(torch.tensor([1.,1.]),torch.tensor([10.,10.]),torch.tensor([True,False]),.9)
    assert torch.equal(target,torch.tensor([1.,10.]))

def test_updates_actor_critic():
    p=PPO(config=PPOConfig(lr=.001));batch=batch_for(p);log=p.update(batch)
    assert log['actor_parameter_delta_l2']>0 and log['critic_parameter_delta_l2']>0
    assert log['optimizer_steps']>0 and all(x['grad_norm']>0 for x in log['losses'])
    assert log['critic_targets'][0]==0
    assert log['critic_targets'][1]==pytest.approx(1/47+.99*.5)

def test_old_policy_guard():
    p=PPO();b=batch_for(p);stale=batch_for(p);p.update(b)
    with pytest.raises(ValueError,match='consumed'):p.update(b)
    with pytest.raises(ValueError,match='Stale'):p.update(stale)

def test_shapes():
    p=PPO();b=batch_for(p);b.old_logprob=b.old_logprob[:,None]
    with pytest.raises(ValueError,match='old_logprob'):p.update(b)

def test_checkpoint_resume(tmp_path):
    p=PPO(config=PPOConfig(lr=.001));p.update(batch_for(p));path=tmp_path/'checkpoint.pt';p.save(path)
    b=batch_for(p);log1=p.update(b)
    q=PPO.load(path);b2=batch_for(q);log2=q.update(b2)
    assert log1==log2
    assert all(torch.equal(v,q.policy.state_dict()[k]) for k,v in p.policy.state_dict().items())

def test_seed_reproducibility():
    torch.manual_seed(33);p=PPO();b=batch_for(p);l=p.update(b)
    torch.manual_seed(33);q=PPO();b2=batch_for(q);assert l==q.update(b2)

def test_toy_improvement():
    p,result=train_toy(seed=7,iterations=25)
    assert result['after']>result['before']+.4 and result['after']>.85

def test_prioritized_replay_used_by_optimizer():
    p=PPO(config=PPOConfig(lr=.001,epochs=2,target_kl=0))
    b=batch_for(p,64);log=p.update(b)
    assert log['replay_mode']=='paper_prioritized_current'
    assert len(log['replay_sampling'])==2
    s=log['replay_sampling'][0];probs=torch.tensor(s['probabilities'])
    assert torch.allclose(probs.sum(),torch.tensor(1.)) and probs.std()>0
    assert len(s['indices'])==64 and len(set(s['indices']))<64
    assert torch.allclose(torch.tensor(s['raw_importance_weights']),(64*probs).pow(-s['beta']))

def test_entropy_only_monitored():
    p=PPO(config=PPOConfig(entropy_coef=.01))
    with pytest.raises(ValueError,match='entropy is monitored'):p.update(batch_for(p))

def test_kl_term_is_weighted_for_priority_sampling():
    from vca.rl.ppo import importance_weighted_kl
    p=torch.tensor([.2,.8]);beta=1.;weights=(2*p).pow(-beta)
    logratio=torch.log(torch.tensor([.5,1.5]));point=logratio.exp()-1-logratio
    weighted_expectation=(p*weights*point).sum()
    assert torch.allclose(weighted_expectation,point.mean())
    assert not torch.isclose((p*point).sum(),point.mean())

def test_mismatched_replay_action_is_rejected():
    from vca.rl.replay import DiagnosticReplay
    p=PPO();b=batch_for(p,4);r=DiagnosticReplay()
    for i in range(4):
        action=b.actions[i].tolist()
        if i==0:action[0]=(action[0]+1)%3
        r.append({'old_logprob':b.old_logprob[i].item(),'policy_version':0,'seed':0,'action':action},1,0)
    with pytest.raises(ValueError,match='action mismatch'):p.update(b,r,list(range(4)),age_clock=0)
