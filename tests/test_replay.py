import pytest,numpy as np
from vca.rl.replay import DiagnosticReplay

def replay():
    r=DiagnosticReplay(decay=.2)
    for i,td in enumerate((1.,2.,0.)):
        r.append({'old_logprob':-.5,'policy_version':i,'seed':0,'action':[0,0,0,0,0]},td,i)
    return r

def test_normalized_sampling():
    r=replay();p=r.probabilities(5)
    raw=np.exp(-.2*np.array([5,4,3]))*np.array([1,2,0])+1e-6
    assert np.allclose(p,raw/raw.sum()) and p.min()>0
    sample=r.sample(1000,5,beta=.7,seed=0)
    expected=(3*p)**(-.7)
    assert np.allclose(sample['raw_importance_weights'],expected[sample['indices']])
    assert sample['use']=='diagnostic_only' and max(sample['importance_weights'])<=1

def test_td_updates():
    r=replay();before=r.probabilities(5);r.update_td([0],[100]);assert r.probabilities(5)[0]>before[0]
    assert r.sample(12,6,seed=2)==r.sample(12,6,seed=2)

def test_behavior_required():
    with pytest.raises(ValueError):DiagnosticReplay().append({},1,0)
