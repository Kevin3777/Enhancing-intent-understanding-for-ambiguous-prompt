"""Explicit finite MDP controls, ENGINEERING_CHOICE for underspecified Eq31-33."""
import hashlib
import torch

ACTION_TYPES=('Add','Swap','Reweight')
HEAD_SIZES=(3,3,3,2,2)
STRENGTHS=(.35,.7,1.)
FRACTIONS=(.35,.7,1.)
SELF_STRENGTHS=(0.,.25)
STATE_DIM=64

def decode_action(action,step_choices=(8,12)):
    a=[int(x) for x in action]
    if len(a)!=5 or any(not 0<=x<n for x,n in zip(a,HEAD_SIZES)):
        raise ValueError('Joint action out of support')
    if len(step_choices)!=2 or any(x<2 for x in step_choices):raise ValueError('Two positive discrete step options required')
    kind=ACTION_TYPES[a[0]]
    return {'type':kind,'operation':{'Add':'refine','Swap':'replace','Reweight':'reweight'}[kind],
            'strength':STRENGTHS[a[1]],'cross_fraction':FRACTIONS[a[2]],
            'self_strength':SELF_STRENGTHS[a[3]],'steps':step_choices[a[4]],
            'reweight_c':2*STRENGTHS[a[1]],'joint_action':a,
            'classification':'ENGINEERING_CHOICE: finite categorical controls'}

def hash_text(text,dim):
    vec=torch.zeros(dim)
    for word in text.lower().split():
        digest=hashlib.sha256(word.encode()).digest()
        vec[int.from_bytes(digest[:4],'little')%dim]+=1 if digest[4]%2 else -1
    return torch.nn.functional.normalize(vec,dim=0) if vec.norm()>0 else vec

def pool_features(x,n):
    x=torch.as_tensor(x).detach().float().reshape(1,1,-1)
    return torch.nn.functional.adaptive_avg_pool1d(x,n).reshape(n).cpu()

def encode_state(image_embedding,prompt_embedding,feedback_embedding,history_text,round_index,last_reward):
    """64D: image16 + current prompt16 + feedback16 + hashed history14 + round/reward2."""
    return torch.cat([pool_features(image_embedding,16),pool_features(prompt_embedding,16),
        pool_features(feedback_embedding,16),hash_text(history_text,14),
        torch.tensor([min(round_index/10,1.),float(last_reward)])])
