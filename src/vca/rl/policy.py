import torch
from torch import nn
from torch.distributions import Categorical
from .state_action import HEAD_SIZES,STATE_DIM

class ActorCritic(nn.Module):
    def __init__(self,state_dim=STATE_DIM,hidden=64):
        super().__init__();self.state_dim=state_dim;self.hidden=hidden
        self.actor=nn.Sequential(nn.Linear(state_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.heads=nn.ModuleList(nn.Linear(hidden,n) for n in HEAD_SIZES)
        self.critic=nn.Sequential(nn.Linear(state_dim,hidden),nn.Tanh(),nn.Linear(hidden,1))

    def distributions(self,states,type_mask=None):
        if states.ndim!=2 or states.shape[1]!=self.state_dim:raise ValueError('State batch shape')
        h=self.actor(states);logits=[head(h) for head in self.heads]
        if type_mask is not None:
            if type_mask.shape!=logits[0].shape or not type_mask.bool().any(-1).all():
                raise ValueError('Invalid/empty type mask')
            logits[0]=logits[0].masked_fill(~type_mask.bool(),float('-inf'))
        return [Categorical(logits=x) for x in logits],self.critic(states).squeeze(-1)

    def sample(self,states,type_mask=None,ignored_heads=()):
        dists,values=self.distributions(states,type_mask)
        actions=torch.stack([d.sample() for d in dists],-1)
        for i in ignored_heads:
            if i==1:actions[:,i]=torch.where(actions[:,0]==2,actions[:,i],torch.zeros_like(actions[:,i]))
            else:actions[:,i]=0
        logprob=sum(d.log_prob(actions[:,i]) for i,d in enumerate(dists) if i not in ignored_heads)
        # Head 1 controls reweighting magnitude for Reweight only. For Add/Swap
        # the independently sampled Beta or fixed alpha supplies the blend.
        if 1 in ignored_heads:logprob=logprob+torch.where(actions[:,0]==2,dists[1].log_prob(actions[:,1]),torch.zeros_like(logprob))
        return actions,logprob,values

    def evaluate(self,states,actions,type_mask=None,ignored_heads=()):
        if actions.shape!=(len(states),len(HEAD_SIZES)):raise ValueError('Action batch shape')
        dists,values=self.distributions(states,type_mask)
        logprob=sum(d.log_prob(actions[:,i]) for i,d in enumerate(dists) if i not in ignored_heads)
        entropy=sum(d.entropy() for i,d in enumerate(dists) if i not in ignored_heads)
        if 1 in ignored_heads:
            active=actions[:,0]==2
            logprob=logprob+torch.where(active,dists[1].log_prob(actions[:,1]),torch.zeros_like(logprob))
            entropy=entropy+torch.where(active,dists[1].entropy(),torch.zeros_like(entropy))
        return logprob,entropy,values
