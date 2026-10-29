from dataclasses import dataclass,field
import torch

def td_targets(rewards,next_values,dones,gamma):
    if rewards.shape!=next_values.shape or rewards.shape!=dones.shape:raise ValueError('TD vectors must share shape')
    return rewards.detach()+gamma*(1-dones.float())*next_values.detach()

@dataclass
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_logprob: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    next_values: torch.Tensor
    dones: torch.Tensor
    versions: torch.Tensor
    type_masks: torch.Tensor|None=None
    metadata: list=field(default_factory=list)
    consumed: bool=False

    def validate(self):
        n=len(self.states)
        if n==0 or self.states.ndim!=2:raise ValueError('Empty/malformed rollout')
        for name in ('old_logprob','values','rewards','next_values','dones','versions'):
            x=getattr(self,name)
            if x.shape!=(n,):raise ValueError(f'{name} must be [N]')
            if not torch.isfinite(x).all():raise ValueError(f'{name} is not finite')
        if len(self.actions)!=n:raise ValueError('Actions length')
        if not torch.isfinite(self.states).all():raise ValueError('States are not finite')
        if self.type_masks is not None and self.type_masks.shape!=(n,3):raise ValueError('Type masks shape')
        if not ((self.dones==0)|(self.dones==1)).all():raise ValueError('Done mask must be binary')
        if self.metadata and len(self.metadata)!=n:raise ValueError('Metadata length must match rollout')
        return self
