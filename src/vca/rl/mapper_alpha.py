"""Opt-in PPO score-function estimator for PDF Eq22-27 mapper/beta.

This is an implementation assumption, not an end-to-end derivative through
the frozen diffusion or CLIP image. Mapper rows sample discrete assignments;
the refinement blend coefficient is a sampled Beta random variable.
"""
import torch
from torch import nn
from torch.distributions import Categorical,Beta

class MapperAlphaPolicy(nn.Module):
    def __init__(self,max_tokens=77,exploration_mass=.05):
        super().__init__()
        if max_tokens<2 or not 0<exploration_mass<1:raise ValueError('Invalid mapper configuration')
        self.max_tokens=max_tokens;self.exploration_mass=exploration_mass
        self.mapper_logits=nn.Parameter(torch.zeros(max_tokens,max_tokens))
        # softplus(5.9975)+1 ~= 7; softplus(1.8546)+1 ~= 3; mean ~= .7
        self.alpha_unconstrained=nn.Parameter(torch.tensor([5.9975,1.8546]))

    def prior(self,alignment):
        base=alignment.mapper.detach().float().cpu()
        ks,kt=base.shape
        if ks>self.max_tokens or kt>self.max_tokens:raise ValueError('Alignment exceeds configured mapper length')
        source_content=torch.tensor([b>a for a,b in alignment.source.offsets],dtype=torch.bool)
        target_content=torch.tensor([b>a for a,b in alignment.target.offsets],dtype=torch.bool)
        target_allowed=target_content & alignment.mapped_target.cpu()
        out=base.clone()
        if target_allowed.any():
            uniform=target_allowed.float()/target_allowed.sum()
            out[source_content]=(1-self.exploration_mass)*base[source_content]+self.exploration_mass*uniform
        if not torch.allclose(out.sum(-1),torch.ones(ks),atol=1e-6):raise ValueError('Invalid mapper prior')
        return out

    def _dists(self,priors):
        if priors.ndim==2:priors=priors.unsqueeze(0)
        if priors.ndim!=3 or priors.shape[-2]>self.max_tokens or priors.shape[-1]>self.max_tokens:
            raise ValueError('Mapper priors must be [N,Ks,Kt]')
        if (priors<0).any() or not torch.isfinite(priors).all() or not torch.allclose(priors.sum(-1),torch.ones_like(priors[...,0]),atol=1e-5):
            raise ValueError('Mapper priors must be finite row probabilities')
        logits=priors.log()+self.mapper_logits[:priors.shape[-2],:priors.shape[-1]]
        ab=torch.nn.functional.softplus(self.alpha_unconstrained)+1
        return Categorical(logits=logits),Beta(ab[0],ab[1])

    @torch.no_grad()
    def sample(self,alignment):
        prior=self.prior(alignment);mapper_dist,alpha_dist=self._dists(prior)
        indices=mapper_dist.sample()[0];alpha=alpha_dist.sample()
        mapper=torch.nn.functional.one_hot(indices,num_classes=prior.shape[1]).float()
        logp=mapper_dist.log_prob(indices.unsqueeze(0)).sum(-1)[0]+alpha_dist.log_prob(alpha)
        return {'active':True,'prior':prior.tolist(),'mapper_indices':indices.tolist(),'alpha':alpha.item(),
                'old_logprob':logp.item(),'mapper':mapper,
                'method':'PPO score-function on sampled mapper and beta (implementation-assumption)',
                'clip_gradient_through_diffusion':False}

    @torch.no_grad()
    def mean_controls(self,alignment):
        """Deterministic expected mapping/beta for isolated before/after evaluation."""
        prior=self.prior(alignment);mapping,alpha=self._dists(prior)
        return mapping.probs[0].cpu(),float(alpha.mean)

    def evaluate(self,metadata):
        """Log probability of saved actions; no gradient through the sampled image."""
        if any(not m.get('active',True) for m in metadata):
            values=[]
            for m in metadata:
                values.append(self.evaluate([m])[0] if m.get('active',True) else
                    (self.mapper_logits.sum()+self.alpha_unconstrained.sum())*0.)
            return torch.stack(values)
        priors=torch.tensor([m['prior'] for m in metadata],dtype=self.mapper_logits.dtype,device=self.mapper_logits.device)
        indices=torch.tensor([m['mapper_indices'] for m in metadata],dtype=torch.long,device=self.mapper_logits.device)
        alphas=torch.tensor([m['alpha'] for m in metadata],dtype=self.mapper_logits.dtype,device=self.mapper_logits.device)
        if (alphas<=0).any() or (alphas>=1).any():raise ValueError('Saved alpha outside Beta support')
        if indices.shape!=priors.shape[:2]:raise ValueError('Mapper action shape mismatch')
        md,ad=self._dists(priors)
        return md.log_prob(indices).sum(-1)+ad.log_prob(alphas)
