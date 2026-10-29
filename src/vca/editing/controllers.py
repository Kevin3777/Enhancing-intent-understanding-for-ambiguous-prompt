"""Dual-branch attention edits. Literal mathematical conflicts are explicit errors."""
import torch
from .attention_capture import BaseAttentionCapture

class InvalidAttention(ValueError):pass

def validate(a, normalized=True):
    if not torch.isfinite(a).all() or (a<0).any():
        raise InvalidAttention('Attention contains negative or nonfinite probabilities (Eq28 conflict)')
    if normalized and not torch.allclose(a.float().sum(-1),torch.ones_like(a[...,0]).float(),atol=2e-3,rtol=0):
        raise InvalidAttention('Attention row sum is not one (paper literal has no normalization)')

def normalize(a):
    if (a<0).any() or not torch.isfinite(a).all():raise InvalidAttention('Cannot normalize invalid probabilities')
    denom=a.float().sum(-1,keepdim=True)
    if (denom<=0).any():raise InvalidAttention('Zero attention row')
    return a.float()/denom

def literal_reweight(a, indices, c):
    if not -2<=c<=2:raise ValueError('Paper Eq28 requires c in [-2,2]')
    out=a.clone();out[...,indices]*=c
    validate(out)  # intentionally raises; no silent clamp or repair
    return out

def safe_reweight(a, indices, c):
    """RESEARCH_VARIANT: c is additive logit bias, not paper's multiplier."""
    if not -2<=c<=2:raise ValueError('Safe bias must be in [-2,2]')
    if c==0:return a.clone()  # exact identity avoids half-precision renormalization drift
    logits=a.float().log()  # zeros remain -inf, no clamp changing the formula
    logits=logits.clone();logits[...,indices]+=c
    out=logits.softmax(-1);validate(out)
    return out.to(a.dtype)

class AttentionController:
    def __init__(self, alignment, operation='none', *, steps=12, cross_fraction=0.7,
                 strength=1.0, target_indices=(), reweight_c=1.2, self_strength=0.0,
                 self_max_q=256, mode='safe_attention', replace_source='base_to_edited_assumption',
                 record_target_samples=False):
        if operation not in ('none','replace','refine','reweight'):raise ValueError(operation)
        if mode not in ('safe_attention','paper_literal'):raise ValueError(mode)
        if replace_source not in ('base_to_edited_assumption','p2p_source','paper_literal_target'):raise ValueError(replace_source)
        for name,x in [('cross_fraction',cross_fraction),('strength',strength),('self_strength',self_strength)]:
            if not 0<=x<=1:raise ValueError(name)
        self.alignment=alignment;self.operation=operation;self.steps=steps
        self.cross_fraction=cross_fraction;self.strength=strength;self.target_indices=list(target_indices)
        self.c=reweight_c;self.self_strength=self_strength;self.self_max_q=self_max_q
        self.mode=mode;self.replace_source=replace_source;self.capture=BaseAttentionCapture()
        self.record_target_samples=record_target_samples;self.target_samples={}

    def __call__(self, probs, *, layer, is_cross, batch_size):
        if batch_size != 4 or probs.shape[0]%4:
            raise ValueError('Expected CFG ordering [uncond_base,uncond_target,cond_base,cond_target]')
        h=probs.shape[0]//4
        a=probs.reshape(4,h,*probs.shape[1:]);source=a[2:3];target=a[3:4]
        step=self.capture.step;active=step < self.steps*self.cross_fraction
        key=self.capture.store(layer,source,is_cross)
        if self.record_target_samples and is_cross and target.shape[-2]<=64 and step in (0,1):
            self.target_samples[f'{step}:{layer}']=target.detach().cpu()
        candidate=target
        if self.operation!='none' and active:
            if is_cross:
                mapped=self.alignment.project(self.capture.cache[key])
                mask=self.alignment.mapped_target.to(probs.device).view(1,1,1,-1)
                if self.operation=='replace':
                    if self.replace_source in ('base_to_edited_assumption','p2p_source'): candidate=mapped
                    else: candidate=target  # literal M* interpreted as updated target; ambiguity retained
                elif self.operation in ('refine','reweight'):
                    candidate=torch.where(mask,mapped,target.float())
                candidate=self.strength*candidate+(1-self.strength)*target.float()
                # If the proposed edit is already identical in the actual attention
                # dtype, preserve the original softmax values bit-for-bit. Repeated
                # fp32 normalization of fp16 probabilities can otherwise create an
                # unintended edit for identical prompt/latent branches.
                if self.mode=='safe_attention' and not torch.equal(candidate.to(target.dtype),target):
                    candidate=normalize(candidate)
                if self.operation=='reweight':
                    if not self.target_indices:raise ValueError('Reweight needs explicit target token indices')
                    candidate=(safe_reweight(candidate,self.target_indices,self.c) if self.mode=='safe_attention'
                               else literal_reweight(candidate,self.target_indices,self.c))
            elif target.shape[-2]<=self.self_max_q:
                candidate=self.self_strength*source.float()+(1-self.self_strength)*target.float()
            candidate=candidate.to(target.dtype)
        validate(candidate)
        delta=(candidate.float()-target.float()).abs().mean().item()
        out=a.clone();out[3:4]=candidate
        raw_t=self.capture.timesteps[step] if step<len(self.capture.timesteps) else None
        self.capture.trace.append({'step_index':step,'diffusion_timestep':raw_t,'layer':layer,'is_cross':is_cross,
            'shape':list(a.shape),'branch_order':['uncond_base','uncond_target','cond_base','cond_target'],
            'source_branch':'cond_base','destination_branch':'cond_target','source_cached':key in self.capture.cache,
            'cache_key':f'{key[0]}:{key[1]}','operation':self.operation,'active':active,
            'source_interpretation':self.replace_source,'mode':self.mode,'delta_l1':delta,
            'row_sum_max_error':float((candidate.float().sum(-1)-1).abs().max()),
            'self_skipped_q_limit':bool(not is_cross and target.shape[-2]>self.self_max_q)})
        return out.reshape_as(probs)
