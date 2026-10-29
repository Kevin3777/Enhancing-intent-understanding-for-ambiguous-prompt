"""Strict on-policy one-step-TD PPO; replay data is diagnostic only."""
from dataclasses import asdict,dataclass
from pathlib import Path
import torch
from .policy import ActorCritic
from .rollout import td_targets

@dataclass
class PPOConfig:
    lr:float=5e-5
    gamma:float=.99
    clip_eps:float=.2
    epochs:int=4
    minibatch:int=64
    value_coef:float=2.2  # p8 numeric value; mapping to MSE coefficient is an implementation-assumption
    entropy_coef:float=0. # monitor entropy; PDF does not specify an entropy optimization objective
    kl_coef:float=.3    # p8 numeric value; mapping to behavior-policy KL is an implementation-assumption
    target_kl:float=.05
    max_grad_norm:float=1.
    normalize_advantage:bool=False  # retain PDF one-step TD advantage without GAE/extra normalization
    replay_mode:str='paper_prioritized_current'
    replay_decay:float=.01
    replay_epsilon:float=1e-6
    replay_beta_start:float=.4
    replay_beta_updates:int=100
    ignore_strength_head:bool=False

def surrogate(new_logprob,old_logprob,advantages,eps=.2):
    ratio=(new_logprob-old_logprob.detach()).exp()
    clipped=ratio.clamp(1-eps,1+eps)
    return torch.minimum(ratio*advantages.detach(),clipped*advantages.detach()),ratio

def importance_weighted_kl(logratio,weights):
    if logratio.shape!=weights.shape:raise ValueError('KL and importance-weight shapes differ')
    return (weights.detach()*((logratio.exp()-1)-logratio)).mean()

class PPO:
    def __init__(self,policy=None,config=None,adapter=None):
        self.policy=policy or ActorCritic();self.config=config or PPOConfig();self.version=0
        self.adapter=adapter
        if adapter is not None:self.config.ignore_strength_head=True
        params=list(self.policy.parameters())+(list(adapter.parameters()) if adapter is not None else [])
        self.optimizer=torch.optim.Adam(params,lr=self.config.lr)

    def update(self,batch,replay_pool=None,replay_indices=None,age_clock=None):
        batch.validate();cfg=self.config
        if cfg.entropy_coef!=0:raise ValueError('PDF-only objective: entropy is monitored, not added as an optimization target')
        if batch.consumed:raise ValueError('Rollout already consumed; recollect from current policy')
        if not (batch.versions==self.version).all():raise ValueError('Stale/off-policy rollout rejected')
        states=batch.states.detach();actions=batch.actions.detach();old=batch.old_logprob.detach()
        ignored=(1,) if cfg.ignore_strength_head else ()
        adaptive=[m.get('adaptive') for m in batch.metadata] if batch.metadata else []
        if self.adapter is not None and (len(adaptive)!=len(states) or any(m is None for m in adaptive)):
            raise ValueError('Adaptive PPO requires saved mapper/alpha actions for every transition')
        if self.adapter is None and any(x is not None for x in adaptive):raise ValueError('Unexpected mapper/alpha actions')
        with torch.no_grad():
            current,_,_=self.policy.evaluate(states,actions,batch.type_masks,ignored)
            if self.adapter is not None:current=current+self.adapter.evaluate(adaptive)
            if not torch.allclose(current,old,atol=1e-4,rtol=0):
                raise ValueError('Behavior log-prob does not match current policy; stale or corrupted rollout')
        targets=td_targets(batch.rewards,batch.next_values,batch.dones,cfg.gamma)
        advantages=(targets-batch.values.detach())
        raw_td=advantages.clone()
        if cfg.normalize_advantage and len(advantages)>1 and advantages.std(unbiased=False)>1e-8:
            advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
        before_actor=torch.cat([p.detach().flatten().clone() for m in (self.policy.actor,self.policy.heads) for p in m.parameters()])
        before_critic=torch.cat([p.detach().flatten().clone() for p in self.policy.critic.parameters()])
        before_type=torch.cat([p.detach().flatten().clone() for p in self.policy.heads[0].parameters()])
        before_mapper=self.adapter.mapper_logits.detach().clone() if self.adapter else None
        before_alpha=self.adapter.alpha_unconstrained.detach().clone() if self.adapter else None
        if cfg.replay_mode not in ('paper_prioritized_current','uniform_on_policy'):raise ValueError('Unsupported replay mode')
        # Eq19 sampling is used inside the same policy-version rollout across PPO
        # epochs. Old-version rows stay in the persistent pool for TD diagnostics;
        # the PDF never defines a correction authorizing arbitrary off-policy reuse.
        created=torch.tensor([float(m.get('age_clock',0)) for m in batch.metadata]) if batch.metadata else torch.zeros(len(states))
        now=float(age_clock if age_clock is not None else created.max().item())
        ages=now-created
        if (ages<0).any():raise ValueError('Replay clock precedes transition')
        beta=min(1.,cfg.replay_beta_start+(1-cfg.replay_beta_start)*self.version/max(cfg.replay_beta_updates,1))
        if replay_pool is not None:
            if replay_indices is None or len(replay_indices)!=len(states):raise ValueError('Replay index correspondence required')
            if len(set(replay_indices))!=len(replay_indices):raise ValueError('Duplicate replay rows in a rollout')
            for i,idx in enumerate(replay_indices):
                stored=replay_pool.rows[idx]['transition']
                if stored['policy_version']!=self.version or abs(stored['old_logprob']-old[i].item())>1e-5:
                    raise ValueError('Replay behavior evidence does not match the current rollout')
                if stored.get('action')!=actions[i].tolist():raise ValueError('Replay action mismatch')
                if 'state' in stored and not torch.allclose(torch.tensor(stored['state']),states[i].cpu(),atol=1e-6,rtol=0):
                    raise ValueError('Replay state mismatch')
                if 'type_mask' in stored and batch.type_masks is not None and stored['type_mask']!=batch.type_masks[i].tolist():
                    raise ValueError('Replay action mask mismatch')
                if 'done' in stored and bool(stored['done'])!=bool(batch.dones[i]):raise ValueError('Replay terminal mask mismatch')
                if 'reward' in stored:
                    r=stored['reward']['reward'] if isinstance(stored['reward'],dict) else stored['reward']
                    if abs(float(r)-batch.rewards[i].item())>1e-6:raise ValueError('Replay reward mismatch')
        logs=[];stopped=False;sampling_logs=[]
        for epoch in range(cfg.epochs):
            with torch.no_grad():
                current_values=self.policy.critic(states).squeeze(-1)
                current_td=targets-current_values
                if replay_pool is not None:replay_pool.update_td(replay_indices,current_td.tolist())
                priority=torch.exp(-cfg.replay_decay*ages)*current_td.abs()+cfg.replay_epsilon
                probabilities=priority/priority.sum()
                if cfg.replay_mode=='paper_prioritized_current':
                    draw=torch.multinomial(probabilities,len(states),replacement=True)
                    importance=(len(states)*probabilities).pow(-beta)
                else:
                    draw=torch.randperm(len(states));importance=torch.ones(len(states))
                sampling_logs.append({'epoch':epoch,'indices':draw.tolist(),'probabilities':probabilities.tolist(),
                    'raw_importance_weights':importance.tolist(),'beta':beta,'ages':ages.tolist(),
                    'eligible_policy_version':self.version,'old_versions_excluded':True})
            for idx in draw.split(cfg.minibatch):
                masks=batch.type_masks[idx] if batch.type_masks is not None else None
                new,entropy,value=self.policy.evaluate(states[idx],actions[idx],masks,ignored)
                if self.adapter is not None:new=new+self.adapter.evaluate([adaptive[int(i)] for i in idx])
                objective,ratio=surrogate(new,old[idx],advantages[idx],cfg.clip_eps)
                logratio=new-old[idx]
                weights=importance[idx].detach()
                approx_kl=importance_weighted_kl(logratio,weights)
                policy_loss=-(weights*objective).mean();value_loss=(weights*(value-targets[idx]).square()).mean()
                loss=policy_loss+cfg.value_coef*value_loss+cfg.kl_coef*approx_kl
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
                self.optimizer.zero_grad(set_to_none=True);loss.backward()
                mapper_grad=float(self.adapter.mapper_logits.grad.norm()) if self.adapter and self.adapter.mapper_logits.grad is not None else 0.
                alpha_grad=float(self.adapter.alpha_unconstrained.grad.norm()) if self.adapter and self.adapter.alpha_unconstrained.grad is not None else 0.
                all_params=list(self.policy.parameters())+(list(self.adapter.parameters()) if self.adapter else [])
                grad=float(torch.nn.utils.clip_grad_norm_(all_params,cfg.max_grad_norm))
                if not torch.isfinite(torch.tensor(grad)):raise FloatingPointError('Nonfinite PPO gradient')
                self.optimizer.step()
                logs.append({'loss':loss.item(),'policy_loss':policy_loss.item(),'value_loss':value_loss.item(),
                    'entropy':entropy.mean().item(),'approx_kl':approx_kl.item(),'grad_norm':grad,
                    'mapper_grad_norm':mapper_grad,'alpha_grad_norm':alpha_grad,
                    'clip_fraction':((ratio-1).abs()>cfg.clip_eps).float().mean().item(),'epoch':epoch})
                if cfg.target_kl and approx_kl.item()>cfg.target_kl:stopped=True;break
            if stopped:break
        after_actor=torch.cat([p.detach().flatten() for m in (self.policy.actor,self.policy.heads) for p in m.parameters()])
        after_critic=torch.cat([p.detach().flatten() for p in self.policy.critic.parameters()])
        after_type=torch.cat([p.detach().flatten() for p in self.policy.heads[0].parameters()])
        batch.consumed=True;old_version=self.version;self.version+=1
        return {'old_policy_version':old_version,'policy_version':self.version,'optimizer_steps':len(logs),
                'actor_parameter_delta_l2':float((after_actor-before_actor).norm()),
                'critic_parameter_delta_l2':float((after_critic-before_critic).norm()),
                'actor_type_head_delta_l2':float((after_type-before_type).norm()),
                'mapper_parameter_delta_l2':float((self.adapter.mapper_logits.detach()-before_mapper).norm()) if self.adapter else None,
                'alpha_parameter_delta_l2':float((self.adapter.alpha_unconstrained.detach()-before_alpha).norm()) if self.adapter else None,
                'td_errors':raw_td.tolist(),'critic_targets':targets.tolist(),'early_stop_kl':stopped,
                'losses':logs,'advantage':'one_step_TD','replay_mode':cfg.replay_mode,'replay_sampling':sampling_logs,
                'loss_classification':'implementation-assumption: p8 value/KL coefficients mapped to value MSE / behavior KL; no entropy objective'}

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'policy':self.policy.state_dict(),'optimizer':self.optimizer.state_dict(),
            'adapter_state':self.adapter.state_dict() if self.adapter else None,
            'adapter_config':{'max_tokens':self.adapter.max_tokens,'exploration_mass':self.adapter.exploration_mass} if self.adapter else None,
            'version':self.version,'config':asdict(self.config),'state_dim':self.policy.state_dim,
            'hidden':self.policy.hidden,'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},path)

    @classmethod
    def load(cls,path,restore_rng=True):
        data=torch.load(path,map_location='cpu',weights_only=True)
        adapter=None
        if data.get('adapter_state') is not None:
            from .mapper_alpha import MapperAlphaPolicy
            adapter=MapperAlphaPolicy(**data['adapter_config']);adapter.load_state_dict(data['adapter_state'])
        obj=cls(ActorCritic(data['state_dim'],data['hidden']),PPOConfig(**data['config']),adapter)
        obj.policy.load_state_dict(data['policy']);obj.optimizer.load_state_dict(data['optimizer']);obj.version=data['version']
        if restore_rng:
            torch.set_rng_state(data['torch_rng'])
            if data['cuda_rng'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(data['cuda_rng'])
        return obj
