"""Transparent contextual bandit, never reported as CLIP or paper data."""
import torch
from .rollout import RolloutBatch
from .ppo import PPO,PPOConfig

@torch.no_grad()
def expected_success(policy):
    states=torch.zeros(3,policy.state_dim);states[:,:3]=torch.eye(3)
    distributions,_=policy.distributions(states)
    return distributions[0].probs.diag().mean().item()

def train_toy(seed=0,iterations=60,batch_size=96):
    torch.manual_seed(seed);trainer=PPO(config=PPOConfig(lr=.005,epochs=4,minibatch=batch_size,target_kl=.1))
    before=expected_success(trainer.policy);history=[]
    for i in range(iterations):
        labels=torch.randint(3,(batch_size,));states=torch.zeros(batch_size,64)
        states[torch.arange(batch_size),labels]=1
        with torch.no_grad():actions,old,values=trainer.policy.sample(states)
        rewards=(actions[:,0]==labels).float()
        batch=RolloutBatch(states,actions,old,values,rewards,torch.zeros_like(values),
                           torch.ones(batch_size,dtype=torch.bool),torch.full((batch_size,),trainer.version))
        log=trainer.update(batch)
        if i%10==0 or i==iterations-1:
            history.append({'iteration':i,'expected_success':expected_success(trainer.policy),
                            'sample_reward':rewards.mean().item(),'actor_delta':log['actor_parameter_delta_l2'],
                            'critic_delta':log['critic_parameter_delta_l2']})
    return trainer,{'seed':seed,'before':before,'after':expected_success(trainer.policy),'history':history,
        'environment':'3-context 3-action contextual bandit','reward':'1 for matching context, 0 otherwise',
        'classification':'ENGINEERING_CHOICE toy verification, not paper experiment'}
