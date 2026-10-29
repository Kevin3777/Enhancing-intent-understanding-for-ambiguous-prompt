import numpy as np

class DiagnosticReplay:
    """Persistent Eq19 pool. PPO uses only matching-version rows, with logprob checks.

    Historical samples remain available for TD/age analysis; sample() alone does
    not authorize them as fresh rollout. PPO.update performs the eligibility check.
    """
    def __init__(self,decay=.01,epsilon=1e-6,capacity=10000):
        if decay<0 or epsilon<=0 or capacity<1:raise ValueError('Replay configuration')
        self.decay=decay;self.epsilon=epsilon;self.capacity=capacity;self.rows=[]

    def append(self,transition,td_error,age_clock):
        for field in ('old_logprob','policy_version','seed','action'):
            if field not in transition:raise ValueError(f'Missing behavior evidence: {field}')
        self.rows.append({'transition':transition,'td_error':float(td_error),'created_at':int(age_clock)})
        self.rows=self.rows[-self.capacity:]

    def probabilities(self,now):
        if not self.rows:raise ValueError('Empty replay')
        ages=np.array([now-r['created_at'] for r in self.rows],dtype=float)
        if (ages<0).any():raise ValueError('Clock precedes stored trajectory')
        priorities=np.exp(-self.decay*ages)*np.abs([r['td_error'] for r in self.rows])+self.epsilon
        return priorities/priorities.sum()

    def sample(self,n,now,beta=.4,seed=0):
        if not 0<=beta<=1 or n<=0:raise ValueError('Replay sample arguments')
        p=self.probabilities(now);indices=np.random.default_rng(seed).choice(len(p),n,p=p,replace=True)
        raw=(len(p)*p)**(-beta);weights=raw/raw.max()
        return {'indices':indices.tolist(),'probabilities':p.tolist(),'importance_weights':weights[indices].tolist(),
                'raw_importance_weights':raw[indices].tolist(),'beta':beta,
                'rows':[self.rows[i] for i in indices],'use':'diagnostic_only',
                'weight_normalization':'max across full buffer (ENGINEERING_CHOICE)'}

    def update_td(self,indices,errors):
        if len(indices)!=len(errors):raise ValueError('Index/error lengths differ')
        for i,e in zip(indices,errors):self.rows[int(i)]['td_error']=float(e)
