from pathlib import Path
import hashlib, json
import torch

class BaseAttentionCapture:
    """Cache by (sampling step, UNet layer), release GPU maps at each step."""
    def __init__(self):
        self.cache={};self.trace=[];self.samples={};self.step=0;self.timesteps=[]

    def store(self, layer, source, is_cross):
        key=(self.step,layer)
        self.cache[key]=source.detach().clone()
        if is_cross and source.shape[-2]<=64 and self.step in (0,1):
            self.samples[f'{self.step}:{layer}']=source.detach().cpu()
        return key

    def advance(self):
        self.cache.clear();self.step+=1

    def save(self, folder):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        with (folder/'attention_trace.jsonl').open('w',encoding='utf-8') as f:
            for row in self.trace:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        torch.save(self.samples,folder/'source_attention_samples.pt')
        return {'calls':len(self.trace),'injections':sum(x['delta_l1']>0 for x in self.trace),
                'cross_calls':sum(x['is_cross'] for x in self.trace)}
