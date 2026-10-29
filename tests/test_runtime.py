"""Offline lifecycle/optimizer test fixtures. Image scores here are explicitly not CLIP."""
from pathlib import Path
from types import SimpleNamespace
import json,torch,pytest
from PIL import Image
from vca.runtime.pipeline import VCARuntime,RuntimeConfig
from vca.runtime.feedback import Feedback
from vca.rl.state_action import hash_text

class FixtureGenerator:
    def __init__(self,tokenizer):self.pipe=SimpleNamespace(tokenizer=tokenizer)
    def generate(self,pair,controller=None,seed=0,steps=8,height=32,width=32):
        rng=torch.Generator().manual_seed(seed)
        x=torch.randn(4,3,24,generator=rng).softmax(-1)
        if controller:
            for i in range(steps):
                y=controller(x,layer='fixture.cross',is_cross=True,batch_size=4);controller.capture.advance()
        else:y=x
        color=(int(sum(pair.edited.encode())%255),int(y[-1].mean()*1000)%255,100)
        return {'image':Image.new('RGB',(height,width),color),'base_image':Image.new('RGB',(height,width),'white'),
                'latents':y,'initial_latents':x}

class FixtureScorer:
    def image_features(self,image):return torch.tensor(list(image.getpixel((0,0)))).float().repeat(6)
    def text_features(self,texts):return torch.stack([hash_text(t,16) for t in texts])
    def score(self,image,previous,new,variant='paper_literal'):
        value=float(sum(image.getpixel((0,0)))/(3*255))
        return {'raw_cosine_previous':value,'raw_cosine_new':value,'reward':value*1.2,
            'variant':variant,'scale':'OFFLINE TEST FIXTURE pixel mean, not CLIP'}

def test_runtime(tokenizer,tmp_path):
    cfg=RuntimeConfig(size=32,step_choices=(2,2),initial_steps=2,max_rounds=3)
    generator=FixtureGenerator(tokenizer);scorer=FixtureScorer()
    r=VCARuntime(tmp_path/'run','a cat on a desk',generator,scorer,cfg)
    for feedback in ('replace cat with dog','add a red hat','emphasize red hat'):
        row=r.step(Feedback(feedback));assert row['status']=='generated'
        assert row['training']['actor_parameter_delta_l2']>0
        assert row['training']['critic_parameter_delta_l2']>0
        assert Path(row['image']).exists() and row['trace']['calls']==2
    summary=r.finalize();assert summary['rounds']==3 and summary['human_satisfaction']['n']==0
    assert r.trainer.version==3
    restored=VCARuntime(tmp_path/'run',None,generator,scorer,cfg)
    assert restored.trainer.version==3 and restored.dialogue.current==r.dialogue.current
    assert len(restored.replay.rows)==3 and restored.dialogue.history[-1]['clip_feedback']['reward']>0
    restored.undo();assert restored.trainer.version==2 and len(restored.session.data['rounds'])==2
    assert len(restored.replay.rows)==2
    row=restored.step(Feedback('emphasize red hat'));assert row['round_id']==4
    assert len(restored.session.data['rolled_back_rounds'])==1

def test_no_fake_human_ratings():
    with pytest.raises(ValueError):Feedback('x',rating=4)
    with pytest.raises(ValueError):Feedback('x',satisfied=True)

def test_clarification_does_not_train(tokenizer,tmp_path):
    r=VCARuntime(tmp_path/'run','a mouse on a desk',FixtureGenerator(tokenizer),FixtureScorer(),RuntimeConfig(size=32,initial_steps=2))
    row=r.step(Feedback(''))
    assert row['status']=='clarification_required' and r.trainer.version==0
