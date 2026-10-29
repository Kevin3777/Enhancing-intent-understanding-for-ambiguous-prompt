import pytest,torch
from diffusers.models.attention_processor import Attention,AttnProcessor
from vca.editing.token_alignment import align,target_tokens
from vca.editing.controllers import AttentionController,safe_reweight,literal_reweight,InvalidAttention
from vca.editing.pipeline_adapter import EditingAttnProcessor

def probabilities(k=24):return torch.randn(8,3,k).softmax(-1)

@pytest.mark.parametrize('op',['replace','refine','reweight'])
def test_three_operators(tokenizer,op):
    a=align(tokenizer,'a cat','a dog with a hat');x=probabilities()
    c=AttentionController(a,op,target_indices=target_tokens(a.target,'dog'))
    y=c(x,layer='down.attn2',is_cross=True,batch_size=4)
    assert not torch.equal(y[6:],x[6:])
    assert torch.equal(y[:6],x[:6])
    assert torch.allclose(y.sum(-1),torch.ones(8,3),atol=1e-6)
    assert c.capture.trace[0]['source_cached']

@pytest.mark.parametrize('op',['replace','refine','reweight'])
def test_identity(tokenizer,op):
    a=align(tokenizer,'a cat','a cat');x=probabilities();x[6:]=x[4:6]
    c=AttentionController(a,op,target_indices=target_tokens(a.target,'cat'),reweight_c=0)
    y=c(x,layer='mid.attn2',is_cross=True,batch_size=4)
    assert torch.allclose(x,y,atol=1e-7)

def test_negative_literal():
    a=torch.ones(1,1,1,4)/4
    for c in (-1.,1.4):
        with pytest.raises(InvalidAttention):literal_reweight(a,[1],c)
    assert torch.equal(literal_reweight(a,[1],1.),a)
    for c in (-2.,0.,2.):assert torch.allclose(safe_reweight(a,[1],c).sum(-1),torch.ones(1,1,1))

def test_schedule_and_cache(tokenizer):
    a=align(tokenizer,'a cat','a dog');x=probabilities()
    c=AttentionController(a,'replace',steps=2,cross_fraction=.5)
    assert not torch.equal(c(x,layer='x',is_cross=True,batch_size=4),x)
    c.capture.advance();assert not c.capture.cache
    assert torch.equal(c(x,layer='x',is_cross=True,batch_size=4),x)

def test_self_injection(tokenizer):
    a=align(tokenizer,'a cat','a dog');x=probabilities(3)
    c=AttentionController(a,'replace',self_strength=.5)
    y=c(x,layer='mid.attn1',is_cross=False,batch_size=4)
    assert torch.allclose(y[6:],.5*x[4:6]+.5*x[6:])

def test_native_processor_noop(tokenizer):
    module=Attention(query_dim=8,cross_attention_dim=6,heads=2,dim_head=4)
    hidden=torch.randn(4,5,8);context=torch.randn(4,24,6)
    module.set_processor(AttnProcessor());baseline=module(hidden,encoder_hidden_states=context)
    c=AttentionController(align(tokenizer,'a cat','a dog'),'none')
    module.set_processor(EditingAttnProcessor(c,'unit.attn2'))
    edited=module(hidden,encoder_hidden_states=context)
    assert torch.allclose(edited,baseline,atol=1e-7)
    assert len(c.capture.trace)==1

def test_literal_direction(tokenizer):
    c=AttentionController(align(tokenizer,'a cat','a dog'),'replace',mode='paper_literal',replace_source='paper_literal_target')
    x=probabilities();assert torch.equal(c(x,layer='x',is_cross=True,batch_size=4),x)

def test_fp16_identity_is_bitwise(tokenizer):
    a=align(tokenizer,'a cat','a cat');x=probabilities().half();x[6:]=x[4:6]
    c=AttentionController(a,'refine',strength=.7)
    assert torch.equal(c(x,layer='x',is_cross=True,batch_size=4),x)
