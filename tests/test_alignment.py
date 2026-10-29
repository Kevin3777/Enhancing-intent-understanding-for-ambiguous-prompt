import pytest,torch
from vca.editing.token_alignment import align,tokenize,target_tokens

def test_mapping_shape(tokenizer):
    a=align(tokenizer,'a cat on a desk','a dog on a wooden desk')
    assert a.mapper.shape==(24,24)
    assert torch.allclose(a.mapper.sum(1),torch.ones(24))
    assert not a.mapped_target[target_tokens(a.target,'wooden')].any()

def test_einsum(tokenizer):
    a=align(tokenizer,'a cat','a dog')
    s=torch.arange(2*3*5*24).reshape(2,3,5,24).float()
    assert a.project(s).shape==(2,3,5,24)
    for j in (0,2,5):
        assert torch.allclose(a.project(s)[...,j],(s*a.mapper[:,j]).sum(-1))

def test_subwords_repeats(tokenizer):
    p=tokenize(tokenizer,'a microscope and a microscope')
    first=target_tokens(p,'microscope',0);second=target_tokens(p,'microscope',1)
    assert len(first)==2 and len(second)==2 and set(first).isdisjoint(second)
    assert all(p.offsets[i]!=(0,0) for i in first+second)

def test_identity(tokenizer):
    a=align(tokenizer,'a cat','a cat');assert torch.equal(a.mapper,torch.eye(24))

def test_delete_and_insert(tokenizer):
    for old,new in [('a red cat','a cat'),('a cat','a cat with a red hat'),('a cat cat','a cat dog')]:
        a=align(tokenizer,old,new);assert torch.allclose(a.mapper.sum(1),torch.ones(24))

def test_truncation_refused(tokenizer):
    with pytest.raises(ValueError,match='truncation'):tokenize(tokenizer,'cat '*40)

def test_absent_occurrence(tokenizer):
    p=tokenize(tokenizer,'a cat')
    with pytest.raises(ValueError):target_tokens(p,'cat',2)
