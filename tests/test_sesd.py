import json,pytest
from vca.sesd.dialogue import Dialogue
from vca.sesd.knowledge_base import KnowledgeBase
from vca.sesd.rag import VectorRetriever
from vca.sesd.llm import LLMClient,normalize_optional_question
from vca.sesd.prompt_schema import Proposal

def test_only_literal_null_question_is_normalized():
    value,changed=normalize_optional_question({'question':' null ','new_prompt':'x'})
    assert changed and value['question'] is None
    value,changed=normalize_optional_question({'question':'What does null mean?'})
    assert not changed and value['question']=='What does null mean?'

def test_vector_retrieval():
    r=VectorRetriever(KnowledgeBase());hits=r.search('photorealism skin textures')
    assert hits[0]['id']=='style-photo' and hits[0]['score']>0 and hits[0]['matched_terms']
    assert all(h['source'] for h in hits)
    assert not r.search('zzzzzzxxxx')

@pytest.mark.parametrize('word,feedback,expected',[('mouse','computer device','computer mouse'),('bat','baseball tool','baseball bat'),
    ('bank','river edge','river bank'),('spring','metal coil','metal coil spring'),('spring','water source','water spring')])
def test_polysemy(word,feedback,expected):
    d=Dialogue(f'a {word} in a scene');assert d.refine('').question
    p=d.refine(feedback);assert expected in p.new_prompt and p.type=='Swap' and not p.question
    assert p.evidence

def test_multiturn():
    d=Dialogue('a mouse sitting on a desk')
    assert 'computer mouse' in d.refine('computer device').new_prompt
    added=d.refine('add a red hat');assert 'red hat' in added.new_prompt and added.type=='Add'
    emphasized=d.refine('emphasize red hat');assert emphasized.type=='Reweight'
    conflict=d.refine('actually an animal rodent');assert 'rodent mouse' in conflict.new_prompt
    assert conflict.conflicts[0]['decision']=='latest explicit user intent wins'
    assert 'computer mouse' not in conflict.new_prompt

def test_table1_contract():
    p=Dialogue('a cat on a table').refine('replace cat with dog')
    assert p.type=='Swap' and p.new_prompt=='a dog on a table'
    assert set(('new_prompt','type'))<=set(p.as_dict())

def test_history_provenance():
    kb=KnowledgeBase();r=VectorRetriever(kb)
    kb.append_session('a blue kitten','blue','test',1)
    assert r.search('blue kitten')[0]['source']=='user-session:test:round:1'

def test_schema_fallback():
    class Broken(LLMClient):
        def generate(self,request):return '{"new_prompt":"x","type":"Destroy"}'
    p=Dialogue('a cat on a table',llm=Broken()).refine('add a hat')
    assert p.schema_fallback and p.type=='Add' and 'hat' in p.new_prompt

def test_limits_and_schema():
    d=Dialogue('a cat on a table',max_rounds=1);d.refine('add a hat')
    with pytest.raises(ValueError):d.refine('add a dog')
    with pytest.raises(ValueError):Proposal.from_dict({'new_prompt':'a cat','type':'Reweight','target_phrase':'dog'})

def test_semantic_vector_backend():
    import torch
    kb=KnowledgeBase()
    def embedding(texts):
        return torch.tensor([[1.,0.] if 'photo' in t.lower() else [0.,1.] for t in texts])
    r=VectorRetriever(kb,embedding);hits=r.search('photo')
    assert hits[0]['id']=='style-photo' and hits[0]['score']==1
    assert 'CLIP text embedding' in hits[0]['retrieval_method']

def test_strict_provider_never_silently_falls_back():
    class Broken(LLMClient):
        def generate(self,request):return '{"bad":true}'
    d=Dialogue('a cat on a desk',llm=Broken(),strict_llm=True)
    with pytest.raises((ValueError,TypeError,KeyError)):d.refine('add a hat')

def test_no_rag_provides_no_knowledge():
    class Spy(LLMClient):
        def __init__(self):self.request=None
        def generate(self,request):
            self.request=request
            return json.dumps({'new_prompt':'a cat on a desk with a hat','type':'Add','target_phrase':'hat','question':None})
    spy=Spy();d=Dialogue('a cat on a desk',llm=spy,rag_enabled=False,strict_llm=True)
    result=d.refine('add a hat')
    assert spy.request['evidence']==[] and spy.request['knowledge']==[] and result.evidence==[]

def test_real_provider_ignored_user_sense_is_not_marked_resolved():
    class Ignoring(LLMClient):
        def generate(self,request):
            return json.dumps({'new_prompt':'a rodent mouse on a desk','type':'Swap','target_phrase':'mouse','question':None})
    d=Dialogue('a mouse on a desk',llm=Ignoring(),strict_llm=True)
    r=d.refine('computer device')
    assert r.question and 'mouse' not in d.resolved
    assert d.current=='a mouse on a desk' and r.assumptions

def test_unclarified_model_guess_is_not_committed():
    class Guess(LLMClient):
        def generate(self,request):
            return json.dumps({'new_prompt':'a computer mouse on a desk','type':'Swap','target_phrase':'mouse','question':None})
    d=Dialogue('a mouse on a desk',llm=Guess(),strict_llm=True)
    r=d.refine('')
    assert r.question and r.new_prompt=='a mouse on a desk' and d.current=='a mouse on a desk'
    assert any('deterministic ambiguity guard' in x for x in r.assumptions)

def test_clear_knowledge_alias_is_accepted_without_repeating_ambiguous_word():
    class Alias(LLMClient):
        def generate(self,request):
            return json.dumps({'new_prompt':'a river edge in a landscape','type':'Swap','target_phrase':'river edge','question':None})
    d=Dialogue('a bank in a landscape',llm=Alias(),strict_llm=True)
    r=d.refine('river edge')
    assert r.question is None and d.resolved['bank']=='river bank'

def test_competing_meanings_in_output_are_not_marked_resolved():
    class Mixed(LLMClient):
        def generate(self,request):
            return json.dumps({'new_prompt':'a spring water source in the warm season','type':'Add','target_phrase':'spring','question':None})
    d=Dialogue('a spring in a forest',llm=Mixed(),strict_llm=True)
    r=d.refine('water source')
    assert r.question and 'spring' not in d.resolved
