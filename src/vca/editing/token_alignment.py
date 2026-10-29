"""Offset-aware token alignment. Fixed mapper is an ENGINEERING_CHOICE, not Eq24 training."""
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import torch

@dataclass
class TokenizedPrompt:
    text: str
    ids: list[int]
    offsets: list[tuple[int,int]]
    pieces: list[str]
    words: list[str]
    word_tokens: list[list[int]]

def tokenize(tokenizer, text: str) -> TokenizedPrompt:
    if not getattr(tokenizer, 'is_fast', False):
        raise TypeError('A fast tokenizer with offset_mapping is required')
    max_len = tokenizer.model_max_length
    raw = tokenizer(text, add_special_tokens=True, truncation=False)['input_ids']
    if len(raw) > max_len:
        raise ValueError(f'Prompt truncation refused: {len(raw)} tokens > {max_len}')
    enc = tokenizer(text, return_offsets_mapping=True, padding='max_length',
                    max_length=max_len, truncation=False)
    spans = list(re.finditer(r"\w+|[^\w\s]", text, re.UNICODE))
    offsets = [tuple(o) for o in enc['offset_mapping']]
    groups = [[i for i,(a,b) in enumerate(offsets) if b>a and a<m.end() and b>m.start()] for m in spans]
    if any(not x for x in groups):
        raise ValueError('Tokenizer dropped an input word')
    return TokenizedPrompt(text, enc['input_ids'], offsets,
                           tokenizer.convert_ids_to_tokens(enc['input_ids']),
                           [s.group().lower() for s in spans], groups)

@dataclass
class TokenAlignment:
    source: TokenizedPrompt
    target: TokenizedPrompt
    mapper: torch.Tensor   # [Ks,Kt], source-row stochastic
    mapped_target: torch.Tensor  # [Kt], inserted tokens False

    def project(self, source_attention):
        if source_attention.ndim != 4 or source_attention.shape[-1] != self.mapper.shape[0]:
            raise ValueError('Expected [B,H,Q,Ks] attention')
        return torch.einsum('bhqk,kj->bhqj', source_attention.float(),
                            self.mapper.to(source_attention.device, torch.float32))

    def as_dict(self):
        return {'source': self.source.text,'target':self.target.text,
                'source_pieces':self.source.pieces,'target_pieces':self.target.pieces,
                'source_offsets':self.source.offsets,'target_offsets':self.target.offsets,
                'mapped_target':self.mapped_target.tolist(), 'mapper':self.mapper.tolist(),
                'classification':'ENGINEERING_CHOICE: fixed offset-aware mapper'}

def align(tokenizer, source: str, target: str, *, substitutions=True) -> TokenAlignment:
    s,t=tokenize(tokenizer,source),tokenize(tokenizer,target)
    ks,kt=len(s.ids),len(t.ids)
    m=torch.zeros(ks,kt); valid=torch.zeros(kt,dtype=torch.bool)
    if source == target:
        return TokenAlignment(s,t,torch.eye(ks),torch.ones(kt,dtype=torch.bool))
    for op,i0,i1,j0,j1 in SequenceMatcher(a=s.words,b=t.words,autojunk=False).get_opcodes():
        if op=='equal':
            blocks=[(s.word_tokens[i], t.word_tokens[j]) for i,j in zip(range(i0,i1),range(j0,j1))]
        elif op=='replace' and substitutions:
            blocks=[(sum(s.word_tokens[i0:i1],[]),sum(t.word_tokens[j0:j1],[]))]
        else: blocks=[]
        for old,new in blocks:
            for oi in old:
                m[oi,new]=1/len(new)
            valid[new]=True
    # BOS/EOS and every padding row have explicit destinations. Deleted content
    # mass flows to target EOS; no arbitrary out-of-range column is selected.
    ss=[i for i,o in enumerate(s.offsets) if o==(0,0)]
    ts=[i for i,o in enumerate(t.offsets) if o==(0,0)]
    for rank,oi in enumerate(ss):
        nj=ts[min(rank,len(ts)-1)];m[oi,nj]=1;valid[nj]=True
    eos=ts[1] if len(ts)>1 else ts[0]
    for oi in range(ks):
        if m[oi].sum()==0: m[oi,eos]=1
    if not torch.allclose(m.sum(1),torch.ones(ks)):
        raise ValueError('Mapper must conserve each source token mass')
    return TokenAlignment(s,t,m,valid)

def target_tokens(prompt: TokenizedPrompt, phrase: str, occurrence=0) -> list[int]:
    matches=list(re.finditer(r'(?<!\w)'+re.escape(phrase)+r'(?!\w)',prompt.text,re.I))
    if not 0 <= occurrence < len(matches):
        raise ValueError(f'Target phrase/occurrence not found: {phrase!r} #{occurrence}')
    m=matches[occurrence]
    result=[i for i,(a,b) in enumerate(prompt.offsets) if b>a and a<m.end() and b>m.start()]
    if not result: raise ValueError('Target has no non-special tokens')
    return result
