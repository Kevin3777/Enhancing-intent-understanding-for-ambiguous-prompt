from dataclasses import dataclass,asdict,field

@dataclass
class Proposal:
    new_prompt:str
    type:str
    target_phrase:str=''
    target_occurrence:int=0
    question:str|None=None
    evidence:list=field(default_factory=list)
    conflicts:list=field(default_factory=list)
    backend:str='mock'
    schema_fallback:str|None=None
    assumptions:list=field(default_factory=list)

    def validate(self):
        if not isinstance(self.new_prompt,str) or not self.new_prompt.strip():raise ValueError('Empty new_prompt')
        if self.type not in ('Add','Swap','Reweight'):raise ValueError('Unsupported type')
        if not isinstance(self.target_phrase,str):raise ValueError('target_phrase must be text')
        if not isinstance(self.target_occurrence,int) or self.target_occurrence<0:raise ValueError('Occurrence must be nonnegative integer')
        if self.question is not None and not isinstance(self.question,str):raise ValueError('question must be text/null')
        for name in ('evidence','conflicts','assumptions'):
            if not isinstance(getattr(self,name),list):raise ValueError(f'{name} must be a list')
        if not isinstance(self.backend,str):raise ValueError('backend must be text')
        if self.schema_fallback is not None and not isinstance(self.schema_fallback,str):raise ValueError('schema_fallback must be text/null')
        if self.type=='Reweight' and (not self.target_phrase or self.target_phrase.lower() not in self.new_prompt.lower()):
            raise ValueError('Reweight target must occur in prompt')
        return self

    @classmethod
    def from_dict(cls,data):
        if not isinstance(data,dict):raise ValueError('LLM response must be an object')
        allowed=set(cls.__dataclass_fields__)
        if set(data)-allowed:raise ValueError('Unknown LLM fields')
        return cls(**data).validate()

    def as_dict(self):return asdict(self)
