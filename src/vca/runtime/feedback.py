from dataclasses import dataclass

@dataclass
class Feedback:
    text:str
    rating:float|None=None
    source:str='scripted_fixture'
    satisfied:bool=False

    def __post_init__(self):
        if self.source not in ('scripted_fixture','dataset_annotation','human'):raise ValueError('Feedback source must be explicit')
        if self.rating is not None and (self.source!='human' or not 1<=self.rating<=5):
            raise ValueError('Only actual human input may supply a 1-5 rating')
        if self.satisfied and self.source!='human':raise ValueError('Scripted fixtures cannot fake human satisfaction')
