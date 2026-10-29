from pathlib import Path
import torch
from transformers import CLIPModel, CLIPProcessor

def cosine(image_features,text_features):
    if image_features.shape[-1]!=text_features.shape[-1]:raise ValueError('Embedding dimensions differ')
    for x in (image_features,text_features):
        if (x.norm(dim=-1)==0).any():raise ValueError('Zero embedding norm')
    return torch.nn.functional.normalize(image_features,dim=-1)@torch.nn.functional.normalize(text_features,dim=-1).T

def dual_reward(prev,new,variant='paper_literal'):
    if variant=='paper_literal':return prev+.2*new
    if variant=='new_priority':return .2*prev+new
    raise ValueError(variant)

class CLIPReward:
    def __init__(self,path,device='cuda'):
        self.device=device
        self.model=CLIPModel.from_pretrained(str(path),local_files_only=True).to(device).eval().requires_grad_(False)
        self.processor=CLIPProcessor.from_pretrained(str(path),local_files_only=True)

    def retrieval_embeddings(self,texts):
        """Reuse frozen PDF CLIP infrastructure for vector retrieval; encoder choice is an assumption."""
        return self.text_features(texts)

    @torch.inference_mode()
    def text_features(self,texts):
        inputs=self.processor(text=texts,return_tensors='pt',padding=True,truncation=True).to(self.device)
        return torch.nn.functional.normalize(self.model.get_text_features(**inputs),dim=-1)

    @torch.inference_mode()
    def image_features(self,image):
        inputs=self.processor(images=image,return_tensors='pt').to(self.device)
        return torch.nn.functional.normalize(self.model.get_image_features(**inputs),dim=-1)

    @torch.inference_mode()
    def score(self,image,previous,new,variant='paper_literal'):
        image_embedding=self.image_features(image)
        text_embedding=self.text_features([previous,new])
        sims=cosine(image_embedding,text_embedding)[0]
        token_counts=[len(self.processor.tokenizer(t,truncation=False)['input_ids']) for t in (previous,new)]
        return {'raw_cosine_previous':sims[0].item(),'raw_cosine_new':sims[1].item(),
                'reward':dual_reward(sims[0].item(),sims[1].item(),variant),'variant':variant,
                'previous_weight':1. if variant=='paper_literal' else .2,
                'new_weight':.2 if variant=='paper_literal' else 1.,'scale':'raw cosine',
                'human_feedback':None,'human_feedback_source':'absent',
                'true_mi':None,'differentiable_generator_optimization':False,
                'text_token_counts':token_counts,'text_truncated':[n>77 for n in token_counts]}
