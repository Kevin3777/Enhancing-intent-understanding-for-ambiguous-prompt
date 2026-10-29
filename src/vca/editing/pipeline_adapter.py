"""Diffusers 0.32.2 native AttentionProcessor path; edits occur in UNet forward."""
from dataclasses import dataclass
from pathlib import Path
import json
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor
from transformers import CLIPTokenizerFast
from .token_alignment import align

class EditingAttnProcessor:
    def __init__(self,controller,layer):self.controller=controller;self.layer=layer

    def __call__(self,attn,hidden_states,encoder_hidden_states=None,attention_mask=None,temb=None,*args,**kwargs):
        residual=hidden_states
        if attn.spatial_norm is not None:hidden_states=attn.spatial_norm(hidden_states,temb)
        ndim=hidden_states.ndim
        if ndim==4:
            b,c,h,w=hidden_states.shape
            hidden_states=hidden_states.view(b,c,h*w).transpose(1,2)
        batch_size=hidden_states.shape[0]
        seq_len=hidden_states.shape[1] if encoder_hidden_states is None else encoder_hidden_states.shape[1]
        attention_mask=attn.prepare_attention_mask(attention_mask,seq_len,batch_size)
        if attn.group_norm is not None:hidden_states=attn.group_norm(hidden_states.transpose(1,2)).transpose(1,2)
        query=attn.to_q(hidden_states)
        is_cross=encoder_hidden_states is not None
        if encoder_hidden_states is None:encoder_hidden_states=hidden_states
        elif attn.norm_cross:encoder_hidden_states=attn.norm_encoder_hidden_states(encoder_hidden_states)
        key=attn.to_k(encoder_hidden_states);value=attn.to_v(encoder_hidden_states)
        query=attn.head_to_batch_dim(query);key=attn.head_to_batch_dim(key);value=attn.head_to_batch_dim(value)
        attention_probs=attn.get_attention_scores(query,key,attention_mask)
        attention_probs=self.controller(attention_probs,layer=self.layer,is_cross=is_cross,batch_size=batch_size)
        hidden_states=torch.bmm(attention_probs,value)
        hidden_states=attn.batch_to_head_dim(hidden_states)
        hidden_states=attn.to_out[0](hidden_states);hidden_states=attn.to_out[1](hidden_states)
        if ndim==4:hidden_states=hidden_states.transpose(-1,-2).reshape(b,c,h,w)
        if attn.residual_connection:hidden_states=hidden_states+residual
        return hidden_states/attn.rescale_output_factor

@dataclass
class PairPrompt:
    base: str
    edited: str

class PairDiffusion:
    def __init__(self,model_path,device='cuda',dtype=None):
        dtype=dtype or (torch.float16 if device.startswith('cuda') else torch.float32)
        self.model_path=Path(model_path);self.device=device;self.dtype=dtype
        self.pipe=StableDiffusionPipeline.from_pretrained(str(model_path),local_files_only=True,
                    torch_dtype=dtype,variant='fp16',use_safetensors=True,safety_checker=None,
                    feature_extractor=None,requires_safety_checker=False)
        self.pipe.scheduler=DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.tokenizer=CLIPTokenizerFast.from_pretrained(str(model_path),subfolder='tokenizer',local_files_only=True)
        for module in (self.pipe.unet,self.pipe.vae,self.pipe.text_encoder):
            module.requires_grad_(False);module.eval()
        self.pipe.to(device);self.pipe.enable_vae_slicing();self.pipe.set_progress_bar_config(disable=True)

    def initial_latents(self,seed,height,width):
        return torch.randn((1,self.pipe.unet.config.in_channels,height//8,width//8),
                           generator=torch.Generator('cpu').manual_seed(seed),dtype=torch.float32)

    @torch.inference_mode()
    def generate(self,pair,controller=None,*,seed=0,steps=12,height=256,width=256,latents=None):
        if height%8 or width%8:raise ValueError('Image dimensions must be multiples of 8')
        alignment=align(self.pipe.tokenizer,pair.base,pair.edited)
        if controller is not None:
            if controller.capture.step or controller.capture.trace:raise ValueError('Controller is single-use per sampling run')
            if controller.steps!=steps:raise ValueError('Controller steps must match sampler steps')
            processors={name:EditingAttnProcessor(controller,name) for name in self.pipe.unet.attn_processors}
        else:processors={name:AttnProcessor() for name in self.pipe.unet.attn_processors}
        self.pipe.unet.set_attn_processor(processors)
        initial=self.initial_latents(seed,height,width) if latents is None else latents.detach().cpu().clone()
        if initial.shape!=(1,4,height//8,width//8):raise ValueError('Initial latent shape mismatch')
        self.pipe.scheduler.set_timesteps(steps,device=self.device)
        if controller:controller.capture.timesteps=[int(t) for t in self.pipe.scheduler.timesteps]
        def end_step(pipe,index,t,callback_kwargs):
            if controller:controller.capture.advance()
            return callback_kwargs
        result=self.pipe([pair.base,pair.edited],height=height,width=width,num_inference_steps=steps,
                    guidance_scale=7.5,latents=initial.to(self.device,self.dtype).repeat(2,1,1,1),
                    generator=torch.Generator(self.device).manual_seed(seed),output_type='latent',
                    callback_on_step_end=end_step).images
        if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite diffusion output')
        images=self.pipe.image_processor.postprocess(
            self.pipe.vae.decode(result/self.pipe.vae.config.scaling_factor,return_dict=False)[0],output_type='pil')
        return {'base_image':images[0],'image':images[1],'initial_latents':initial,
                'latents':result.detach().cpu(),'alignment':alignment.as_dict(),
                'seed':seed,'steps':steps,'height':height,'width':width,'scheduler':'DDIM',
                'safety_checker':'not loaded for local research fixture','guidance_scale':7.5}
