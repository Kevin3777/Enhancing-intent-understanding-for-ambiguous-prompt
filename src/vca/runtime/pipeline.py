from dataclasses import asdict,dataclass
from pathlib import Path
import copy,json,re,shutil
import torch
from PIL import Image
from ..editing.pipeline_adapter import PairPrompt
from ..editing.token_alignment import align,target_tokens,TokenAlignment
from ..editing.controllers import AttentionController
from ..rl.ppo import PPO,PPOConfig
from ..rl.rollout import RolloutBatch
from ..rl.state_action import ACTION_TYPES,decode_action,encode_state
from ..rl.replay import DiagnosticReplay
from ..sesd.dialogue import Dialogue
from ..sesd.prompt_schema import Proposal
from ..scoring.metrics import ssim,human_satisfaction
from .session import Session
from .feedback import Feedback
from .logging import write_json

@dataclass
class RuntimeConfig:
    seed:int=20261027
    size:int=256
    step_choices:tuple=(8,12)
    initial_steps:int=12
    max_rounds:int=3
    stop_clip:float=.95
    reward_variant:str='paper_literal'
    attention:bool=True
    sesd:bool=True
    ppo:bool=True
    action_authority:str='user'
    disabled_operations:tuple=()
    mode:str='safe_attention'
    mapper_alpha_mode:str='legacy'  # legacy, fixed, learned; explicit implementation-assumption
    fixed_alpha:float=.7
    rag_enabled:bool=True
    strict_llm:bool=False
    policy_mode:str='stochastic'  # fixed is a deterministic evaluation-only ablation
    save_attention_samples:bool=True

class VCARuntime:
    def __init__(self,folder,prompt,generator,scorer,config=None,trainer=None,llm=None):
        self.config=config or RuntimeConfig();self.generator=generator;self.scorer=scorer
        self.session=Session(folder,prompt);self.dialogue=Dialogue(self.session.data['original_prompt'],llm=llm,max_rounds=100,
                                               retrieval_embedder=getattr(scorer,'retrieval_embeddings',None),
                                               rag_enabled=self.config.rag_enabled,strict_llm=self.config.strict_llm)
        checkpoint=self.session.data.get('checkpoint')
        if self.config.mapper_alpha_mode not in ('legacy','fixed','learned'):raise ValueError('Invalid mapper_alpha_mode')
        if self.config.policy_mode not in ('stochastic','fixed'):raise ValueError('Invalid policy mode')
        if self.config.policy_mode=='fixed' and (self.config.ppo or self.config.mapper_alpha_mode=='learned'):
            raise ValueError('Fixed behavior is evaluation-only with fixed mapper and alpha')
        if not 0<=self.config.fixed_alpha<=1:raise ValueError('fixed_alpha must be in [0,1]')
        if trainer is not None:self.trainer=trainer
        elif checkpoint:self.trainer=PPO.load(checkpoint)
        else:
            adapter=None
            if self.config.mapper_alpha_mode=='learned':
                from ..rl.mapper_alpha import MapperAlphaPolicy
                adapter=MapperAlphaPolicy()
            self.trainer=PPO(config=PPOConfig(minibatch=1,ignore_strength_head=self.config.mapper_alpha_mode!='legacy'),adapter=adapter)
        if (self.trainer.adapter is not None)!=(self.config.mapper_alpha_mode=='learned'):
            raise ValueError('Trainer mapper/alpha mode does not match session configuration')
        if self.trainer.config.ignore_strength_head!=(self.config.mapper_alpha_mode!='legacy'):
            raise ValueError('Trainer strength head must match mapper/alpha mode')
        self.replay=DiagnosticReplay()
        self.replay.rows=copy.deepcopy(self.session.data.get('replay_rows',[]))
        if 'dialogue' in self.session.data:self._restore_dialogue(self.session.data['dialogue'])
        write_json(self.session.folder/'runtime_config.json',asdict(self.config))

    def _dialogue_snapshot(self):
        return {'history':copy.deepcopy(self.dialogue.history),'resolved':dict(self.dialogue.resolved),
            'current':self.dialogue.current,'knowledge':copy.deepcopy(self.dialogue.kb.entries),'kb_version':self.dialogue.kb.version}

    def _restore_dialogue(self,d):
        self.dialogue.history=copy.deepcopy(d['history']);self.dialogue.resolved=dict(d['resolved']);self.dialogue.current=d['current']
        self.dialogue.kb.entries=copy.deepcopy(d['knowledge']);self.dialogue.kb.version=d['kb_version'];self.dialogue.rag.rebuild()

    def initialize(self):
        if 'initial_image' in self.session.data:return
        cfg=self.config;prompt=self.dialogue.current
        result=self.generator.generate(PairPrompt(prompt,prompt),seed=cfg.seed,steps=cfg.initial_steps,height=cfg.size,width=cfg.size)
        image=self.session.folder/'initial.png';result['image'].save(image)
        torch.save(result['initial_latents'],self.session.folder/'initial_latents.pt')
        ckpt=self.session.folder/'checkpoint_initial.pt';self.trainer.save(ckpt)
        self.session.data.update(initial_image=str(image),current_image=str(image),checkpoint=str(ckpt),dialogue=self._dialogue_snapshot(),replay_rows=[])
        self.session.save()

    def _state(self,image,prompt,feedback,round_index,last_reward):
        img=self.scorer.image_features(image);texts=self.scorer.text_features([prompt,feedback or prompt])
        history=' '.join(x['user_input'] for x in self.dialogue.history)
        return encode_state(img,texts[0],texts[1],history,round_index,last_reward)

    def step(self,feedback:Feedback):
        cfg=self.config;self.initialize();data=self.session.data
        if data['stop_reason']:raise ValueError('Session stopped; explicitly continue or undo before generating')
        if len(data['rounds'])>=cfg.max_rounds:raise ValueError('Configured round budget exhausted')
        before={k:copy.deepcopy(data[k]) for k in ('current_prompt','current_image','checkpoint','dialogue')}
        before['replay_rows']=copy.deepcopy(data.get('replay_rows',[]))
        before_prompt=data['current_prompt'];previous_image=Image.open(data['current_image']).convert('RGB')
        if cfg.sesd:proposal=self.dialogue.refine(feedback.text)
        else:
            proposal=Proposal(before_prompt+' '+feedback.text,'Add',target_phrase=feedback.text,backend='ablation_no_sesd_rag')
            self.dialogue.current=proposal.new_prompt
        if proposal.question:
            data['dialogue']=self._dialogue_snapshot();self.session.save()
            return {'status':'clarification_required','question':proposal.question,'proposal':proposal.as_dict()}
        state=self._state(previous_image,before_prompt,feedback.text,len(data['rounds']),data['rounds'][-1]['reward']['reward'] if data['rounds'] else 0)
        mask=torch.tensor([[kind not in cfg.disabled_operations for kind in ACTION_TYPES]])
        if cfg.action_authority=='user':
            mask[:]=False
            # A removed semantic operator becomes no-attention, preserving the actual
            # rewritten prompt; never silently reinterpret a user's instruction.
            mask[0,ACTION_TYPES.index(proposal.type)]=True
        elif cfg.action_authority!='policy':raise ValueError('Action authority must be user or policy')
        if not mask.any():raise ValueError('All policy action types disabled')
        ignored=(1,) if self.trainer.config.ignore_strength_head else ()
        with torch.no_grad():
            if cfg.policy_mode=='fixed':
                action=torch.tensor([[ACTION_TYPES.index(proposal.type),1,1,0,0]])
                value=self.trainer.policy.distributions(state[None],mask)[1]
                old_logprob=torch.zeros(1)  # deterministic behavior probability 1; never used in PPO
            else:action,old_logprob,value=self.trainer.policy.sample(state[None],mask,ignored)
        old_actor_logprob=old_logprob.item()
        params=decode_action(action[0],tuple(cfg.step_choices))
        # In the no-PPO ablation the same initialized stochastic actor remains
        # frozen. Only optimizer updates are removed; the action path is matched.
        operation=params['operation'] if cfg.attention and params['type'] not in cfg.disabled_operations else 'none'
        alignment=align(self.generator.pipe.tokenizer,before_prompt,proposal.new_prompt)
        adaptive=None;controlled_alignment=alignment
        if cfg.mapper_alpha_mode=='fixed':params['strength']=cfg.fixed_alpha
        elif cfg.mapper_alpha_mode=='learned':
            if operation!='none':
                sample=self.trainer.adapter.sample(alignment);mapper=sample.pop('mapper')
                adaptive=sample;old_logprob=old_logprob+sample['old_logprob'];params['strength']=sample['alpha']
                controlled_alignment=TokenAlignment(alignment.source,alignment.target,mapper,alignment.mapped_target)
            else:adaptive={'active':False,'old_logprob':0.,'method':'inactive; no attention operation'}
        phrase=proposal.target_phrase
        if not phrase or phrase.lower() not in proposal.new_prompt.lower():
            phrase=next((w for w in re.findall(r'\w+',proposal.new_prompt) if len(w)>3),'')
        indices=target_tokens(alignment.target,phrase,proposal.target_occurrence) if operation=='reweight' else []
        controller=AttentionController(controlled_alignment,operation,steps=params['steps'],strength=params['strength'],
            cross_fraction=params['cross_fraction'],self_strength=params['self_strength'],target_indices=indices,
            reweight_c=params['reweight_c'],mode=cfg.mode,
            replace_source='paper_literal_target' if cfg.mode=='paper_literal' else 'base_to_edited_assumption')
        round_id=data['next_round_id'];folder=self.session.folder/f'round_{round_id:03d}';folder.mkdir(exist_ok=False)
        result=self.generator.generate(PairPrompt(before_prompt,proposal.new_prompt),controller,seed=cfg.seed,
                steps=params['steps'],height=cfg.size,width=cfg.size)
        image_path=folder/'image.png';result['image'].save(image_path);result['base_image'].save(folder/'source_regenerated.png')
        previous_image.save(folder/'previous_actual.png')
        torch.save({'initial':result['initial_latents'],'final':result['latents']},folder/'latents.pt')
        reward=self.scorer.score(result['image'],before_prompt,proposal.new_prompt,cfg.reward_variant)
        reward.update(human_feedback=feedback.rating,human_feedback_source=feedback.source,
                      feedback_in_reward=False)
        done_reason=('human_satisfied' if feedback.satisfied else 'clip_threshold' if reward['raw_cosine_new']>=cfg.stop_clip
                     else 'finite_round_budget' if len(data['rounds'])+1>=cfg.max_rounds else None)
        next_state=self._state(result['image'],proposal.new_prompt,feedback.text,len(data['rounds'])+1,reward['reward'])
        with torch.no_grad():next_value=self.trainer.policy.distributions(next_state[None])[1]
        transition={'seed':cfg.seed,'latents_path':str(folder/'latents.pt'),'prompt_previous':before_prompt,
            'prompt_new':proposal.new_prompt,'user_feedback':asdict(feedback),'action':action[0].tolist(),
            'action_params':params,'old_logprob':old_logprob.item(),'value':value.item(),'next_value':next_value.item(),
            'reward':reward,'done':done_reason is not None,'termination_reason':done_reason,'policy_version':self.trainer.version,
            'type_mask':mask[0].tolist(),'state':state.tolist(),'next_state':next_state.tolist(),
            'old_actor_logprob':old_actor_logprob,'adaptive':adaptive,
            'policy_behavior':'deterministic_fixed' if cfg.policy_mode=='fixed' else 'sampled_policy',
            'semantic_type_source':'SESD/user instruction','action_type_authority':cfg.action_authority,
            'source_image_method':'same-seed regenerated base branch, not inversion of previous image','age_clock':round_id}
        td=reward['reward']+self.trainer.config.gamma*(not transition['done'])*next_value.item()-value.item()
        self.replay.append(transition,td,round_id);pool_index=len(self.replay.rows)-1
        training=None
        if cfg.ppo:
            batch=RolloutBatch(state[None],action,old_logprob,value,torch.tensor([reward['reward']]),next_value,
                torch.tensor([transition['done']]),torch.tensor([self.trainer.version]),mask,[transition])
            training=self.trainer.update(batch,replay_pool=self.replay,replay_indices=[pool_index],age_clock=round_id)
        checkpoint=folder/'checkpoint.pt';self.trainer.save(checkpoint)
        if not cfg.save_attention_samples:controller.capture.samples.clear()
        trace=controller.capture.save(folder)
        clip_feedback={k:reward[k] for k in ('raw_cosine_previous','raw_cosine_new','reward')}
        if self.dialogue.history:
            self.dialogue.history[-1].update(image_path=str(image_path),clip_feedback=clip_feedback,
                policy_version_after=self.trainer.version)
        self.dialogue.kb.append_session(proposal.new_prompt,feedback.text,data['id'],round_id,str(image_path),clip_feedback)
        row={'round_id':round_id,'image':str(image_path),'previous_prompt':before_prompt,'new_prompt':proposal.new_prompt,
            'feedback':asdict(feedback),'proposal':proposal.as_dict(),'action':params,'executed_operation':operation,
            'reward':reward,'training':training,'trace':trace,'done':transition['done'],
            'structure_ssim_to_previous_generated':ssim(previous_image,result['image']),
            'structure_reference':'previous_generated, not ground truth','before_snapshot':before}
        write_json(folder/'transition.json',transition);write_json(folder/'result.json',row)
        write_json(folder/'alignment.json',alignment.as_dict())
        if adaptive is not None:write_json(folder/'mapper_alpha_action.json',adaptive)
        write_json(folder/'replay_diagnostic.json',self.replay.sample(min(8,len(self.replay.rows)),round_id,beta=.4+.6*min(round_id/cfg.max_rounds,1)))
        data['checkpoint']=str(checkpoint);data['dialogue']=self._dialogue_snapshot();data['stop_reason']=done_reason
        data['replay_rows']=copy.deepcopy(self.replay.rows)
        self.session.append(row)
        return {'status':'generated',**row}

    def undo(self):
        self.session.undo();self.trainer=PPO.load(self.session.data['checkpoint'])
        self._restore_dialogue(self.session.data['dialogue']);self.replay=DiagnosticReplay()
        self.replay.rows=copy.deepcopy(self.session.data.get('replay_rows',[]))

    def continue_session(self):
        if len(self.session.data['rounds'])>=self.config.max_rounds:raise ValueError('Increase max_rounds explicitly before continuing')
        self.session.data['stop_reason']=None;self.session.save()

    def finalize(self):
        self.initialize();data=self.session.data;target=data['current_prompt']
        candidates=[{'image':data['initial_image'],'round_id':0}]+[{'image':r['image'],'round_id':r['round_id']} for r in data['rounds']]
        ranked=[]
        for row in candidates:
            score=self.scorer.score(Image.open(row['image']).convert('RGB'),data['original_prompt'],target,self.config.reward_variant)
            ranked.append({**row,'raw_cosine_final_target':score['raw_cosine_new']})
        ranked.sort(key=lambda x:-x['raw_cosine_final_target'])
        final=self.session.folder/'final.png';shutil.copyfile(ranked[0]['image'],final)
        selection={'final_image':str(final),'target_prompt':target,'ranked_candidates':ranked,
            'selection_rule':'rescore active history against common final target; ENGINEERING_CHOICE',
            'human_satisfaction':human_satisfaction([r['feedback'] for r in data['rounds']]),
            'rounds':len(data['rounds']),'original_paper_metrics_reproduced':False}
        write_json(self.session.folder/'final_selection.json',selection);return selection
