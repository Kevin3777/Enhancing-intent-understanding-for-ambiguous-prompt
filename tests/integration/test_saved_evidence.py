"""Verify actual GPU run artifacts when present; inference entrypoints produce them."""
from pathlib import Path
import json,sys,pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from verify_evidence import check_session

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def test_actual_attention_evidence():
    path=ROOT/'outputs/m1_attention_fixed/summary.json'
    if not path.exists():pytest.skip('Run scripts/test_attention_effect.py with matching output folder')
    result=read(path);assert result['status']=='passed' and result['identical_prompt_max_error']<.005
    assert {r['operator'] for r in result['operators']}=={'replace','refine','reweight'}
    assert all(r['source_latents_unchanged'] and r['no_op_max_latent_error']==0 and r['trace']['injections']>0 for r in result['operators'])

def test_actual_connected_session_and_checkpoint():
    folder=ROOT/'outputs/m4_connected'
    if not (folder/'final_selection.json').exists():pytest.skip('Actual connected session not yet run')
    assert check_session(folder)['checkpoint_behavior_action_reproduced']

def test_actual_dataset_training_and_type_head():
    folder=ROOT/'outputs/m3_diffusion_dataset/seed_42_episode_0'
    if not (folder/'final_selection.json').exists():pytest.skip('Actual verified-dataset training not yet run')
    assert all(x>0 for x in check_session(folder,expect_types=True)['type_head_deltas'])

def test_actual_ablation_statistics():
    folder=ROOT/'outputs/m5_ablation'
    if not (folder/'summary.json').exists():pytest.skip('Full local ablation not yet completed')
    rows=read(folder/'measurements.json');assert len(rows)==24
    assert all(r['status']=='measured' for r in rows)
    assert all(r['lpips_initial_final']['status']=='measured' for r in rows)
    summary=read(folder/'summary.json')['summary'];assert len(summary)==8
    assert all(v['raw_cosine_target']['n']==3 and v['raw_cosine_target']['ci95'] for v in summary.values())

def test_actual_nonuniform_real_diffusion_replay():
    import torch
    folder=ROOT/'outputs/m3_prioritized_batch/seed_42_batches'
    if not (folder/'summary.json').exists():pytest.skip('Multi-trajectory real-image PPO not yet run')
    summary=read(folder/'summary.json');u=summary['updates'][0]
    assert u['transitions']==6 and u['episodes']==2 and u['behavior_version']==0 and u['new_version']==1
    p=torch.tensor(u['probabilities_first_epoch']);w=torch.tensor(u['importance_first_epoch'])
    assert p.std()>1e-4 and torch.allclose(p.sum(),torch.tensor(1.))
    assert w.std()>1e-4 and max(u['ages'])>min(u['ages'])
    assert u['actor_delta']>0 and u['critic_delta']>0 and u['type_head_delta']>0
    rows=read(folder/'update_000/rollout.json');assert all(r['policy_version']==0 for r in rows)
    assert sum(r['done'] for r in rows)==2
