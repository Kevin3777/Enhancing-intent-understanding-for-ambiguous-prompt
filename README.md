# Visual Co-Adaptation (VCA): paper-based implementation

This repository contains an independent engineering reimplementation of the core method in **“Enhancing intent understanding for ambiguous prompt: A human–machine co-adaption strategy”**, *Neurocomputing*, Volume 646 (2025), Article 130415. The paper PDF is the sole method source for this implementation; the earlier README code examples have been replaced with descriptions tied to the current source.

> This project reimplements the paper’s core workflow. It does not reproduce the paper’s original private model weights, full training scale, human study, or reported metrics. Where the paper does not determine an executable detail, the choice is documented as an implementation assumption.

![VCA framework](assets/fig2_framework_overview.png)

**Citation**

> “Enhancing intent understanding for ambiguous prompt: A human–machine co-adaption strategy.” *Neurocomputing* 646 (2025), 130415. https://doi.org/10.1016/j.neucom.2025.130415

## What is implemented

- **SESD and CA-RAG**: multi-turn prompt history, ambiguity handling, CLIP-embedding knowledge retrieval, retrieved evidence passed into a configurable LLM, and prompt refinement. Rule fixtures are available for offline tests; they are not a substitute for a real LLM run.
- **PPCO and cross-attention editing**: Diffusers U-Net attention capture and three edit operations (Replace, Refine, Reweight) connected to image generation.
- **CLIP feedback**: historical-prompt and new-request cosine rewards, logged separately and combined according to the paper’s dual-objective expression.
- **PPO and replay**: Actor–Critic, old-policy log probabilities and policy versions, one-step TD targets, prioritized replay with importance weights, and clipped PPO updates.
- **Mapper/Alpha and closed loop**: learnable edit mapping and mixing parameters can be updated with PPO score-function gradients; generated image feedback and policy updates feed the next SESD round.

Core implementation is under `src/vca/`; runnable entry points are in `scripts/`; tests are in `tests/`. The paper-to-code table and formula audit are in `docs/`.

## Environment setup

Python 3.12 is required. The pinned dependency set targets PyTorch 2.6 with CUDA 12.4 on Windows/Linux. The published pinned dependency file targets CUDA 12.4; select an appropriate PyTorch build separately if you need a CPU-only environment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e . --no-deps
```

`requirements-lock.txt` records the tested full package set. `requirements-metrics.txt` adds optional LPIPS evaluation dependencies. The local LLM extension is optional and listed in `configs/requirements-local-llm.txt`.

## Model weights

Weights are not committed. The model configuration pins public model IDs and revisions in `configs/model.yaml` and `scripts/fetch_assets.py`:

- Stable Diffusion 1.5: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- CLIP ViT-B/32: `openai/clip-vit-base-patch32`
- A local LLM is optional; the provided GGUF fetcher uses `Qwen/Qwen2.5-1.5B-Instruct-GGUF` at a pinned revision as an engineering substitute for the paper’s unspecified/fine-tuned LLM.

Review and accept each model’s license and access conditions on its official model page before downloading. Then run:

```powershell
.\.venv\Scripts\python scripts\fetch_assets.py
.\.venv\Scripts\python -m pip install -r configs/requirements-local-llm.txt
.\.venv\Scripts\python scripts\fetch_local_llm_gguf.py --out models\qwen25_15b_q4
```

If a provider-compatible LLM endpoint is preferred, set `VCA_LLM_ENDPOINT`, `VCA_LLM_API_KEY`, and `VCA_LLM_MODEL` in the process environment. Credentials are read at runtime and must never be committed. Model download scripts write under `models/` and verify pinned revisions/hashes where available.

## Multi-turn inference

From the repository root, after preparing the SD1.5, CLIP, and local LLM weights:

```powershell
.\.venv\Scripts\python scripts\interactive_demo.py `
  --prompt "a mouse sitting on a wooden desk" `
  --local-llm models\qwen25_15b_q4 `
  --mapper-alpha learned --action-authority policy `
  --size 256 --steps 8 --max-rounds 3 `
  --out outputs\interactive-run
```

Enter actual user feedback at `Feedback>`. `/undo`, `/rate 1..5`, `/satisfied`, and `/quit` are supported. Use `--fixture` only for a scripted integration smoke path; it does not represent a real user study or a real LLM session. Use `--no-ppo`, `--no-rag`, or `--mapper-alpha fixed` for corresponding ablations.

## Training

A small CPU-friendly PPO wiring check, without diffusion weights:

```powershell
.\.venv\Scripts\python scripts\train_ppo.py --env toy --seeds 0 1 2 --iterations 60 --out outputs\toy-run
```

A real diffusion/PPO run requires the models above and a locally prepared, license-compliant catalog. The catalog is intentionally not included because dataset redistribution rights have not been established. After preparing `data/catalog.jsonl`, run a low-budget example:

```powershell
.\.venv\Scripts\python scripts\train_ppo.py `
  --env diffusion --catalog data\catalog.jsonl `
  --episodes 2 --rollout-episodes 2 --seeds 42 --minibatch 6 `
  --size 256 --steps 8 --local-llm models\qwen25_15b_q4 `
  --mapper-alpha learned --out outputs\ppo-run
```

Choose a new output directory for each run. The training script rejects an existing output folder. The catalog must contain eligible training records; validation data must remain separate. The default configuration and preprocessing choices are implementation assumptions, not the paper’s unpublished original split.

## Tests

The fast test suite does not need downloaded diffusion or LLM weights:

```powershell
.\.venv\Scripts\python -m pytest -q --basetemp .pytest_tmp
```

Saved-evidence integration checks skip when their local generated artifacts are absent. A real model smoke run requires the weights above and a CUDA-capable GPU. A 256-pixel, 8-step configuration is intended as a low-budget starting point; this is not a guaranteed memory/performance profile for every 10 GB GPU.

## Paper fidelity and known limits

`docs/paper_to_code_matrix.md` maps the PDF formulas and algorithms to source and tests. `docs/equation_audit.md` records the formula-level review and `docs/equation_mapping_addendum.md` describes the implemented connected path. `docs/execution_principles.md`, `docs/reproduction_scope.md`, and `docs/open_questions.md` separate paper-specified behavior from implementation assumptions and items that cannot be uniquely recovered.

Key limits include: a public local Qwen model substitutes for the paper’s LLM; model checkpoints, parts of the original training procedure, and human evaluation are unavailable; several attention/mapper dimensions and gradient details are underspecified. CLIP reward does not backpropagate through the discrete image-generation path into Mapper/Alpha; those parameters use PPO score-function gradients, an explicit engineering assumption. The project does not claim the paper’s reported scores or that the full method outperforms every ablation.

## License and reuse

The paper figures already present in `assets/` are retained from the existing repository. Check the paper and model licenses before redistributing those assets or downloaded weights. The model weights and training dataset are not included.
