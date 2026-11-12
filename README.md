# Training a Decision Model with Qwen

**Course revision and training pipeline by Vansh Gupta**

This repository explores training a language model to make structured decisions over a state and a set of candidate answers. It builds on the BEV/JEV decision-model walkthrough and implementation by Neural Breakdown with AVB, with the goal of understanding and running the training workflow as a course revision.

The model supports three task types: selecting an option, answering yes/no, and choosing a score from an ordered scale. A Qwen3-0.6B backbone encodes the prompt; a small attention-based head then scores the available choices. LoRA can optionally adapt selected backbone layers.

> This repository documents the implementation and training workflow. Add your own run details and measured evaluation results here after training; no benchmark result is claimed by this README.

## What this project covers

- Prompt and token construction for state, question, and candidate choices.
- Position IDs and attention masks that let each choice use the question context without attending to other choices.
- A choice-scoring head that combines task, answer, and option representations.
- Configurable training with a frozen backbone or LoRA, gradient accumulation, validation, and checkpoint initialization.
- Evaluation of overall accuracy, per-task accuracy, loss, and yes/no AUC when both labels are present.

## Architecture overview

```text
State + task instruction + question + choices
                    │
                    ▼
    Qwen3 backbone with structured attention
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
 Option representations   Answer representation
        └───────────┬───────────┘
                    ▼
     Attention-based choice-scoring head
                    │
                    ▼
       Scores → selected decision
```

The attention mask gives each option access to the question and its own tokens, while preventing it from reading the other options. The answer segment follows all choices and can attend to the full preceding prompt. See [`arch.ipynb`](arch.ipynb) and the diagrams in [`assets/`](assets/) for the walkthrough.

## Repository guide

| File | Purpose |
| --- | --- |
| [`tokenization.py`](tokenization.py) | Formats prompts, builds position IDs and attention masks, and identifies representation positions. |
| [`dataset.py`](dataset.py) | Loads and expands decision questions, tokenizes examples, and batches variable-length inputs. |
| [`network.py`](network.py) | Loads the Qwen backbone, configures optional LoRA, and scores choices with the task head. |
| [`train.py`](train.py) | Runs training, validation, gradient accumulation, and checkpoint initialization. |
| [`inference.py`](inference.py) | Loads a saved checkpoint and evaluates it. |
| [`logger.py`](logger.py) | Saves run metadata, metrics, and head/adapter checkpoints. |
| [`configs/`](configs/) | Example smoke-test, head-only, and LoRA training configurations. |

## Data and model

- **Backbone:** [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Dataset:** [`avbiswas/bev-decision-150K`](https://huggingface.co/datasets/avbiswas/bev-decision-150K)
- **Reference walkthrough:** [Training JEV networks from Qwen](https://youtu.be/sF3CNPbWA8o)
- **Original architecture livestream:** [Architecture walkthrough](https://youtube.com/live/AzxoU7kxjig)

The dataset is downloaded automatically on the first run. Please consult the linked model, dataset, and original repository for their respective licenses and attribution requirements.

## Setup

Install [uv](https://docs.astral.sh/uv/) and sync the locked dependencies:

```bash
uv sync
```

The first run downloads the model weights, tokenizer, and dataset. The code selects CUDA, then Apple MPS, then CPU. Full training configurations are intended for a GPU; reduce the batch size for smaller devices.

For a fresh Ubuntu GPU machine, [`cloud_startup_script.sh`](cloud_startup_script.sh) installs system tools, sets up uv, and checks CUDA availability:

```bash
bash cloud_startup_script.sh
```

## Train

Run a small overfitting sanity check with the 10-question configuration:

```bash
uv run python train.py configs/smoke.yaml --name smoke
```

For full training with the first 20 Qwen layers and LoRA on the last 12 kept layers (layers 8–19):

```bash
uv run python train.py configs/full_lora_k12.yaml --name full-lora-k12
```

This configuration trains the choice head and LoRA adapters. It uses 1,024 state tokens and accumulates two microbatches of 32 questions per optimizer step. `optim.grad_accum_steps` defaults to 1. Logging, validation, and save intervals count optimizer steps; an incomplete final accumulation group is skipped.

Other configurations include [`full_head_only.yaml`](configs/full_head_only.yaml) for a frozen backbone and [`full_lora.yaml`](configs/full_lora.yaml) for LoRA on the last four of the 20 kept layers.

Each run writes its config, metrics, summary, and checkpoints under `runs/<name>-<timestamp>/`. Checkpoints contain the choice head and, when enabled, LoRA weights; the Qwen base weights are loaded separately. `best` tracks test accuracy, `latest` is overwritten at each save interval, and `final` is saved at the end.

## Continue from saved weights

Pass a checkpoint directory or a run ID (which selects that run's final checkpoint):

```bash
uv run python train.py configs/full_lora_k12.yaml --name continued \
  --resume runs/<run-id>/checkpoints/final
```

Use the same backbone, head architecture, and LoRA configuration as the checkpoint. Resume initializes a new training run from the saved weights; the optimizer, learning-rate schedule, and step counter start fresh.

## Evaluate

```bash
uv run python inference.py runs/<run-id>/checkpoints/final \
  --max_questions 100 --batch_size 2 --num_workers 0
```

Evaluation reports loss, overall accuracy, accuracy for each task type, and yes/no AUC when both label classes are present. It uses the token limits recorded in the checkpoint.

## Author

**Vansh Gupta** — course revision and training workflow exploration.

This repository is based on [avbiswas/bev-train](https://github.com/avbiswas/bev-train). The original project and its associated materials remain credited to their respective authors.
