# Thinnka Podsmith

Provision Runpod Pods for GRPO fine-tuning with the Open R1 repo. This project:
- checks if a Hugging Face model is gated
- estimates model size to pick a GPU with enough VRAM
- prefers L40S, then A100 SXM, then H100 SXM
- creates a Runpod Pod with a target Docker image
- SSHes into the Pod to run Open R1 GRPO training
- pushes the result to the Hugging Face Hub using your `HF_TOKEN`
- streams training progress to an async webhook

## Prereqs
- Windows (or WSL) with Python 3.11
- Runpod API key
- Hugging Face token with write access
- SSH key pair added to Runpod account settings
- Pod image that allows SSH on port 22

## Setup (Windows)
1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`

## Environment variables
- `RUNPOD_API_KEY` (required)
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` (required)
- `PROGRESS_WEBHOOK_URL` (optional, async progress events)
- `SSH_PUBLIC_KEY` or `RUNPOD_SSH_PUBLIC_KEY` (recommended)
- `SSH_PRIVATE_KEY_PATH` (optional, defaults to `~/.ssh/id_ed25519`)
- `WANDB_API_KEY` (optional if you pass `--report-to wandb`)

## SSH note
The script waits for a public IP and a mapped port for `22/tcp`.
If you use a custom image, ensure `sshd` is running and port 22 is exposed.

## Progress webhook
Set `PROGRESS_WEBHOOK_URL` to receive JSON updates. Payload includes:
- `event`
- `message`
- `timestamp`
- `stage` (setup or train when streaming logs)
- `project`
- `repo_id`

## Quick start
Runs the test case model and creates a Pod with the default image (no answer tags):
- `python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1`

Use the custom image if you build and push it (see below):
- `python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1 --image reeeon/thinnka:latest`
- Add `--skip-setup` when using the custom image to avoid reinstalling Open R1.

Enable Weights and Biases logging:
- `python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1 --report-to wandb`

## Custom image (Open R1 preinstalled)
This repo includes a Dockerfile that builds `reeeon/thinnka:latest`.

Build and push:
- `docker build -t reeeon/thinnka:latest .`
- `docker push reeeon/thinnka:latest`

Notes:
- Building can take time because Open R1 installs vLLM and flash-attn.
- Open R1 recommends CUDA 12.4 and PyTorch 2.6.0; the base image here is CUDA 12.8.1.
- If you build on macOS, use `--platform linux/amd64`.

## How it works
1. Fetches `model_info` from the Hub and stops if the model is gated.
2. Sums model file sizes to estimate required VRAM.
3. Queries Runpod GPU types and picks from: L40S, A100 SXM, H100 SXM.
4. Creates a Pod with your selected GPU count (1, 2, 4, 6, 8).
5. Waits for SSH on port 22, then installs Open R1 if needed.
6. Generates a GRPO config and launches training.
7. Uses `HF_TOKEN` to push to the Hub.
8. Streams progress lines to `PROGRESS_WEBHOOK_URL`.

## Reasoning tag behavior
By default, the config uses `<think> ... </think>` and **no answer tags** (accuracy reward only).
If you enable answer tags, the script uses Open R1's default format rewards.
If you change the reasoning or answer tags, the script falls back to safer rewards
because Open R1 format rewards are hardcoded to the default tags.
Use `--reasoning-tag`, `--reasoning-end-tag`, `--answer-tag`, and `--answer-end-tag` to customize.

Enable answer tags:
- `--answer-tag "<answer>" --answer-end-tag "</answer>"`

## References
- Runpod docs: https://docs.runpod.io/overview
- Runpod GraphQL Pods: https://docs.runpod.io/sdks/graphql/manage-pods
- Hugging Face Hub Python: https://huggingface.co/docs/huggingface_hub/index
- Open R1 repo: https://github.com/huggingface/open-r1
