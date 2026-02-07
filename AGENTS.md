# Thinnka Podsmith - Agent Guide

This guide helps AI agents work effectively with the Thinnka Podsmith codebase.

## Project Overview

**Thinnka Podsmith** is a Python CLI tool that automates Runpod Pod provisioning and executes Open R1 GRPO or SFT fine-tuning training. The tool:
- Checks if Hugging Face models are gated
- Estimates model VRAM requirements and selects appropriate GPUs
- Creates Runpod Pods with SSH access
- Installs Open R1 training framework on the pod
- Runs GRPO (Group Relative Policy Optimization) or SFT (Supervised Fine-Tuning) training
- Uploads results to Hugging Face Hub
- Streams training progress via webhooks (optional)

**Single-file architecture**: The entire project logic is in `thinnka_runner.py` (~1500 lines).

## Key Commands

### Running the Project

```bash
# Basic GRPO training run
python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1

# SFT training mode
python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1 --sft

# Force specific GPU type
python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1 --gpu-type "L40S"

# Use custom Docker image
python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1 --image reeeon/thinnka:latest

# Enable Weights & Biases logging
python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1 --report-to wandb

# Dry run (validate GPU selection only)
python thinnka_runner.py --repo-id unsloth/gemma-2b --gpu-count 1 --dry-run
```

### Docker Commands

```bash
# Build the custom SFT-focused image
docker build -t reeeon/thinnka:latest .

# Push to Docker Hub
docker push reeeon/thinnka:latest

# Local testing with GPUs
docker run --gpus all -it --rm -e HF_TOKEN=your_token thinnka-sft bash
```

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp example.env .env
# Edit .env with your API keys and tokens
```

## Project Structure

```
thinkingConversion/
├── thinnka_runner.py    # Main CLI application (all logic)
├── requirements.txt     # Python dependencies
├── Dockerfile          # Custom SFT Docker image
├── entrypoint.sh       # SSH server startup script
├── example.env         # Environment variable template
├── README.md           # User documentation
└── .gitignore         # Git ignore patterns
```

## Code Organization

### Main Components in `thinnka_runner.py`

1. **ProgressReporter** (lines 110-201): Async webhook/Discord progress reporting
2. **RunpodGraphQLClient** (lines 203-302): Runpod API GraphQL wrapper
3. **parse_args()** (lines 304-396): CLI argument parser with extensive configuration
4. **cleanup_pod()** (lines 408-451): Pod termination on failure
5. **ensure_model_not_gated()** (lines 453-464): Hugging Face model access check
6. **compute_model_size_gb()** (lines 467-480): VRAM requirement calculation
7. **resolve_chat_template()** (lines 483-509): Chat template handling for models
8. **select_gpu_candidates()** (lines 512-535): GPU selection logic with priority (L40S > A100 SXM > H100 SXM)
9. **build_grpo_config()** (lines 594-684): GRPO training configuration
10. **build_sft_config()** (lines 723-769): SFT training configuration with QLoRA support
11. **build_accelerate_config()** (lines 772-798): DeepSpeed ZeRO config generation
12. **build_setup_script()** (lines 872-989): Remote Open R1 installation script
13. **build_train_script()** (lines 992-1275): Remote training execution script with auto-length detection
14. **main()** (lines 1278-1501): Orchestration of entire workflow

### Key Constants

```python
ALLOWED_GPU_COUNTS = {1, 2, 4, 6, 8}
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
CUSTOM_IMAGE = "reeeon/thinnka:latest"
GPU_PRIORITY = [("L40S", ...), ("A100 SXM", ...), ("H100 SXM", ...)]
```

## Dependencies

**Core dependencies** (from requirements.txt):
- `huggingface_hub` - Hugging Face Hub API client
- `httpx` - Async HTTP client (used for webhooks)
- `discord-webhook-async` - Discord notifications (optional)
- `paramiko` - SSH client for pod communication
- `PyYAML` - YAML config parsing
- `python-dotenv` - Environment variable loading

**Remote dependencies** (installed on pod):
- PyTorch 2.6.0+cu124
- vLLM 0.8.5 (GRPO only)
- flash-attn (optional, based on `--attn-implementation`)
- transformers (latest from git)
- peft, bitsandbytes (for QLoRA)
- Open R1 (cloned from GitHub)

## Environment Variables

**Required:**
- `RUNPOD_API_KEY` - Runpod API authentication
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` - Hugging Face write token (read-only fails)

**Recommended:**
- `SSH_PUBLIC_KEY` or `RUNPOD_SSH_PUBLIC_KEY` - Public key for pod SSH access
- `SSH_PRIVATE_KEY_PATH` - Private key path (defaults to `~/.ssh/id_ed25519`)

**Optional:**
- `PROGRESS_WEBHOOK_URL` - Async progress webhook endpoint
- `DISCORD_WEBHOOK_URL` - Discord notifications
- `WANDB_API_KEY` - Weights & Biases integration

**Note:** `.env` file is loaded automatically with highest priority, overriding system environment variables.

## Code Patterns and Conventions

### Error Handling

- Most operations wrap exceptions and raise `RuntimeError` with descriptive messages
- SSH command failures include recent output in error messages
- Cleanup is performed in `finally` block if an error occurs during pod provisioning/training

### SSH Pattern

```python
# Upload text to remote
upload_text(ssh_client, remote_path, content)

# Execute command and stream output
run_ssh_command(ssh_client, f"bash -lc \"command\"", reporter, stage)
```

### Config Generation Pattern

Training configs are generated as Python dicts, then dumped to YAML:
```python
config = {...}  # Build config dict
config_text = yaml.safe_dump(config, sort_keys=False)  # Convert to YAML
upload_text(ssh_client, remote_path, config_text)  # Upload to pod
```

### Progress Reporting Pattern

```python
reporter.send("event_name", "Human-readable message", extra={...})
# Standard stages: "setup", "train"
# Events include: "start", "setup_start", "setup_done", "train_start", "train_done", "error", etc.
```

## Training Modes

### GRPO (Default)

- Online reinforcement learning with vLLM generation
- Reward functions: accuracy, format, tag_count (if using default tags)
- Use `--no-vllm` to switch to transformers generation (slower, safer)
- Generation count auto-adjusted to divide effective batch size
- Tags: default `<think>` and `</think>` (no answer tags unless specified)

### SFT

- Supervised fine-tuning using QLoRA (4-bit quantization + LoRA adapters)
- QLoRA config: lora_r=16, lora_alpha=32, lora_dropout=0.05, all-linear targets
- Forces ZeRO-2 (QLoRA incompatible with ZeRO-3)
- Use `--shard-model` for large models to disable QLoRA and use ZeRO-3 sharding instead
- Chat templates auto-injected if model lacks one

## GPU Selection Logic

1. Calculate required VRAM from model size (sum of weight file sizes)
2. If `--shard-model`, divide VRAM by GPU count for per-GPU estimate
3. Query Runpod for available GPU types
4. Filter GPUs with sufficient VRAM matching optional `--gpu-type` substring
5. Select based on priority: L40S → A100 SXM → H100 SXM
6. Try creating pod with each candidate, first success wins

## Critical Gotchas

### ZeRO Stage Conflicts

- **QLoRA + SFT**: Forces ZeRO-2 (incompatible with ZeRO-3)
- **--shard-model**: Forces ZeRO-3 regardless of `--deepspeed-stage` setting
- Default for GRPO: ZeRO-3, Default for SFT: ZeRO-2

### Chat Templates

- Some models (e.g., `unsloth/gemma-2b`) lack chat templates
- Runner injects `DEFAULT_CHAT_TEMPLATE` automatically
- Override with `--chat-template path/to/template.jinja`

### Attention Implementation

- Default: `sdpa` (avoids FlashAttention ABI issues)
- Use `--attn-implementation flash_attention_2` for FlashAttention
- FlashAttention install happens during pod setup if selected

### Tag Behavior (GRPO)

- Default: `<think>` tags only, accuracy reward only
- Custom tags disable Open R1 format rewards (falls back to safer rewards)
- Answer tags must be set as a pair: both start and end

### Dataset Length Auto-detection

The training script runs on the pod to auto-detect optimal lengths:
- Loads the dataset, tokenizes with model tokenizer
- If dataset has `messages` column, splits last assistant message as completion
- Otherwise treats entire text as prompt
- Caps lengths at `tokenizer.model_max_length` if < 1,000,000
- Updates config YAML on-the-fly before training starts

### SSH Key Handling

- Private key defaults to `~/.ssh/id_ed25519`
- Supports both Ed25519 and RSA keys (via `load_private_key()`)
- Public key can be string or read from `~/.ssh/id_ed25519.pub`
- Public key is uploaded to pod via environment variables

### Token Persistence

- HF_TOKEN is written to `/workspace/thinnka/hf_token` on pod
- Also set in pod environment variables
- Token file is chmodded to 600 for security
- Used by both setup and training scripts

### Failure Cleanup

If an error occurs after pod creation:
- Pod is stopped via GraphQL mutation
- Waits for pod to exit RUNNING/RESTARTING state
- Attempts termination via REST API up to 5 times
- Does NOT create new pod automatically

## Dockerfile Notes

The included Dockerfile builds an SFT-focused image:
- Base: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (CUDA 12.8.1)
- Uses `uv` for fast package management
- Clones Open R1 to `/opt/open-r1`
- Installs peft, bitsandbytes
- Replaces transformers with latest git version
- Sets up SSH server on port 22
- Runs infinite sleep as default command

**Note:** This image does NOT include vLLM or flash-attn (SFT-only).

## Naming Conventions

- CLI flags: kebab-case (`--gpu-count`, `--max-steps`)
- Config keys: snake_case (`model_name_or_path`, `per_device_train_batch_size`)
- Functions: snake_case (`build_grpo_config`, `select_gpu_candidates`)
- Classes: PascalCase (`ProgressReporter`, `RunpodGraphQLClient`)
- Constants: UPPER_SNAKE_CASE (`ALLOWED_GPU_COUNTS`, `GPU_PRIORITY`)

## No Testing

The project currently has **no test suite**. No pytest, unittest, or any test files exist.

## No Linting/Formatting

The project currently has **no linting or formatting configuration**:
- No `.pylintrc`, `flake8`, `ruff`, `black`, `isort`, or similar config files
- No pre-commit hooks

## File Paths

The codebase uses `pathlib.Path` for path operations:
- Absolute paths with `Path.home()`, `Path.cwd()`, etc.
- Cross-platform compatible (uses forward slashes in code, works on Windows/Unix)
- SSH private key: `~/.ssh/id_ed25519` (expanduser() called)

## Important URLs

- Runpod GraphQL: `https://api.runpod.io/graphql`
- Runpod REST: `https://rest.runpod.io/v1`
- Open R1 repo: `https://github.com/huggingface/open-r1`
- Base image docs: Runpod PyTorch images

## Memory File Notes

If you discover:
- Build/test/lint commands → Add to memory
- Code style preferences → Document in memory
- Project-specific patterns or conventions → Add to memory
