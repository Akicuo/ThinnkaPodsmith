#!/usr/bin/env python3
"""
Thinnka Podsmith: Runpod Pod provisioning + Open R1 GRPO runner.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import paramiko
import yaml
from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

PROJECT_NAME = "Thinnka Podsmith"
RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"

DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
CUSTOM_IMAGE = "reeeon/thinnka:latest"

ALLOWED_GPU_COUNTS = {1, 2, 4, 6, 8}
WEIGHT_EXTENSIONS = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt"}

GPU_PRIORITY = [
    ("L40S", lambda name: "l40s" in name),
    ("A100 SXM", lambda name: "a100" in name and "sxm" in name),
    ("H100 SXM", lambda name: "h100" in name and "sxm" in name),
]


class ProgressReporter:
    def __init__(self, url: Optional[str], base_payload: Optional[Dict[str, Any]] = None) -> None:
        self.url = url
        self.base_payload = base_payload or {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue] = None
        self._thread: Optional[threading.Thread] = None
        if self.url:
            self._start()

    def _start(self) -> None:
        def runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            queue: asyncio.Queue = asyncio.Queue()
            self._loop = loop
            self._queue = queue
            loop.create_task(self._worker())
            loop.run_forever()

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        while self._loop is None or self._queue is None:
            time.sleep(0.01)

    async def _worker(self) -> None:
        if not self.url or not self._queue:
            return
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                payload = await self._queue.get()
                if payload is None:
                    break
                try:
                    await client.post(self.url, json=payload)
                except Exception:
                    pass
        if self._loop:
            self._loop.stop()

    def send(self, event: str, message: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        if not self.url or not self._loop or not self._queue:
            return
        payload = dict(self.base_payload)
        payload["event"] = event
        payload["timestamp"] = time.time()
        if message is not None:
            payload["message"] = message
        if extra:
            payload.update(extra)
        self._loop.call_soon_threadsafe(self._queue.put_nowait, payload)

    def close(self) -> None:
        if not self.url or not self._loop or not self._queue or not self._thread:
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        self._thread.join(timeout=5)


class RunpodGraphQLClient:
    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.client = httpx.Client(timeout=timeout)

    def query(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{RUNPOD_GRAPHQL_URL}?api_key={self.api_key}"
        response = self.client.post(url, json={"query": query, "variables": variables or {}})
        response.raise_for_status()
        payload = response.json()
        if "errors" in payload:
            message = payload["errors"][0].get("message", "Unknown GraphQL error")
            raise RuntimeError(message)
        return payload["data"]

    def get_gpu_types(self) -> List[Dict[str, Any]]:
        query = "query GpuTypes { gpuTypes { id displayName memoryInGb } }"
        data = self.query(query)
        return data.get("gpuTypes", [])

    def create_pod(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        query = """
        mutation PodFindAndDeploy($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) {
            id
            name
            imageName
            machineId
            machine { podHostId }
          }
        }
        """
        data = self.query(query, {"input": input_data})
        return data["podFindAndDeployOnDemand"]

    def get_pod_ports(self, pod_id: str) -> List[Dict[str, Any]]:
        query = """
        query Pod($podId: String!) {
          pod(input: { podId: $podId }) {
            id
            runtime {
              ports { ip isIpPublic privatePort publicPort type }
            }
          }
        }
        """
        data = self.query(query, {"podId": pod_id})
        pod = data.get("pod") or {}
        runtime = pod.get("runtime") or {}
        return runtime.get("ports") or []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provision a Runpod Pod and run Open R1 GRPO training.")
    parser.add_argument("--repo-id", default="unsloth/gemma-2b", help="Hugging Face model repo id.")
    parser.add_argument("--gpu-count", type=int, default=1, help="GPU count (1, 2, 4, 6, 8).")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image for the Pod.")
    parser.add_argument("--use-thinnka-image", action="store_true", help="Use reeeon/thinnka:latest.")
    parser.add_argument("--pod-name", default=None, help="Custom Pod name.")
    parser.add_argument("--cloud-type", default="ALL", choices=["ALL", "SECURE", "COMMUNITY"])
    parser.add_argument("--volume-gb", type=int, default=None, help="Persistent volume size in GB.")
    parser.add_argument("--container-disk-gb", type=int, default=None, help="Container disk size in GB.")
    parser.add_argument("--min-vcpu", type=int, default=4)
    parser.add_argument("--min-memory", type=int, default=16)
    parser.add_argument("--dataset-name", default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--dataset-prompt-column", default="problem")
    parser.add_argument("--reasoning-tag", default="<think>")
    parser.add_argument("--reasoning-end-tag", default="</think>")
    parser.add_argument("--answer-tag", default="")
    parser.add_argument("--answer-end-tag", default="")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--vllm-mode", default="colocate")
    parser.add_argument("--report-to", default="none", help="Set to wandb to enable W&B logging.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-setup", action="store_true", help="Skip Open R1 setup (useful with custom image).")
    parser.add_argument("--dry-run", action="store_true", help="Validate and select GPU only.")
    parser.add_argument("--ssh-private-key", default=None, help="Path to SSH private key.")
    parser.add_argument("--ssh-public-key", default=None, help="SSH public key string.")
    parser.add_argument("--progress-url", default=None, help="Override PROGRESS_WEBHOOK_URL.")
    parser.add_argument("--ssh-timeout-min", type=int, default=20)
    return parser.parse_args()


def ensure_model_not_gated(api: HfApi, repo_id: str, token: str):
    try:
        info = api.model_info(repo_id, token=token, files_metadata=True)
    except GatedRepoError as exc:
        raise RuntimeError(f"Model is gated: {repo_id}") from exc
    except RepositoryNotFoundError as exc:
        raise RuntimeError(f"Model not found: {repo_id}") from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(f"Hugging Face error: {exc}") from exc
    if getattr(info, "gated", False):
        raise RuntimeError(f"Model is gated: {repo_id}")
    return info


def compute_model_size_gb(model_info) -> float:
    total_bytes = 0
    weight_bytes = 0
    for sibling in model_info.siblings or []:
        size = getattr(sibling, "size", None)
        if size is None:
            continue
        total_bytes += size
        filename = getattr(sibling, "rfilename", "") or ""
        if any(filename.endswith(ext) for ext in WEIGHT_EXTENSIONS):
            weight_bytes += size
    if weight_bytes == 0:
        weight_bytes = total_bytes
    return weight_bytes / (1024 ** 3)


def select_gpu_candidates(
    gpu_types: List[Dict[str, Any]],
    required_vram_gb: float,
    explicit_choice: Optional[str] = None,
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for label, matcher in GPU_PRIORITY:
        if explicit_choice and explicit_choice != label:
            continue
        matches = []
        for gpu in gpu_types:
            name = f"{gpu.get('displayName', '')} {gpu.get('id', '')}".lower()
            if matcher(name):
                memory = gpu.get("memoryInGb") or 0
                if memory >= required_vram_gb:
                    matches.append(gpu)
        matches.sort(key=lambda g: g.get("memoryInGb", 0), reverse=True)
        for match in matches:
            yield label, match


def build_pod_env(ssh_public_key: str, hf_token: str) -> List[Dict[str, str]]:
    env = [
        {"key": "HF_TOKEN", "value": hf_token},
        {"key": "HUGGINGFACE_HUB_TOKEN", "value": hf_token},
        {"key": "HF_HOME", "value": "/workspace/.cache/huggingface"},
    ]
    if ssh_public_key:
        env.append({"key": "SSH_PUBLIC_KEY", "value": ssh_public_key})
        env.append({"key": "PUBLIC_KEY", "value": ssh_public_key})
    return env


def ensure_ssh_key_material(args: argparse.Namespace) -> Tuple[str, Path]:
    if args.ssh_public_key:
        public_key = args.ssh_public_key.strip()
    else:
        public_key = os.getenv("SSH_PUBLIC_KEY") or os.getenv("RUNPOD_SSH_PUBLIC_KEY") or ""
        if not public_key:
            default_pub = Path.home() / ".ssh" / "id_ed25519.pub"
            if default_pub.exists():
                public_key = default_pub.read_text(encoding="utf-8").strip()
    if not public_key:
        raise RuntimeError("SSH public key not found. Set SSH_PUBLIC_KEY or --ssh-public-key.")

    private_key_path = args.ssh_private_key or os.getenv("SSH_PRIVATE_KEY_PATH")
    if private_key_path:
        private_key_file = Path(private_key_path).expanduser()
    else:
        private_key_file = Path.home() / ".ssh" / "id_ed25519"
    if not private_key_file.exists():
        raise RuntimeError(f"SSH private key not found: {private_key_file}")
    return public_key, private_key_file


def load_private_key(path: Path) -> paramiko.PKey:
    try:
        return paramiko.Ed25519Key.from_private_key_file(str(path))
    except paramiko.SSHException:
        return paramiko.RSAKey.from_private_key_file(str(path))


def wait_for_ssh_port(
    client: RunpodGraphQLClient, pod_id: str, timeout_min: int
) -> Tuple[str, int]:
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        ports = client.get_pod_ports(pod_id)
        for port in ports:
            if int(port.get("privatePort") or 0) == 22:
                ip = port.get("ip")
                public_port = port.get("publicPort")
                if ip and public_port:
                    return ip, int(public_port)
        time.sleep(10)
    raise TimeoutError("SSH port did not become available in time.")


def build_grpo_config(args: argparse.Namespace, hub_model_id: str) -> Dict[str, Any]:
    default_reasoning = args.reasoning_tag == "<think>" and args.reasoning_end_tag == "</think>"
    default_answer = args.answer_tag == "<answer>" and args.answer_end_tag == "</answer>"
    use_default_tags = default_reasoning and default_answer
    if use_default_tags:
        reward_funcs = ["accuracy", "format", "tag_count"]
    else:
        reward_funcs = ["accuracy"]

    report_to = [] if args.report_to == "none" else [args.report_to]
    output_dir = args.output_dir or f"/workspace/models/{hub_model_id.replace('/', '-')}"

    if args.reasoning_tag and args.reasoning_end_tag:
        reasoning_instruction = (
            f"First reason inside {args.reasoning_tag} and {args.reasoning_end_tag}."
        )
    else:
        reasoning_instruction = "First think step by step."

    if args.answer_tag and args.answer_end_tag:
        answer_instruction = (
            f"Then provide the final answer inside {args.answer_tag} and {args.answer_end_tag}."
        )
    else:
        answer_instruction = "Then provide the final answer directly after thinking."

    system_prompt = (
        "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
        f"{reasoning_instruction} {answer_instruction}"
    )

    return {
        "model_name_or_path": args.repo_id,
        "model_revision": "main",
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "dataset_name": args.dataset_name,
        "dataset_prompt_column": args.dataset_prompt_column,
        "system_prompt": system_prompt,
        "bf16": True,
        "use_vllm": True,
        "do_eval": False,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "hub_model_id": hub_model_id,
        "hub_strategy": "every_save",
        "learning_rate": 1.0e-06,
        "log_completions": True,
        "log_level": "info",
        "logging_first_step": True,
        "logging_steps": 1,
        "logging_strategy": "steps",
        "lr_scheduler_type": "cosine_with_min_lr",
        "lr_scheduler_kwargs": {"min_lr_rate": 0.1},
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": 16,
        "num_train_epochs": args.num_train_epochs,
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.per_device_train_batch,
        "push_to_hub": True,
        "report_to": report_to,
        "reward_funcs": reward_funcs,
        "reward_weights": [1.0] * len(reward_funcs),
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "seed": 42,
        "temperature": 0.7,
        "use_liger_kernel": True,
        "warmup_ratio": 0.1,
    }


def upload_text(ssh: paramiko.SSHClient, remote_path: str, content: str) -> None:
    sftp = ssh.open_sftp()
    with sftp.file(remote_path, "w") as handle:
        handle.write(content)
    sftp.close()


def run_ssh_command(
    ssh: paramiko.SSHClient,
    command: str,
    reporter: ProgressReporter,
    stage: str,
) -> None:
    transport = ssh.get_transport()
    if not transport:
        raise RuntimeError("SSH transport not available.")
    channel = transport.open_session()
    channel.get_pty()
    channel.exec_command(command)

    buffer = ""
    last_report = 0.0

    def handle_line(line: str) -> None:
        nonlocal last_report
        now = time.time()
        if now - last_report >= 1.0:
            reporter.send("training_progress", line, extra={"stage": stage})
            last_report = now

    while True:
        if channel.recv_ready():
            data = channel.recv(4096).decode("utf-8", errors="ignore")
            buffer += data
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line.strip():
                    handle_line(line.strip())
        if channel.recv_stderr_ready():
            data = channel.recv_stderr(4096).decode("utf-8", errors="ignore")
            buffer += data
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                if line.strip():
                    handle_line(line.strip())
        if channel.exit_status_ready():
            break
        time.sleep(0.2)

    if buffer.strip():
        handle_line(buffer.strip())

    exit_status = channel.recv_exit_status()
    if exit_status != 0:
        raise RuntimeError(f"Remote command failed with exit status {exit_status}")


def build_setup_command() -> str:
    return (
        "bash -lc \""
        "set -euo pipefail; "
        "if [ ! -d /opt/open-r1 ]; then "
        "apt-get update; "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y "
        "git git-lfs curl build-essential python3-dev ninja-build ca-certificates; "
        "git lfs install; "
        "git clone https://github.com/huggingface/open-r1 /opt/open-r1; "
        "fi; "
        "if [ ! -d /opt/openr1-venv ]; then "
        "python -m pip install --upgrade pip; "
        "python -m pip install uv; "
        "uv venv /opt/openr1-venv --python 3.11; "
        "fi; "
        "source /opt/openr1-venv/bin/activate; "
        "cd /opt/open-r1; "
        "uv pip install --upgrade pip; "
        "uv pip install vllm==0.8.5.post1; "
        "uv pip install setuptools; "
        "uv pip install flash-attn --no-build-isolation; "
        "GIT_LFS_SKIP_SMUDGE=1 uv pip install -e \\\".[dev]\\\"; "
        "\""
    )


def build_train_command(config_path: str, vllm_mode: str) -> str:
    return (
        "bash -lc \""
        "set -euo pipefail; "
        "source /opt/openr1-venv/bin/activate; "
        "cd /opt/open-r1; "
        "ACCELERATE_LOG_LEVEL=info "
        "accelerate launch --config_file recipes/accelerate_configs/zero3.yaml "
        f"src/open_r1/grpo.py --config {config_path} --vllm_mode {vllm_mode}"
        "\""
    )


def main() -> int:
    args = parse_args()
    if args.use_thinnka_image:
        args.image = CUSTOM_IMAGE

    if bool(args.answer_tag) ^ bool(args.answer_end_tag):
        raise RuntimeError("Answer tags must include both start and end or be empty.")
    if bool(args.reasoning_tag) ^ bool(args.reasoning_end_tag):
        raise RuntimeError("Reasoning tags must include both start and end or be empty.")

    if args.gpu_count not in ALLOWED_GPU_COUNTS:
        raise RuntimeError("GPU count must be one of 1, 2, 4, 6, 8.")

    runpod_api_key = os.getenv("RUNPOD_API_KEY")
    if not runpod_api_key:
        raise RuntimeError("RUNPOD_API_KEY is required.")

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required.")

    ssh_public_key, ssh_private_key_path = ensure_ssh_key_material(args)
    progress_url = args.progress_url or os.getenv("PROGRESS_WEBHOOK_URL")

    reporter = ProgressReporter(progress_url, {"project": PROJECT_NAME, "repo_id": args.repo_id})
    ssh_client: Optional[paramiko.SSHClient] = None

    try:
        reporter.send("start", "Starting run.")

        hf_api = HfApi()
        model_info = ensure_model_not_gated(hf_api, args.repo_id, hf_token)
        model_size_gb = compute_model_size_gb(model_info)
        required_vram_gb = math.ceil(model_size_gb)

        reporter.send(
            "model_checked",
            f"Model size {model_size_gb:.2f} GB, required VRAM {required_vram_gb} GB.",
        )

        client = RunpodGraphQLClient(runpod_api_key)
        gpu_types = client.get_gpu_types()

        candidates = list(select_gpu_candidates(gpu_types, required_vram_gb))
        if not candidates:
            raise RuntimeError("No eligible GPU type meets the VRAM requirement.")

        volume_gb = args.volume_gb or max(40, int(math.ceil(model_size_gb * 2)))
        container_disk_gb = args.container_disk_gb or max(40, int(math.ceil(model_size_gb * 1.5)))

        pod_name = args.pod_name or f"thinnka-{args.repo_id.replace('/', '-')}-{args.gpu_count}x"
        env = build_pod_env(ssh_public_key, hf_token)

        pod = None
        last_error = None
        for label, gpu in candidates:
            reporter.send("gpu_try", f"Trying {label} ({gpu.get('displayName')}).")
            input_data = {
                "cloudType": args.cloud_type,
                "gpuCount": args.gpu_count,
                "volumeInGb": volume_gb,
                "containerDiskInGb": container_disk_gb,
                "minVcpuCount": args.min_vcpu,
                "minMemoryInGb": args.min_memory,
                "gpuTypeId": gpu.get("id"),
                "name": pod_name,
                "imageName": args.image,
                "dockerArgs": "",
                "ports": "22/tcp,8888/http",
                "volumeMountPath": "/workspace",
                "env": env,
            }
            if args.dry_run:
                reporter.send("dry_run", f"Selected GPU {gpu.get('displayName')}.")
                return 0
            try:
                pod = client.create_pod(input_data)
                reporter.send("pod_created", f"Pod {pod.get('id')} created with {gpu.get('displayName')}.")
                break
            except Exception as exc:
                last_error = exc
                reporter.send("gpu_fail", f"{label} failed: {exc}")

        if pod is None:
            raise RuntimeError(f"Unable to create a Pod. Last error: {last_error}")

        pod_id = pod.get("id")
        if not pod_id:
            raise RuntimeError("Pod creation did not return a pod id.")

        reporter.send("ssh_wait", "Waiting for SSH port 22.")
        ssh_host, ssh_port = wait_for_ssh_port(client, pod_id, args.ssh_timeout_min)
        reporter.send("ssh_ready", f"SSH available at {ssh_host}:{ssh_port}.")

        private_key = load_private_key(ssh_private_key_path)
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(
            hostname=ssh_host,
            port=ssh_port,
            username="root",
            pkey=private_key,
            timeout=30,
        )

        reporter.send("ssh_connected", "SSH connected.")

        hub_model_id = args.hub_model_id
        if not hub_model_id:
            whoami = hf_api.whoami(token=hf_token)
            user = whoami.get("name") or whoami.get("user") or "unknown"
            base_name = args.repo_id.split("/")[-1]
            hub_model_id = f"{user}/{base_name}-grpo-thinnka"

        config = build_grpo_config(args, hub_model_id)
        config_text = yaml.safe_dump(config, sort_keys=False)
        remote_dir = "/workspace/thinnka"
        remote_config_path = f"{remote_dir}/grpo_config.yaml"

        run_ssh_command(ssh_client, f"bash -lc \"mkdir -p {remote_dir}\"", reporter, "setup")
        upload_text(ssh_client, remote_config_path, config_text)
        reporter.send("config_uploaded", f"Config uploaded to {remote_config_path}.")

        if not args.skip_setup:
            reporter.send("setup_start", "Installing Open R1 on the Pod.")
            run_ssh_command(ssh_client, build_setup_command(), reporter, "setup")
            reporter.send("setup_done", "Open R1 setup completed.")
        else:
            reporter.send("setup_skip", "Skipping Open R1 setup.")

        reporter.send("train_start", "Starting GRPO training.")
        run_ssh_command(
            ssh_client,
            build_train_command(remote_config_path, args.vllm_mode),
            reporter,
            "train",
        )
        reporter.send("train_done", "Training finished.")
        reporter.send("done", "Run completed.")
        return 0
    except Exception as exc:
        reporter.send("error", str(exc))
        raise
    finally:
        if ssh_client:
            ssh_client.close()
        reporter.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
