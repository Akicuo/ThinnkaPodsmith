"""Constants for Thinnka Podsmith."""

PROJECT_NAME = "Thinnka Podsmith"
RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"
RUNPOD_REST_URL = "https://rest.runpod.io/v1"

DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
CUSTOM_IMAGE = "reeeon/thinnka:latest"

ALLOWED_GPU_COUNTS = {1, 2, 4, 6, 8}
WEIGHT_EXTENSIONS = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt"}

GPU_PRIORITY = [
    ("L40S", lambda name: "l40s" in name),
    ("A100 SXM", lambda name: "a100" in name and "sxm" in name),
    ("H100 SXM", lambda name: "h100" in name and "sxm" in name),
]

DEFAULT_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ message['content'] + '\\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ 'User: ' + message['content'] + '\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ 'Assistant: ' + message['content'] + '\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ 'Assistant: ' }}"
    "{% endif %}"
)

DISCORD_MAX_CONTENT = 2000
