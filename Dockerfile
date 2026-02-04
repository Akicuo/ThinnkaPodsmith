FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    openssh-server \
    curl \
    ca-certificates \
    build-essential \
    python3-dev \
    ninja-build \
  && rm -rf /var/lib/apt/lists/*

RUN git lfs install

RUN python -m pip install --upgrade pip && python -m pip install uv

RUN uv venv /opt/openr1-venv --python 3.11
ENV VIRTUAL_ENV=/opt/openr1-venv
ENV PATH="/opt/openr1-venv/bin:$PATH"

RUN git clone --depth 1 https://github.com/huggingface/open-r1 /opt/open-r1
WORKDIR /opt/open-r1

RUN uv pip install --upgrade pip \
  && uv pip install vllm==0.8.5.post1 \
  && uv pip install setuptools \
  && uv pip install flash-attn --no-build-isolation \
  && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
  && mkdir -p /run/sshd

EXPOSE 22
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash", "-lc", "sleep infinity"]
