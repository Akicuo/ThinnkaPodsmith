#!/usr/bin/env bash
set -euo pipefail

mkdir -p /run/sshd
mkdir -p /root/.ssh
chmod 700 /root/.ssh

if [ -n "${SSH_PUBLIC_KEY:-}" ]; then
  echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
elif [ -n "${PUBLIC_KEY:-}" ]; then
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
fi

if [ -f /root/.ssh/authorized_keys ]; then
  chmod 600 /root/.ssh/authorized_keys
fi

ssh-keygen -A >/dev/null 2>&1
/bin/sh -c "grep -q '^PermitRootLogin' /etc/ssh/sshd_config && \
  sed -i 's/^PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config || \
  echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config"
/usr/sbin/sshd

exec "$@"
