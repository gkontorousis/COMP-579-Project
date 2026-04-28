#!/usr/bin/env bash
set -euo pipefail

SSH_DIR="/root/.ssh"
AUTH_KEYS_FILE="${SSH_DIR}/authorized_keys"

mkdir -p /run/sshd "${SSH_DIR}"
chmod 700 "${SSH_DIR}"
touch "${AUTH_KEYS_FILE}"
chmod 600 "${AUTH_KEYS_FILE}"

# Option 1: mount an existing authorized_keys file at /workspace/.ssh/authorized_keys.
if [[ -f /workspace/.ssh/authorized_keys ]]; then
  cp /workspace/.ssh/authorized_keys "${AUTH_KEYS_FILE}"
  chmod 600 "${AUTH_KEYS_FILE}"
fi

# Option 2: pass a public key in SSH_PUBLIC_KEY env var.
if [[ -n "${SSH_PUBLIC_KEY:-}" ]]; then
  if ! grep -qxF "${SSH_PUBLIC_KEY}" "${AUTH_KEYS_FILE}"; then
    printf '%s\n' "${SSH_PUBLIC_KEY}" >> "${AUTH_KEYS_FILE}"
  fi
  chmod 600 "${AUTH_KEYS_FILE}"
fi

if [[ ! -s "${AUTH_KEYS_FILE}" ]]; then
  echo "Error: no SSH key configured for root." >&2
  echo "Provide a build-time key, /workspace/.ssh/authorized_keys, or SSH_PUBLIC_KEY env var." >&2
  exit 1
fi

ssh-keygen -A >/dev/null 2>&1
exec /usr/sbin/sshd -D -e
