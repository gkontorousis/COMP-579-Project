#!/usr/bin/env bash
set -uo pipefail

SSH_DIR="/root/.ssh"
AUTH_KEYS_FILE="${SSH_DIR}/authorized_keys"

log_section() {
  printf '\n=== %s ===\n' "$1"
}

log_section "COMP579 debug container startup"
date || true
echo "user: $(whoami 2>/dev/null || echo unknown)"
id || true
uname -a || true
echo "hostname: ${HOSTNAME:-unset}"
echo "pwd: $(pwd)"

log_section "Filesystem"
ls -ld /workspace /root /root/.ssh 2>/dev/null || true
df -h / /workspace 2>/dev/null || true

log_section "NVIDIA runtime"
nvidia-smi || true
ls -l /dev/nvidia* 2>/dev/null || true

log_section "TensorFlow"
python - <<'PY' || true
import sys
print("Python:", sys.version)
try:
    import tensorflow as tf
    print("TF:", tf.__version__)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("GPUs:", tf.config.list_physical_devices("GPU"))
except Exception as exc:
    print("TensorFlow check failed:", repr(exc))
PY

log_section "SSH key setup"
mkdir -p /run/sshd "${SSH_DIR}" || true
chmod 700 "${SSH_DIR}" 2>/dev/null || true
touch "${AUTH_KEYS_FILE}" 2>/dev/null || true
chmod 600 "${AUTH_KEYS_FILE}" 2>/dev/null || true

if [[ -f /workspace/.ssh/authorized_keys ]]; then
  echo "Using /workspace/.ssh/authorized_keys"
  cp /workspace/.ssh/authorized_keys "${AUTH_KEYS_FILE}" || true
  chmod 600 "${AUTH_KEYS_FILE}" 2>/dev/null || true
fi

if [[ -n "${SSH_PUBLIC_KEY:-}" ]]; then
  echo "Using SSH_PUBLIC_KEY environment variable"
  if ! grep -qxF "${SSH_PUBLIC_KEY}" "${AUTH_KEYS_FILE}" 2>/dev/null; then
    printf '%s\n' "${SSH_PUBLIC_KEY}" >> "${AUTH_KEYS_FILE}" || true
  fi
  chmod 600 "${AUTH_KEYS_FILE}" 2>/dev/null || true
fi

key_count="$(wc -l < "${AUTH_KEYS_FILE}" 2>/dev/null || echo 0)"
echo "authorized_keys line count: ${key_count}"
if [[ "${key_count}" = "0" ]]; then
  echo "Warning: no SSH public key configured; SSH login will fail until a key is provided."
fi

log_section "Starting sshd"
ssh-keygen -A || true
/usr/sbin/sshd -t || echo "Warning: sshd config test failed"

/usr/sbin/sshd -D -e
status=$?
echo "sshd exited with status ${status}; keeping container alive for platform diagnostics."
tail -f /dev/null
