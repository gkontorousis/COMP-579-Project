# TensorFlow GPU Image

This folder contains a reproducible Docker image setup for your COMP-579 work.

It is based on your diagnostics:
- Host OS/runtime: Ubuntu 22.04
- GPU: NVIDIA H100
- Host driver: 550.90.07 (CUDA 12.4 capability shown by `nvidia-smi`)

The previous container had a TensorFlow/CUDA/cuDNN mismatch (`tensorflow==2.19.0` but runtime libs were not aligned), which caused `GPUs: []`.

## Files

- `Dockerfile`: custom TensorFlow GPU image
- `Dockerfile.debug`: debug image that prints startup diagnostics and stays alive if `sshd` fails
- `run_tensorflow_container.sh`: convenience runner (starts SSH container)
- `workspace_ssh_sync.sh`: helper for syncing SSH keys from `/workspace/.ssh`
- `sshd_entrypoint.sh`: configures user key auth and starts `sshd`

## Build

From repository root:

```bash
cd docker-image
docker build \
  --build-arg SSH_PUBLIC_KEY="$(cat ~/.ssh/id_ed25519.pub)" \
  -t comp579-tf-gpu:2.18.0 \
  .
```

## Build Debug Tag

Use this when the platform fails before showing useful logs. The debug image prints startup diagnostics, attempts `nvidia-smi` and TensorFlow GPU detection, starts `sshd`, and keeps the container alive if `sshd` exits.

```bash
cd docker-image
docker buildx build \
  --platform linux/amd64 \
  --build-arg SSH_PUBLIC_KEY="$(cat ~/.ssh/fpt_ai.pub)" \
  -t <dockerhub-user>/<repo>:debug \
  -f Dockerfile.debug \
  --push \
  .
```

## Run

Public key options:
- bake it into the image at build time with `--build-arg SSH_PUBLIC_KEY=...` (recommended)
- or mount `/workspace/.ssh/authorized_keys`
- or set `SSH_PUBLIC_KEY` when running

```bash
bash docker-image/run_tensorflow_container.sh
```

By default, it maps host port `2222` to container port `22`.

SSH in:

```bash
ssh -p 2222 root@<docker-host>
```

## Quick GPU check after SSH login

```bash
python - <<'PY'
import tensorflow as tf
print("TF:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY
```

## SSH key sync helper

If your keys are persisted in `/workspace/.ssh` and you want outbound git SSH setup, run:

```bash
/usr/local/bin/workspace_ssh_sync.sh
```
