#!/usr/bin/env bash
set -euo pipefail

# Runs the local TensorFlow GPU image with SSH enabled.
# Override defaults with environment variables:
#   IMAGE_NAME, CONTAINER_NAME, HOST_WORKSPACE, HOST_SSH_PORT
# Requires a root public key available via one of:
#   - build-time SSH_PUBLIC_KEY in Docker build
#   - /workspace/.ssh/authorized_keys (mounted from HOST_WORKSPACE)
#   - SSH_PUBLIC_KEY environment variable at runtime

IMAGE_NAME="${IMAGE_NAME:-comp579-tf-gpu:2.17.1}"
CONTAINER_NAME="${CONTAINER_NAME:-comp579-tf-dev}"
HOST_WORKSPACE="${HOST_WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}"
HOST_SSH_PORT="${HOST_SSH_PORT:-2222}"

docker run --rm -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  -p "${HOST_SSH_PORT}:22" \
  -v "${HOST_WORKSPACE}:/workspace" \
  -w /workspace \
  "${IMAGE_NAME}"

echo "Container started: ${CONTAINER_NAME}"
echo "SSH port mapping: localhost:${HOST_SSH_PORT} -> container:22"
echo "Connect with: ssh -p ${HOST_SSH_PORT} root@<docker-host>"
