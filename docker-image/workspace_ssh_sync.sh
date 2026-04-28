#!/usr/bin/env bash
# Run after login: copy keys from /workspace/.ssh into ~/.ssh and add GitHub SSH config.
set -euo pipefail

SRC="/workspace/.ssh"
MARKER="# workspace_ssh_sync"

mkdir -p "${HOME}/.ssh"
chmod 700 "${HOME}/.ssh"

if [[ -d "${SRC}" ]]; then
  shopt -s nullglob
  files=( "${SRC}"/* )
  shopt -u nullglob
  for f in "${files[@]}"; do
    [[ -f "$f" ]] || continue
    cp -f "$f" "${HOME}/.ssh/$(basename "$f")"
  done
else
  echo "Warning: ${SRC} not found" >&2
fi

find "${HOME}/.ssh" -maxdepth 1 -type f ! -name '*.pub' ! -name 'config' ! -name 'known_hosts' ! -name 'authorized_keys' -exec chmod 600 {} \; 2>/dev/null || true
find "${HOME}/.ssh" -maxdepth 1 -type f -name '*.pub' -exec chmod 644 {} \; 2>/dev/null || true

touch "${HOME}/.ssh/config"
chmod 600 "${HOME}/.ssh/config"
if ! grep -qF "${MARKER}" "${HOME}/.ssh/config" 2>/dev/null; then
  cat >> "${HOME}/.ssh/config" <<'EOF'

# workspace_ssh_sync
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/github
  IdentitiesOnly yes
EOF
fi

echo "Done: keys from ${SRC} -> ${HOME}/.ssh, GitHub block in config."
