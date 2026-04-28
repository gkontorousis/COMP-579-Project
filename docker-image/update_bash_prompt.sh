#!/usr/bin/env bash
set -euo pipefail

# Updates Bash prompt to:
# username@[first-part-of-hostname-before-'-']:current_folder
# If current folder exceeds 25 chars, it is truncated.

TARGET_HOME="${1:-${TARGET_HOME:-$HOME}}"
TARGET_BASHRC="${TARGET_HOME}/.bashrc"
TARGET_BASH_PROFILE="${TARGET_HOME}/.bash_profile"
START_MARKER="# >>> custom short prompt >>>"
END_MARKER="# <<< custom short prompt <<<"

read -r -d '' PROMPT_BLOCK <<'EOF' || true
# >>> custom short prompt >>>
_custom_prompt_path_segment() {
  local dir_name
  dir_name="$(basename "$PWD")"
  if [ "${#dir_name}" -gt 25 ]; then
    dir_name="${dir_name:0:22}..."
  fi
  printf '%s' "$dir_name"
}

_custom_prompt_host_prefix() {
  local host_name
  host_name="${HOSTNAME:-$(hostname)}"
  printf '%s' "${host_name%%-*}"
}

PS1='\u@[$(_custom_prompt_host_prefix)]:$(_custom_prompt_path_segment)\$ '
# <<< custom short prompt <<<
EOF

ensure_prompt_block() {
  local file_path="$1"
  mkdir -p "$(dirname "$file_path")"
  touch "$file_path"

  # Remove any previous managed block and append a fresh one.
  awk -v start="$START_MARKER" -v end="$END_MARKER" '
    $0 == start { skip=1; next }
    $0 == end   { skip=0; next }
    !skip { print }
  ' "$file_path" > "${file_path}.tmp"

  {
    printf '\n%s\n' "$PROMPT_BLOCK"
  } >> "${file_path}.tmp"

  mv "${file_path}.tmp" "$file_path"
}

ensure_prompt_block "$TARGET_BASHRC"
ensure_prompt_block "$TARGET_BASH_PROFILE"

printf 'Updated prompt config in:\n- %s\n- %s\n' "$TARGET_BASHRC" "$TARGET_BASH_PROFILE"
printf 'Run: source %s\n' "$TARGET_BASHRC"
