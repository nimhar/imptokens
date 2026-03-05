#!/usr/bin/env bash
# imptokens installer
# Builds or downloads the binary, installs helpers, and optionally wires Claude Code hooks.
#
# One-liner usage (non-interactive, auto-detects GPU):
#   curl -fsSL https://raw.githubusercontent.com/nimhar/imptokens/main/install.sh | bash
#
# Interactive usage:
#   bash install.sh
#
# Flags:
#   -y / --yes   Skip all prompts (same as non-interactive)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-install.sh}")" 2>/dev/null && pwd || pwd)"
BIN_DIR="${HOME}/.local/bin"
BINARY="imptokens"
REPO="nimhar/imptokens"
VERSION="latest"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BOLD='\033[1m'; RESET='\033[0m'
ok()   { echo -e "${GREEN}✓${RESET} $*"; }
warn() { echo -e "${YELLOW}⚠${RESET}  $*"; }
die()  { echo -e "${RED}✗${RESET} $*" >&2; exit 1; }

# ── Non-interactive detection ─────────────────────────────────────────────────
# Auto-detect when piped (curl | bash) or when -y flag is passed.
INTERACTIVE=true
for arg in "$@"; do
  case "$arg" in -y|--yes|--non-interactive) INTERACTIVE=false ;; esac
done
[ ! -t 0 ] && INTERACTIVE=false   # stdin is not a tty (piped)
[ ! -t 1 ] && INTERACTIVE=false   # stdout is not a tty (piped)

prompt_yn() {
  # Usage: prompt_yn "Question?" default_answer
  # Returns 0 (yes) or 1 (no)
  local question="$1" default="${2:-n}"
  if ! $INTERACTIVE; then
    [[ "$default" =~ ^[Yy]$ ]] && return 0 || return 1
  fi
  read -r -p "$question [y/N] " ans
  [[ "$ans" =~ ^[Yy]$ ]]
}

# ── 1. Banner ─────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}imptokens installer${RESET}\n"

# ── 2. Detect GPU backend ─────────────────────────────────────────────────────
detect_feature() {
  if [[ "$(uname)" == "Darwin" ]]; then
    echo "metal"
  elif command -v nvcc &>/dev/null || [[ -d /usr/local/cuda ]]; then
    echo "cuda"
  elif command -v vulkaninfo &>/dev/null; then
    echo "vulkan"
  else
    echo ""
  fi
}

GPU_FEATURE="$(detect_feature)"
if [[ -n "$GPU_FEATURE" ]]; then
  ok "GPU backend detected: $GPU_FEATURE"
else
  warn "No GPU backend detected — will build/install CPU-only binary (slower inference)."
fi

# ── 3. Get the binary ─────────────────────────────────────────────────────────
OS="$(uname -s)"    # Darwin | Linux
ARCH="$(uname -m)"  # arm64 | x86_64
ARTIFACT="${BINARY}-${OS}-${ARCH}"
RELEASE_URL="https://github.com/${REPO}/releases/${VERSION}/download/${ARTIFACT}"

# Use a tmpdir for the downloaded/built binary so we don't pollute cwd.
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
BUILT_BINARY="${WORK_DIR}/${BINARY}"

# Try pre-built binary first (fast, no Rust required).
download_binary() {
  echo "Trying pre-built binary from GitHub Releases…"
  if curl -fsSL "$RELEASE_URL" -o "$BUILT_BINARY"; then
    chmod +x "$BUILT_BINARY"
    ok "Downloaded pre-built binary (${OS}/${ARCH})"
    return 0
  fi
  return 1
}

# Fall back to compiling from source.
build_from_source() {
  # Must be run from the repo root (not when piped via curl).
  if [[ ! -f "${SCRIPT_DIR}/Cargo.toml" ]]; then
    die "Cannot build from source: Cargo.toml not found in ${SCRIPT_DIR}.\n  Either download the repo first:\n    git clone https://github.com/${REPO}.git && cd imptokens && bash install.sh\n  Or wait for a pre-built binary at: https://github.com/${REPO}/releases"
  fi

  # Ensure Rust is available.
  if ! command -v cargo &>/dev/null; then
    if [[ -f "$HOME/.cargo/env" ]]; then
      # shellcheck source=/dev/null
      source "$HOME/.cargo/env"
    fi
  fi
  if ! command -v cargo &>/dev/null; then
    if [[ "$(uname)" == "Darwin" ]]; then
      die "Rust not found. Install it with:\n    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh\n  Or skip building entirely:\n    brew tap nimhar/imptokens https://github.com/nimhar/imptokens && brew install imptokens"
    else
      die "Rust not found. Install from https://rustup.rs/ and re-run."
    fi
  fi

  # Ensure cmake is available (required by llama-cpp-2 build).
  if ! command -v cmake &>/dev/null; then
    if [[ "$(uname)" == "Darwin" ]]; then
      die "cmake not found. Install it with:\n    brew install cmake\n  Or skip building entirely:\n    brew tap nimhar/imptokens https://github.com/nimhar/imptokens && brew install imptokens"
    else
      die "cmake not found. Install it with:\n    sudo apt install cmake   # Debian/Ubuntu\n    sudo dnf install cmake   # Fedora"
    fi
  fi

  # Ensure C/C++ toolchain is available (macOS needs Xcode CLT).
  if [[ "$(uname)" == "Darwin" ]] && ! xcode-select -p &>/dev/null; then
    die "Xcode Command Line Tools not found. Install with:\n    xcode-select --install\n  Or skip building entirely:\n    brew tap nimhar/imptokens https://github.com/nimhar/imptokens && brew install imptokens"
  fi

  ok "Rust $(rustc --version | cut -d' ' -f2) — building from source (~2 min on first run)…"
  local feature_flag=""
  [[ -n "$GPU_FEATURE" ]] && feature_flag="--features $GPU_FEATURE"
  cd "$SCRIPT_DIR"
  # shellcheck disable=SC2086
  cargo build --release --quiet $feature_flag
  cp "target/release/${BINARY}" "$BUILT_BINARY"
  ok "Binary built → ${BUILT_BINARY}"
}

# Offer Homebrew as zero-dependency alternative (macOS only).
try_brew_install() {
  [[ "$(uname)" != "Darwin" ]] && return 1
  if ! command -v brew &>/dev/null; then
    warn "Homebrew not found — cannot offer brew install fallback."
    return 1
  fi
  echo ""
  if prompt_yn "Install via Homebrew instead? (no Rust/cmake needed)" "n"; then
    brew tap nimhar/imptokens https://github.com/nimhar/imptokens
    brew install imptokens
    ok "Installed via Homebrew."
    # Homebrew puts it on PATH already; skip our manual install steps.
    INSTALLED_VIA_BREW=true
    return 0
  fi
  return 1
}

INSTALLED_VIA_BREW=false

if ! download_binary; then
  warn "Pre-built binary not available for ${OS}/${ARCH} — trying alternatives."
  if ! try_brew_install; then
    build_from_source
  fi
fi

# ── 4. Install binary ─────────────────────────────────────────────────────────
if [[ "$INSTALLED_VIA_BREW" != "true" ]]; then
mkdir -p "$BIN_DIR"
cp "$BUILT_BINARY" "${BIN_DIR}/${BINARY}"
ok "Installed → ${BIN_DIR}/${BINARY}"

# ── 5. Install helpers ────────────────────────────────────────────────────────

# compress-if-large: transparent compression filter for piped commands
cat > "${BIN_DIR}/compress-if-large" << 'SCRIPT'
#!/usr/bin/env bash
# Compress stdin if it exceeds MIN_CHARS. Falls back to raw output on error.
BINARY="$HOME/.local/bin/imptokens"
MIN_CHARS="${COMPRESS_MIN_CHARS:-4000}"
RATIO="${COMPRESS_RATIO:-0.6}"
content=$(cat)
if [ ${#content} -le "$MIN_CHARS" ] || [ ! -x "$BINARY" ]; then
  printf '%s' "$content"; exit 0
fi
compressed=$("$BINARY" --keep-ratio "$RATIO" 2>/dev/null <<< "$content")
if [ -n "$compressed" ]; then printf '%s' "$compressed"
else printf '%s' "$content"; fi
SCRIPT
chmod +x "${BIN_DIR}/compress-if-large"
ok "Installed → ${BIN_DIR}/compress-if-large"

# compress-paste: compress clipboard and put it back (macOS only)
if [[ "$(uname)" == "Darwin" ]]; then
  cat > "${BIN_DIR}/compress-paste" << 'SCRIPT'
#!/usr/bin/env bash
# Compress clipboard content and replace it, ready to paste.
# Usage: compress-paste [keep-ratio]   (default: 0.5)
BINARY="$HOME/.local/bin/imptokens"
RATIO="${1:-0.5}"
if [[ ! -x "$BINARY" ]]; then echo "Error: $BINARY not found." >&2; exit 1; fi
ORIGINAL=$(pbpaste)
EST_TOKENS=$(( ${#ORIGINAL} / 4 ))
if [[ $EST_TOKENS -lt 100 ]]; then
  echo "Text too short (~${EST_TOKENS} tokens), skipping." >&2; exit 0
fi
COMPRESSED=$(echo "$ORIGINAL" | "$BINARY" --keep-ratio "$RATIO" 2>/dev/null)
SAVED=$(( EST_TOKENS - ${#COMPRESSED} / 4 ))
PCT=$(( SAVED * 100 / EST_TOKENS ))
echo "$COMPRESSED" | pbcopy
echo "✓ Compressed ~${EST_TOKENS} → ~$(( ${#COMPRESSED} / 4 )) tokens (saved ~${PCT}%). Ready to paste." >&2
SCRIPT
  chmod +x "${BIN_DIR}/compress-paste"
  ok "Installed → ${BIN_DIR}/compress-paste"
fi
fi # end !INSTALLED_VIA_BREW

# ── 6. PATH check ─────────────────────────────────────────────────────────────
if ! echo "$PATH" | tr ':' '\n' | grep -qx "$BIN_DIR"; then
  warn "${BIN_DIR} is not in PATH. Add to ~/.zshrc or ~/.bashrc:"
  echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

# ── 7. Claude Code integration (optional) ─────────────────────────────────────
echo ""
if prompt_yn "Wire imptokens into Claude Code? (auto-compresses every large prompt)" "n"; then
  if command -v imptokens &>/dev/null || [[ -x "${BIN_DIR}/${BINARY}" ]]; then
    "${BIN_DIR}/${BINARY}" --setup-claude && ok "Claude Code hook configured."
  else
    warn "imptokens not in PATH yet — run 'imptokens --setup-claude' after restarting your shell."
  fi
fi

echo ""
if prompt_yn "Add /compress-paste slash command for Claude Code?" "n"; then
  mkdir -p "${HOME}/.claude/commands"
  cat > "${HOME}/.claude/commands/compress-paste.md" << 'CMD'
Run `compress-paste $ARGUMENTS` to compress the current clipboard content and
replace it with the compressed version, then report the result.
If no argument is provided, use the default ratio of 0.5.
CMD
  ok "Slash command /compress-paste registered."
fi

# ── 8. Done ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Installation complete!${RESET}"
echo ""
echo "  Quick start:"
echo "    echo 'your text' | imptokens --stats"
echo "    cat bigfile.txt | imptokens --keep-ratio 0.5"
if [[ "$(uname)" == "Darwin" ]]; then
  echo "    compress-paste                  # compress clipboard"
fi
echo ""
echo "  Wire into Claude Code at any time:"
echo "    imptokens --setup-claude"
echo ""
echo "  Restart your shell or run:  export PATH=\"\$HOME/.local/bin:\$PATH\""
echo ""
