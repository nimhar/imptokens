<h1 align="center">imptokens</h1>

<p align="center">
  <strong>Cut your Claude / GPT token costs by 30–60%.</strong><br/>
  Runs 100% locally. No data leaves your machine. No API. No subscription.
</p>

<p align="center">
  <a href="https://github.com/nimhar/imptokens/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-16a34a"></a>
  <a href="https://github.com/nimhar/imptokens"><img alt="Platform" src="https://img.shields.io/badge/platform-macOS%20%7C%20Windows%20%7C%20Linux-111827"></a>
  <a href="https://www.rust-lang.org/"><img alt="Rust" src="https://img.shields.io/badge/rust-1.70%2B-f97316"></a>
  <a href="https://github.com/nimhar/imptokens/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/nimhar/imptokens?style=flat-square&logo=github&label=Stars"></a>
</p>

![imptokens demo](video/out/Imptokens.gif)

```bash
curl -fsSL https://raw.githubusercontent.com/nimhar/imptokens/main/install.sh | bash
```

Then wire it into Claude Code once — and never think about it again:

```bash
imptokens --setup-claude
# ✓ Claude Code hook configured
#   Every prompt over 500 tokens is automatically compressed before it reaches the model.
#   Restart Claude Code to activate.
```

`imptokens` has two compression engines: a **sentence mode** that needs no model (sub-5ms, term-overlap scoring) and a **logprob mode** that uses a small local model for token-level scoring. Both can cut context size 30–70% with no meaningful quality loss.

```bash
git diff HEAD~5 | imptokens --threshold=-0.075 --stats
# tokens: 392/624 kept  (37.2% reduction, 232 saved)
```

## Table of contents

- [Why imptokens](#why-imptokens)
- [Demo](#demo)
- [Real workflow impact](#real-workflow-impact)
- [Quick start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [CLI reference](#cli-reference)
- [How it works](#how-it-works)
- [Sentence mode](#sentence-mode)
- [Quality benchmark](#quality-benchmark)
- [Performance](#performance)
- [Claude Code integration](#claude-code-integration)
- [Model guide](#model-guide)
- [Examples](#examples)
- [Author](#author)
- [Contributing](#contributing)
- [Architecture](#architecture)
- [Roadmap](#roadmap)
- [License](#license)

## Why imptokens

`imptokens` is a **general-purpose LLM context preprocessor**. It compresses text before it reaches any language model — Claude, GPT-4, Gemini, local Ollama, or anything behind a LiteLLM proxy. It's not a Claude Code feature: Claude Code is just one of several supported workflows.

- Works in any shell pipeline — no framework or SDK required
- Designed for AI-heavy developer workflows: diffs, logs, long docs, and generated outputs
- Fully local execution on Apple Silicon via `llama.cpp` + Metal (also CUDA, Vulkan, CPU)
- CLI-first design for shell pipelines, agent hooks, and LLM API wrappers
- Multiple output formats (`text`, `token-ids`, `json`) and token-level debug view
- Includes Claude Code integration helpers (`compress-if-large`, `compress-paste`, hook mode) — optional

## Real workflow impact

These are practical “before -> after” examples that show what the project contributes in day-to-day work:

### 1) Large git diff before review

```bash
git diff HEAD~8 | imptokens --threshold=-0.075 --stats
# Keeps key symbols and changed lines while cutting context cost
```

### 2) Long error output before asking Claude

```bash
pytest -q 2>&1 | imptokens --threshold=-0.075 --stats
# Preserves stack anchors and failure hints, trims repeated noise
```

### 3) Pre-compress clipboard for Claude prompts

```bash
compress-paste 0.5
# Paste a denser prompt with lower token spend
```

### 4) Optional RTK hook flow (only if RTK is installed)

```text
rtk read bigfile.py | compress-if-large
rtk git diff HEAD~5 | compress-if-large
```

### 5) Hook-mode compression for UserPromptSubmit events

```bash
echo '{"prompt":"...very long prompt..."}' | imptokens --hook-mode --hook-threshold 500
```

## Quick start

```bash
# Install (auto-detects GPU, no Rust required)
curl -fsSL https://raw.githubusercontent.com/nimhar/imptokens/main/install.sh | bash

# Wire into Claude Code — one time, then forget about it
imptokens --setup-claude

# Try it
echo "Your long text goes here" | imptokens --threshold=-0.075 --stats
```

## Installation

### One-liner (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/nimhar/imptokens/main/install.sh | bash
```

- Downloads a pre-built binary for your platform (no Rust needed)
- Falls back to compiling from source if no binary is available
- Auto-detects GPU backend: Metal → CUDA → Vulkan → CPU
- Installs to `~/.local/bin`

### Homebrew (macOS)

```bash
brew tap nimhar/imptokens https://github.com/nimhar/imptokens
brew install imptokens
```

### Build from source

Requires Rust 1.70+:

```bash
# macOS Apple Silicon (Metal)
cargo build --release --features metal

# NVIDIA GPU — requires CUDA toolkit installed
cargo build --release --features cuda

# Cross-platform Vulkan (Windows, Linux, etc.)
cargo build --release --features vulkan

# CPU-only fallback (any platform, no GPU required)
cargo build --release
```

The default model is downloaded once and cached at `~/.cache/huggingface/hub/`.

## Usage

### Inline text

```bash
imptokens "Your long text here" --stats
```

### From file

```bash
imptokens --file document.txt --threshold=-0.075
```

### From stdin

```bash
cat bigfile.py | imptokens --threshold=-0.075
git diff HEAD~5 | imptokens --threshold=-0.075 --stats
curl -s https://example.com/api | imptokens --threshold=-0.15   # JSON: push harder
```

### Output formats

```bash
imptokens "text" --output-format text
imptokens "text" --output-format token-ids
imptokens "text" --output-format json
imptokens "text" --debug
```

## CLI reference

### Input

| Flag | Description |
|---|---|
| `[TEXT]` | Positional text input |
| `-f, --file <PATH>` | Read from file (`-` for stdin) |

### Model

| Flag | Default | Description |
|---|---|---|
| `-m, --model <REPO>` | `bartowski/Llama-3.2-1B-Instruct-GGUF` | HuggingFace repo ID |
| `--model-file <FILE>` | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | GGUF file in the repo |
| `--local-model <PATH>` | none | Use local GGUF path |

### Compression strategy (logprob mode)

Use one strategy at a time. Requires model download on first run.

| Flag | Default | Description |
|---|---|---|
| `-t, --threshold <LOGPROB>` | `-1.0` | Keep tokens where `logprob < threshold` (recommended: `-0.075`) |
| `-k, --keep-ratio <RATIO>` | none | Keep top most-surprising fraction (`0 < ratio <= 1`) |

### Sentence mode (no model required)

| Flag | Default | Description |
|---|---|---|
| `--sentence-mode` | off | Enable sentence-level query-relevant extraction |
| `--query <TEXT>` | `""` | Question or key terms used for relevance scoring |
| `--target-reduction <RATIO>` | `0.1` | Fraction of tokens to remove (`0 < ratio <= 1`) |
| `--max-sentences <N>` | `10` | Maximum sentences to keep |

### Output and diagnostics

| Flag | Description |
|---|---|
| `-o, --output-format <FMT>` | `text`, `token-ids`, or `json` |
| `-d, --debug` | Full per-token JSON (sentence mode: adds sentence counts) |
| `-s, --stats` | Print compression stats to stderr |

### Hook mode

| Flag | Default | Description |
|---|---|---|
| `--hook-mode` | `false` | Claude Code UserPromptSubmit hook mode |
| `--hook-threshold <N>` | `500` | Min estimated tokens before compression |

### Claude Code setup

| Flag | Description |
|---|---|
| `--setup-claude` | Write hook config to `~/.claude/settings.json` |

## How it works

`imptokens` runs a small local model to assign an **information-density score** to every token in your input. Tokens that carry signal are kept. Tokens that are structurally predictable — boilerplate, repeated patterns, filler — are dropped.

The scoring is context-aware: the same word can be high-value in one position and redundant in another, depending on what surrounds it. This is what separates it from keyword filtering or stopword removal, which are context-blind.

```text
Input:   The model predicts the next token. The model predicts the next word.
kept?:    Y     Y      Y     Y    Y    Y      Y    Y      Y     N    N    Y
Output:  The model predicts the next token. The model predicts word.
```

The second sentence is mostly predictable repetition, so only the novel ending is retained. The scoring runs in a single local forward pass — no generation, no sampling, no network calls.

## Sentence mode

Sentence mode extracts the most relevant sentences from a context, with no model download required. It scores each sentence by term overlap with your query and a length-normalization bonus, then drops low-value sentences until the target reduction is met.

```bash
# Extract the most relevant sentences from a long document (10% reduction by default)
cat document.txt | imptokens --sentence-mode --query "what is the API rate limit?"

# More aggressive: remove 45% of tokens
git diff HEAD~5 | imptokens --sentence-mode --target-reduction 0.45 --stats
```

**Scoring formula** (per sentence):

```
score = (overlap × 2)
      + (overlap / queryTermCount)
      + min(sentenceLength, 40) / 40
      - boilerplatePenalty
```

Where `overlap` = number of query terms found in the sentence (stop words excluded). Sentences with identical normalized content are deduplicated before scoring. Results are returned in original document order.

**When to use sentence mode vs logprob mode:**

| | Sentence mode | Logprob mode |
|---|---|---|
| Model required | No | Yes (~700 MB download) |
| Speed | <5ms | ~0.5–15s (GPU) |
| Input | Any text | Any text |
| Best for | Long docs with a clear query | Diffs, logs, code |
| Strategy | Relevance extraction | Information density |

Use `--sentence-mode` when you have a specific question and want fast, zero-dependency compression. Use the default logprob mode when you want density-based compression without a specific query.

## Quality benchmark

### Sentence mode (no model required)

Measured with `examples/03_quality_benchmark.py --mode sentence`.

| Text type | Target reduction | Actual reduction | Key-phrase survival | Latency |
|---|---:|---:|---:|---:|
| Dense technical prose | 10% | ~10% | ~85% | <5ms |
| Repetitive documentation | 10% | ~10% | ~90% | <5ms |
| Git diff output | 45% | ~45% | ~92% | <5ms |
| Error log / stack trace | 10% | ~10% | ~88% | <5ms |

Practical defaults:

- `--target-reduction 0.1` for general use (light trim, keeps almost everything)
- `--target-reduction 0.45` for long repetitive docs where you have a clear query

### Logprob mode — threshold sweep (local model)

Measured with `examples/06_claude_quality_benchmark.py` — rigorous key-fact verification via Claude judge, not heuristic key-phrase matching.

| Threshold | Token reduction | Key-fact coverage | Verdict |
|---:|---:|---:|:---:|
| −0.025 | 29.6% | 97% | MARGINAL |
| −0.050 | 34.4% | 96% | PASS ✓ |
| **−0.075** | **37.2%** | **95%** | **PASS ✓** |
| −0.100 | 41.8% | 90% | PASS ✓ |
| −0.200 | 52.4% | 78% | MARGINAL |
| −0.500 | 63.1% | 61% | FAIL |

**`--threshold=-0.075` is the recommended default**: 37% reduction with 95% key-fact retention.

### By content type (at −0.075)

| Input type | Token reduction | Key-fact coverage |
|---|---:|---:|
| Library README (long doc) | 37% | 100% |
| Production log / error cascade | 37% | 100% |
| Architecture Decision Record | 37% | 95% |
| Application settings file | 37% | 88% |
| Incident post-mortem | 37% | 83% |

### Latency overhead

| Step | Time |
|---|---:|
| imptokens compression (local) | ~280 ms |
| LLM API (full context) | ~1,840 ms |
| LLM API (compressed) | ~1,580 ms |
| Net overhead | **~20 ms** |

Compression is nearly free in wall-clock time — the API speedup from shorter input offsets most of the compression cost.

Run locally:

```bash
python3 examples/03_quality_benchmark.py                      # sentence mode (fast, no model)
python3 examples/06_claude_quality_benchmark.py               # threshold sweep with Claude judge
python3 examples/02_token_viz.py --ratio 0.5
```

## Performance

`imptokens` only runs a single forward pass (no generation) so throughput is dominated by prompt-processing speed, which is the fast path in llama.cpp.

### Tokens per second (Llama 3.2 1B Q4\_K\_M)

| Hardware | Estimated throughput |
|---|---|
| Apple M2 / M3 (Metal) | ~2,000–4,000 tok/s |
| Apple M1 (Metal) | ~1,000–2,000 tok/s |
| NVIDIA RTX 3090 (CUDA) | ~3,000–6,000 tok/s |
| CPU-only (modern laptop) | ~200–500 tok/s |

### Wall-clock time for common input sizes (Apple M2, Metal)

| Input | Tokens | Time |
|---|---:|---|
| Typical diff | ~2K | ~0.5–1s |
| Large diff / log | ~10K | ~2–5s |
| Full source file | ~30K | ~7–15s |
| Long document | ~50K | ~12–25s |
| Model context limit | ~128K | ~30–65s |

> **Note:** the default model (Llama 3.2 1B) supports up to 128K tokens. Inputs beyond that require chunking — see roadmap.

### KV cache memory at scale

The KV cache is allocated proportional to input size. At 128K tokens it requires ~2–4 GB of VRAM depending on the model. A 1M-token input would require ~32 GB — not feasible on consumer hardware today without chunking.

## Use with any LLM

`imptokens` is a plain stdin/stdout filter. It doesn't know or care which model receives the output. You can pipe compressed text into any LLM API call.

### OpenAI / GPT-4

```python
import subprocess, openai

def compress(text: str, threshold: float = -0.075) -> str:
    result = subprocess.run(
        ["imptokens", "--file", "-", f"--threshold={threshold}"],
        input=text, capture_output=True, text=True, check=True,
    )
    return result.stdout

context = open("long_doc.txt").read()
compressed = compress(context)

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": compressed + "\n\nSummarize the above."}],
)
```

### LiteLLM (any provider)

```python
import litellm, subprocess

def compress(text):
    r = subprocess.run(["imptokens", "--file", "-", "--threshold=-0.075"],
                       input=text, capture_output=True, text=True, check=True)
    return r.stdout

response = litellm.completion(
    model="anthropic/claude-haiku-4-5-20251001",
    messages=[{"role": "user", "content": compress(long_context)}],
)
```

### Shell — works before any LLM CLI

```bash
# Compress once, use anywhere
cat context.md | imptokens --threshold=-0.075 > compressed.md

# OpenAI CLI, llm, aichat, sgpt — all work the same way
cat compressed.md | llm "What are the key points?"
cat compressed.md | sgpt "Summarize"
```

> **Framework integrations** (LangGraph, OpenAI Agents SDK, Google ADK, LiteLLM middleware) are on the roadmap — see below.

## Claude Code integration

### Automatic compression for every prompt (recommended)

One command wires `imptokens` into Claude Code as a `UserPromptSubmit` hook. Every prompt above the threshold is silently compressed before it reaches the model — you don't change anything about how you work.

```bash
imptokens --setup-claude
```

```
✓ Claude Code hook configured
  File:      ~/.claude/settings.json
  Threshold: 500 estimated tokens
  Ratio:     60% of tokens kept

Restart Claude Code for changes to take effect.
```

What it adds to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "matcher": "",
      "hooks": [{ "type": "command", "command": "/path/to/imptokens --hook-mode --hook-threshold 500 --keep-ratio 0.6" }]
    }]
  }
}
```

Adjust the threshold and aggressiveness:

```bash
# Conservative: only compress large prompts (recommended starting point)
imptokens --setup-claude --hook-threshold 1000 --keep-ratio 0.7

# Balanced: compress anything over 500 tokens (default)
imptokens --setup-claude

# Aggressive: compress anything over 200 tokens
imptokens --setup-claude --hook-threshold 200 --keep-ratio 0.5
```

> **Latency note:** the hook adds ~2–4s on Apple Silicon (model cold-start). Only triggers above your threshold.

### Clipboard pre-compression (macOS)

```bash
# 1) Copy text to clipboard
compress-paste        # default keep-ratio 0.5
compress-paste 0.3    # more aggressive
# 2) Paste into Claude — already compressed
```

### Pipe any command output through compression

```bash
cat bigfile.py   | compress-if-large
git diff HEAD~5  | compress-if-large
pytest 2>&1      | compress-if-large
```

Tune behavior:

```bash
export COMPRESS_MIN_CHARS=2000   # only compress if > 2000 chars
export COMPRESS_RATIO=0.5        # keep 50% of tokens
```

### Slash command

```bash
/compress-paste          # compress clipboard, registered by install.sh
```

### Hook mode (raw API)

```bash
echo '{"prompt":"...long text..."}' | imptokens --hook-mode --hook-threshold 500
```

- Claude Code docs: https://docs.anthropic.com/en/docs/claude-code/hooks

## Model guide

Any GGUF model can be used.

| Model | Size | Speed | Quality |
|---|---|---|---|
| `Llama-3.2-1B-Instruct-Q4_K_M` | ~700 MB | Fast | Good default |
| `Llama-3.2-3B-Instruct-Q4_K_M` | ~1.8 GB | Medium | Better semantic retention |
| `Qwen2.5-1.5B-Instruct-Q4_K_M` | ~900 MB | Medium-fast | Good |

Use explicit model selection:

```bash
imptokens --model bartowski/Llama-3.2-3B-Instruct-GGUF --model-file Llama-3.2-3B-Instruct-Q4_K_M.gguf "text"
```

Or local model file:

```bash
imptokens --local-model /path/to/model.gguf "text"
```

## Examples

| File | Description |
|---|---|
| `examples/01_basic.sh` | Basic threshold vs keep-ratio comparison |
| `examples/02_token_viz.py` | Token-level color visualization |
| `examples/03_quality_benchmark.py` | Compression quality benchmark (sentence + logprob modes) |
| `examples/04_demo.py` | Demo report generation |
| `examples/05_qa_demo.py` | QA preservation demo |
| `examples/06_claude_quality_benchmark.py` | Claude API quality benchmark: threshold sweep with key-fact verification judge |

## Author

Built by [Nimrod Harel](https://x.com/harel_nimrod). If this saves you tokens, a star or a mention is appreciated.

To cite this work:

```bibtex
@software{harel2025imptokens,
  author  = {Harel, Nimrod},
  title   = {imptokens: Local Logprob-Based Token Compression for LLM Pipelines},
  year    = {2025},
  url     = {https://github.com/nimhar/imptokens},
  license = {MIT}
}
```

## Contributing

Contributions are welcome. Open an issue or PR — feedback on quality benchmarks and new framework integrations especially welcome.

### Good first contributions

- Add benchmark datasets and report scripts for real-world traces.
- Improve Claude/RTK integration docs and setup robustness.
- Add tests around compression strategy edge cases.
- Add new backend experiments while keeping CLI compatibility.

### Local dev loop

```bash
cargo build
cargo run -- --help
python3 examples/03_quality_benchmark.py
```

## Architecture

```text
imptokens/
├── src/
│   ├── main.rs          # CLI + hook mode + sentence mode dispatch
│   ├── lib.rs           # public API
│   ├── compressor.rs    # Compressor + Strategy
│   ├── result.rs        # CompressResult and stats
│   ├── threshold.rs     # fixed_threshold / target_ratio / target_count
│   ├── sentence.rs      # sentence-level extraction (no model required)
│   ├── cache.rs         # provider-side prompt cache simulation
│   ├── benchmark.rs     # workload benchmark runner with memoization
│   └── backend/
│       ├── mod.rs       # Backend trait
│       └── llama.rs     # llama.cpp backend (Metal / CUDA / Vulkan / CPU)
└── examples/
```

`Backend` is intentionally abstracted so additional inference backends can be added without changing CLI UX. `sentence.rs` is zero-dependency — no model, no network, no allocation beyond the input string.

## Roadmap

### Framework integrations

imptokens is a CLI tool today, but the same compression can be a drop-in middleware layer for any LLM framework. Planned integrations:

- **LangChain / LangGraph** — `ImptokensContextCompressor` as a document transformer or retriever wrapper
- **OpenAI Agents SDK** — preprocessing hook that compresses tool outputs before they re-enter the context window
- **Google ADK** — context compression step in agent pipelines
- **LiteLLM middleware** — transparent proxy-layer compression for any model behind LiteLLM
- **LlamaIndex** — `ImptokensNodePostprocessor` for RAG pipelines

If you want any of these sooner, open an issue or PR.

### Core

- **Auto-chunking for large inputs** — sliding-window chunking with overlap to support inputs beyond the model's context limit (targeting 1M+ tokens via chunk-and-merge)
- **Persistent daemon mode** — keep the model loaded between calls to eliminate the ~2–4s cold-start overhead in hook mode
- **Streaming mode** (`--stream`) for chunked large inputs with incremental output
- **CoreML backend** — Apple Neural Engine exploration for lower power draw on macOS
- **JSON/JSONL selective field compression** — score and compress individual fields rather than treating JSON as flat text
- **Embedding-based sentence scoring** — replace term-overlap with cosine similarity for better semantic recall in sentence mode
- **Editor integrations** — Cursor, VS Code extension, Neovim, GitHub Actions

## License

MIT
