use std::io::{self, Read};
use std::path::PathBuf;

use anyhow::{Context, anyhow, bail};
use clap::{Parser, ValueEnum};

/// 100 MB — hard cap on stdin to prevent OOM from unbounded input.
const MAX_INPUT_BYTES: u64 = 100 * 1024 * 1024;

use imptokens::{
    Compressor, Strategy,
    backend::LlamaCppBackend,
    stats,
};

// ─── CLI definition ──────────────────────────────────────────────────────────

/// Fast logprob-based token compression (llama.cpp).
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Text to compress (mutually exclusive with --file).
    text: Option<String>,

    /// Read input from file ('-' for stdin).
    #[arg(short, long, value_name = "PATH")]
    file: Option<String>,

    // ── Model ─────────────────────────────────────────────────────────────
    /// HuggingFace repo id (e.g. bartowski/Llama-3.2-1B-Instruct-GGUF).
    #[arg(short, long, default_value = "bartowski/Llama-3.2-1B-Instruct-GGUF")]
    model: String,

    /// GGUF filename within the repo.
    #[arg(long, default_value = "Llama-3.2-1B-Instruct-Q4_K_M.gguf")]
    model_file: String,

    /// Use a local model file instead of downloading from HuggingFace.
    #[arg(long, value_name = "PATH")]
    local_model: Option<PathBuf>,

    // ── Strategy (mutually exclusive) ─────────────────────────────────────
    /// Keep tokens whose logprob is below this threshold [default: -1.0].
    #[arg(short, long, value_name = "LOGPROB")]
    threshold: Option<f32>,

    /// Keep this fraction of the most surprising tokens (0 < ratio ≤ 1).
    #[arg(short, long, value_name = "RATIO")]
    keep_ratio: Option<f32>,

    // ── Output ────────────────────────────────────────────────────────────
    #[arg(short, long = "output-format", default_value = "text")]
    output_format: OutputFormat,

    /// Print full per-token JSON (overrides --output-format).
    #[arg(short, long)]
    debug: bool,

    /// Print compression stats to stderr.
    #[arg(short, long)]
    stats: bool,

    // ── Hook mode ─────────────────────────────────────────────────────────
    /// Run as a Claude Code UserPromptSubmit hook: read JSON from stdin,
    /// compress if above --hook-threshold tokens, write hook JSON to stdout.
    #[arg(long)]
    hook_mode: bool,

    /// Minimum estimated token count before compressing in hook mode.
    #[arg(long, default_value_t = 500)]
    hook_threshold: usize,

    // ── Logprob mode enhancements ───────────────────────────────────────────
    /// Use quantized log-softmax for faster (slightly less precise) scoring.
    #[arg(long)]
    fast_scoring: bool,

    /// Use two-stage filtering: coarse pass removes obvious noise, then
    /// fine pass uses context-aware scoring. Overrides --threshold / --keep-ratio.
    #[arg(long)]
    two_stage: bool,

    /// Coarse threshold for two-stage filtering (default: -0.01).
    /// Tokens with logprob above this are dropped in the first pass.
    #[arg(long, default_value_t = -0.01)]
    coarse_threshold: f32,

    /// Fine threshold for two-stage filtering (default: -1.0).
    #[arg(long, default_value_t = -1.0)]
    fine_threshold: f32,

    /// Context alpha for two-stage filtering (default: 0.3).
    /// Weight of surrounding-token surprise bonus.
    #[arg(long, default_value_t = 0.3)]
    context_alpha: f32,

    // ── Sentence mode ─────────────────────────────────────────────────────────
    /// Use sentence-level query-relevant compression instead of logprob scoring.
    /// No model required. Sentences are scored by term overlap with --query.
    #[arg(long)]
    sentence_mode: bool,

    /// Query for sentence mode: sentences with highest term overlap are kept.
    /// Pass the user's question or key terms.
    #[arg(long, value_name = "TEXT")]
    query: Option<String>,

    /// Fraction of tokens to remove in sentence mode (0 < ratio ≤ 1).
    /// 0.05 = very light, 0.1 = light (default), 0.45 = moderate.
    #[arg(long, default_value_t = 0.1)]
    target_reduction: f32,

    /// Maximum sentences to keep in sentence mode.
    #[arg(long, default_value_t = 10)]
    max_sentences: usize,

    /// Use hash-based sketch scoring for sentence mode (distributional similarity).
    #[arg(long)]
    sketch_sentences: bool,

    // ── Claude Code setup ─────────────────────────────────────────────────────
    /// Wire imptokens into Claude Code as a UserPromptSubmit hook.
    /// Writes the hook entry to ~/.claude/settings.json and prints instructions.
    #[arg(long)]
    setup_claude: bool,

    // ── Savings counter ───────────────────────────────────────────────────────
    /// Show cumulative token savings across all runs.
    #[arg(long)]
    gain: bool,

    // ── Installation check ────────────────────────────────────────────────────
    /// Verify the installation: binary, model download, and a smoke-test compression.
    #[arg(long)]
    check: bool,
}

#[derive(Clone, ValueEnum)]
enum OutputFormat {
    Text,
    TokenIds,
    Json,
}

// ─── Entry point ─────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.gain {
        return run_gain();
    }

    if cli.check {
        return run_check(&cli);
    }

    if cli.setup_claude {
        return run_setup_claude(&cli);
    }

    if cli.hook_mode {
        return run_hook(&cli);
    }

    if cli.sentence_mode {
        return run_sentence_mode(&cli);
    }

    // Validate strategy flags
    if cli.threshold.is_some() && cli.keep_ratio.is_some() {
        bail!("--threshold and --keep-ratio are mutually exclusive");
    }

    // Read input
    let text = read_input(&cli)?;
    if text.trim().is_empty() {
        bail!("input text is empty");
    }

    // Build compressor
    let mut compressor = build_compressor(&cli)?;
    let model_path = resolve_model(&cli)?;
    compressor.load(&model_path)?;

    // Compress
    let result = compressor.compress(&text)?;

    // Stats to stderr
    if cli.stats {
        let stats = imptokens::result::CompressionStats::from(&result);
        eprintln!(
            "tokens: {}/{} kept  ({:.1}% reduction, {} saved)",
            result.n_kept(),
            result.n_original(),
            (1.0 - result.compression_ratio()) * 100.0,
            result.n_original() - result.n_kept(),
        );
        let _ = stats; // already printed above
    }

    stats::record(result.n_original() as u64, result.n_kept() as u64);

    // Output
    if cli.debug {
        println!("{}", serde_json::to_string_pretty(&result.to_dict())?);
    } else {
        match cli.output_format {
            OutputFormat::Text => println!("{}", result.to_text()),
            OutputFormat::TokenIds => {
                let ids: Vec<String> = result.to_token_ids().iter().map(|i| i.to_string()).collect();
                println!("{}", ids.join(" "));
            }
            OutputFormat::Json => {
                let j = serde_json::json!({
                    "original_text": result.original_text,
                    "compressed_text": result.to_text(),
                    "n_original": result.n_original(),
                    "n_kept": result.n_kept(),
                    "compression_ratio": result.compression_ratio(),
                });
                println!("{}", serde_json::to_string_pretty(&j)?);
            }
        }
    }

    Ok(())
}

// ─── Hook mode ───────────────────────────────────────────────────────────────

/// Claude Code UserPromptSubmit hook.
///
/// Reads JSON from stdin, compresses the prompt if it exceeds
/// `--hook-threshold` estimated tokens, then writes hook output JSON to stdout.
///
/// Input:  `{"prompt": "...", "session_id": "...", ...}`
/// Output: `{}` (no-op) or `{"hookSpecificOutput": {"hookOutputText": "..."}}`
fn run_hook(cli: &Cli) -> anyhow::Result<()> {
    // Read stdin (bounded to prevent OOM from malicious/accidental huge input).
    let mut stdin_buf = String::new();
    io::stdin()
        .take(MAX_INPUT_BYTES)
        .read_to_string(&mut stdin_buf)
        .context("failed to read stdin")?;

    let input: serde_json::Value = serde_json::from_str(&stdin_buf).unwrap_or_default();
    let prompt = match input.get("prompt").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => {
            // Not a UserPromptSubmit event or no prompt field — pass through.
            println!("{{}}");
            return Ok(());
        }
    };

    // Rough token estimate: ~4 chars per token
    let estimated_tokens = prompt.len() / 4;
    if estimated_tokens < cli.hook_threshold {
        println!("{{}}");
        return Ok(());
    }

    // Compress with conservative ratio (keep 60% — light, fast).
    let strategy = Strategy::TargetRatio(cli.keep_ratio.unwrap_or(0.6));
    let backend = LlamaCppBackend::new().context("failed to create backend")?;
    let mut compressor = Compressor::new(Box::new(backend), strategy);
    let model_path = resolve_model(cli)?;
    compressor.load(&model_path)?;

    let result = compressor.compress(prompt)?;
    let compressed = result.to_text();
    let saved_pct = (1.0 - result.compression_ratio()) * 100.0;

    // Return hook output: inject compressed prompt as context annotation.
    let hook_text = format!(
        "[imptokens: prompt compressed {:.0}% ({}/{} tokens kept)]\n{}",
        saved_pct,
        result.n_kept(),
        result.n_original(),
        compressed,
    );

    stats::record(result.n_original() as u64, result.n_kept() as u64);

    let output = serde_json::json!({
        "hookSpecificOutput": {
            "hookOutputText": hook_text,
        }
    });
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

// ─── Sentence mode ───────────────────────────────────────────────────────────

fn run_sentence_mode(cli: &Cli) -> anyhow::Result<()> {
    use imptokens::sentence::{compress_context, CompressContextOptions};

    let text = read_input(cli)?;
    if text.trim().is_empty() {
        bail!("input text is empty");
    }

    if !(0.0 < cli.target_reduction && cli.target_reduction <= 1.0) {
        bail!("--target-reduction must be in (0, 1]");
    }

    let query = cli.query.as_deref().unwrap_or("");
    let opts = CompressContextOptions {
        target_reduction: cli.target_reduction,
        max_sentences: cli.max_sentences,
        use_sketch: cli.sketch_sentences,
        ..Default::default()
    };

    let result = compress_context(&text, query, opts);
    let compression_ratio = if result.estimated_original_tokens > 0 {
        result.estimated_kept_tokens as f64 / result.estimated_original_tokens as f64
    } else {
        0.0
    };

    if cli.stats {
        eprintln!(
            "sentences: {}/{} kept  tokens: {}/{} ({:.1}% reduction)",
            result.n_kept_sentences,
            result.n_original_sentences,
            result.estimated_kept_tokens,
            result.estimated_original_tokens,
            (1.0 - compression_ratio) * 100.0,
        );
    }

    stats::record(
        result.estimated_original_tokens as u64,
        result.estimated_kept_tokens as u64,
    );

    if cli.debug {
        let j = serde_json::json!({
            "original_text": text,
            "compressed_text": result.compressed,
            "n_original": result.estimated_original_tokens,
            "n_kept": result.estimated_kept_tokens,
            "compression_ratio": compression_ratio,
            "n_original_sentences": result.n_original_sentences,
            "n_kept_sentences": result.n_kept_sentences,
        });
        println!("{}", serde_json::to_string_pretty(&j)?);
    } else {
        match cli.output_format {
            OutputFormat::Text => println!("{}", result.compressed),
            OutputFormat::TokenIds => bail!("--output-format token-ids is not supported in --sentence-mode"),
            OutputFormat::Json => {
                let j = serde_json::json!({
                    "original_text": text,
                    "compressed_text": result.compressed,
                    "n_original": result.estimated_original_tokens,
                    "n_kept": result.estimated_kept_tokens,
                    "compression_ratio": compression_ratio,
                });
                println!("{}", serde_json::to_string_pretty(&j)?);
            }
        }
    }

    Ok(())
}

// ─── Savings counter ─────────────────────────────────────────────────────────

fn run_gain() -> anyhow::Result<()> {
    let s = stats::load();
    if s.total_runs == 0 {
        println!("No compressions recorded yet. Run imptokens on some input first.");
        return Ok(());
    }
    println!("Tokens saved:  {:>12} / {} total input tokens  ({:.1}% average reduction)",
        format_num(s.total_saved()),
        format_num(s.total_original_tokens),
        s.avg_reduction_pct(),
    );
    println!("Compression runs: {}", s.total_runs);
    Ok(())
}

fn format_num(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { out.push(','); }
        out.push(c);
    }
    out.chars().rev().collect()
}

// ─── Installation check ───────────────────────────────────────────────────────

fn run_check(cli: &Cli) -> anyhow::Result<()> {
    // 1. Binary is clearly working if we got here — just print version.
    let version = env!("CARGO_PKG_VERSION");
    println!("  binary    OK  (imptokens v{version})");

    // 2. Model resolution.
    let model_path = match resolve_model(cli) {
        Ok(p) => {
            println!("  model     OK  ({})", p.display());
            p
        }
        Err(e) => {
            println!("  model     FAIL  {e}");
            anyhow::bail!("installation check failed");
        }
    };

    // 3. Smoke-test compression.
    let smoke = "The quick brown fox jumps over the lazy dog. \
                 The quick brown fox jumps over a fence.";
    let strategy = Strategy::FixedThreshold(cli.threshold.unwrap_or(-1.0));
    let backend = LlamaCppBackend::new().context("failed to create backend")?;
    let mut compressor = Compressor::new(Box::new(backend), strategy);
    compressor.load(&model_path)?;
    let result = compressor.compress(smoke)?;
    println!(
        "  smoke test  OK  ({}/{} tokens kept, {:.0}% reduction)",
        result.n_kept(),
        result.n_original(),
        (1.0 - result.compression_ratio()) * 100.0,
    );

    println!("\nInstallation OK.");
    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn build_compressor(cli: &Cli) -> anyhow::Result<Compressor> {
    let strategy = if cli.two_stage {
        Strategy::TwoStage {
            coarse_threshold: cli.coarse_threshold,
            fine_threshold: cli.fine_threshold,
            context_alpha: cli.context_alpha,
        }
    } else if let Some(r) = cli.keep_ratio {
        if !(0.0 < r && r <= 1.0) {
            bail!("--keep-ratio must be in (0, 1]");
        }
        Strategy::TargetRatio(r)
    } else {
        Strategy::FixedThreshold(cli.threshold.unwrap_or(-1.0))
    };

    let mut backend = LlamaCppBackend::new().context("failed to initialise llama.cpp backend")?;
    backend.fast_scoring = cli.fast_scoring;
    Ok(Compressor::new(Box::new(backend), strategy))
}

fn resolve_model(cli: &Cli) -> anyhow::Result<PathBuf> {
    if let Some(p) = &cli.local_model {
        return Ok(p.clone());
    }
    download_model(&cli.model, &cli.model_file)
}

fn download_model(repo_id: &str, filename: &str) -> anyhow::Result<PathBuf> {
    use hf_hub::api::sync::Api;
    eprintln!("fetching model {repo_id}/{filename} (cached after first download)…");
    let api = Api::new().context("failed to create HuggingFace API client")?;
    let hf_repo = api.model(repo_id.to_string());
    let path = hf_repo.get(filename).with_context(|| format!("failed to download {filename}"))?;
    Ok(path)
}

fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    { std::env::var("USERPROFILE").ok().map(PathBuf::from) }
    #[cfg(not(windows))]
    { std::env::var("HOME").ok().map(PathBuf::from) }
}

// ─── Claude Code setup ────────────────────────────────────────────────────────

fn run_setup_claude(cli: &Cli) -> anyhow::Result<()> {
    // Use the running binary's path so the hook works regardless of PATH.
    let binary = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("imptokens"));
    let keep_ratio = cli.keep_ratio.unwrap_or(0.6);
    // Quote the path so it survives spaces (e.g. /Users/John Doe/.local/bin/imptokens).
    let command = format!(
        "\"{}\" --hook-mode --hook-threshold {} --keep-ratio {}",
        binary.display(),
        cli.hook_threshold,
        keep_ratio,
    );

    let home = home_dir().ok_or_else(|| anyhow!("cannot determine home directory"))?;
    let settings_path = home.join(".claude").join("settings.json");

    // Read existing settings, or start fresh.
    let mut settings: serde_json::Value = if settings_path.exists() {
        let raw = std::fs::read_to_string(&settings_path)
            .context("failed to read ~/.claude/settings.json")?;
        serde_json::from_str(&raw).unwrap_or(serde_json::json!({}))
    } else {
        serde_json::json!({})
    };

    // Bail early if already configured.
    let already = settings
        .pointer("/hooks/UserPromptSubmit")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter().any(|e| {
                e.pointer("/hooks/0/command")
                    .and_then(|c| c.as_str())
                    .map(|c| c.contains("imptokens"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);

    if already {
        println!("imptokens hook is already present in {}", settings_path.display());
        println!("Remove it manually and re-run to reconfigure.");
        return Ok(());
    }

    // Insert hook entry.
    let hook_entry = serde_json::json!({
        "matcher": "",
        "hooks": [{ "type": "command", "command": command }]
    });

    let hooks = settings
        .as_object_mut()
        .ok_or_else(|| anyhow!("settings.json is not a JSON object"))?
        .entry("hooks")
        .or_insert(serde_json::json!({}));

    hooks
        .as_object_mut()
        .ok_or_else(|| anyhow!("hooks is not a JSON object"))?
        .entry("UserPromptSubmit")
        .or_insert(serde_json::json!([]))
        .as_array_mut()
        .ok_or_else(|| anyhow!("UserPromptSubmit is not an array"))?
        .push(hook_entry);

    std::fs::create_dir_all(settings_path.parent().unwrap())
        .context("failed to create ~/.claude directory")?;
    std::fs::write(&settings_path, serde_json::to_string_pretty(&settings)?)
        .context("failed to write ~/.claude/settings.json")?;

    println!("✓ Claude Code hook configured");
    println!("  File:      {}", settings_path.display());
    println!("  Threshold: {} estimated tokens", cli.hook_threshold);
    println!("  Ratio:     {:.0}% of tokens kept", keep_ratio * 100.0);
    println!();
    println!("Restart Claude Code for changes to take effect.");
    Ok(())
}

fn read_input(cli: &Cli) -> anyhow::Result<String> {
    if let Some(text) = &cli.text {
        return Ok(text.clone());
    }
    if let Some(file) = &cli.file {
        if file == "-" {
            let mut buf = String::new();
            io::stdin()
                .take(MAX_INPUT_BYTES)
                .read_to_string(&mut buf)
                .context("failed to read stdin")?;
            return Ok(buf);
        }
        return std::fs::read_to_string(file)
            .with_context(|| format!("failed to read file {file}"));
    }
    // Fall back to stdin if no positional text and no --file
    let mut buf = String::new();
    io::stdin()
        .take(MAX_INPUT_BYTES)
        .read_to_string(&mut buf)
        .context("failed to read stdin")?;
    Ok(buf)
}
