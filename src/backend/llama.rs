use std::num::NonZeroU32;
use std::path::Path;

use std::convert::TryFrom;

use anyhow::{Context, anyhow};
#[allow(deprecated)]
use llama_cpp_2::model::Special;
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaModel},
    model::params::LlamaModelParams,
    token::LlamaToken,
};

use super::{Backend, ScoredTokens};

/// llama.cpp backend with optional GPU acceleration.
///
/// Which GPU backend is compiled in depends on the Cargo feature enabled at
/// build time (`metal`, `cuda`, or `vulkan`).  Without any feature the binary
/// runs on CPU only.
///
/// Uses `Box::leak` to give [`LlamaBackend`] a `'static` lifetime so both the
/// backend guard and the model can coexist in the same struct without fighting
/// the borrow checker.  For a CLI tool that runs-and-exits this is fine —
/// the OS reclaims any leaked memory on process exit.
pub struct LlamaCppBackend {
    backend: &'static LlamaBackend,
    model: Option<LlamaModel>,
    /// When true, use quantized log-softmax for faster (but slightly less
    /// precise) scoring.
    pub fast_scoring: bool,
}

impl LlamaCppBackend {
    pub fn new() -> anyhow::Result<Self> {
        let backend = LlamaBackend::init().context("failed to init llama.cpp backend")?;
        let backend: &'static LlamaBackend = Box::leak(Box::new(backend));
        Ok(Self { backend, model: None, fast_scoring: false })
    }

    fn model(&self) -> anyhow::Result<&LlamaModel> {
        self.model.as_ref().ok_or_else(|| anyhow!("backend not loaded — call load() first"))
    }

    /// Numerically-stable log-softmax over a raw logit slice.
    fn log_softmax(logits: &[f32]) -> Vec<f32> {
        let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = logits.iter().map(|&l| (l - max_v).exp()).sum();
        let log_sum_exp = max_v + sum_exp.ln();
        logits.iter().map(|&l| l - log_sum_exp).collect()
    }

    /// Quantized log-softmax: trades ~0.1% precision for 2-4x speed on the
    /// softmax computation by operating on i16 quantized logits.
    ///
    /// Steps:
    /// 1. Find max and min logits, compute scale = max - min.
    /// 2. Quantize shifted logits to i16: `q = round(32767 * (logit - max) / scale)`.
    /// 3. Compute approximate log-softmax on quantized values using integer
    ///    arithmetic for the exponential approximation.
    /// 4. Dequantize back to f32.
    fn quantized_log_softmax(logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return Vec::new();
        }

        let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_v = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let scale = (max_v - min_v).max(1e-8); // avoid division by zero

        // Quantize to i16 (shifted so max maps to 0, everything else is negative)
        let quantized: Vec<i16> = logits
            .iter()
            .map(|&l| ((32767.0 * (l - max_v) / scale) as i32).clamp(-32767, 0) as i16)
            .collect();

        // Dequantize back to f32 shifted logits for the exp sum.
        // Each quantized value represents (logit - max) approximately.
        let dequant = |q: i16| -> f32 { (q as f32 / 32767.0) * scale };

        let sum_exp: f32 = quantized.iter().map(|&q| dequant(q).exp()).sum();
        let log_sum = sum_exp.ln();

        quantized
            .iter()
            .map(|&q| dequant(q) - log_sum)
            .collect()
    }
}

impl Backend for LlamaCppBackend {
    fn load(&mut self, model_path: &Path) -> anyhow::Result<()> {
        // Offload all transformer layers to the GPU when a backend is compiled in
        // (Metal, CUDA, Vulkan).  A CPU-only build simply ignores this setting.
        let model_params = LlamaModelParams::default().with_n_gpu_layers(u32::MAX);
        let model = LlamaModel::load_from_file(self.backend, model_path, &model_params)
            .with_context(|| format!("failed to load model from {}", model_path.display()))?;
        self.model = Some(model);
        Ok(())
    }

    fn score(&self, text: &str) -> anyhow::Result<ScoredTokens> {
        let model = self.model()?;

        let tokens: Vec<LlamaToken> =
            model.str_to_token(text, AddBos::Never).context("tokenization failed")?;

        if tokens.is_empty() {
            return Ok(ScoredTokens { ids: vec![], logprobs: vec![] });
        }
        if tokens.len() == 1 {
            return Ok(ScoredTokens {
                ids: vec![tokens[0].0 as u32],
                logprobs: vec![None],
            });
        }

        let n_tokens = tokens.len();
        // Context window must fit all input tokens.
        let n_ctx = NonZeroU32::new((n_tokens as u32 + 4).max(8));
        let ctx_params = LlamaContextParams::default().with_n_ctx(n_ctx);

        let mut ctx = model
            .new_context(self.backend, ctx_params)
            .context("failed to create llama context")?;

        // Build a batch with logits enabled at every position so we can
        // extract log-probabilities for every token, not just the last one.
        let mut batch = LlamaBatch::new(n_tokens, 1);
        for (i, token) in tokens.iter().enumerate() {
            batch
                .add(token.clone(), i as i32, &[0], true /* compute logits here */)
                .context("failed to add token to batch")?;
        }

        ctx.decode(&mut batch).context("llama decode failed")?;

        // logprobs[0] = None (first token has no left context).
        // logprobs[i] = log P(token[i] | token[0..i]) for i > 0.
        let mut logprobs: Vec<Option<f32>> = Vec::with_capacity(n_tokens);
        logprobs.push(None);

        for pos in 0..n_tokens - 1 {
            // Raw logits at position `pos` → log-softmax → logprob of next token.
            let raw_logits = ctx.get_logits_ith(pos as i32);
            let log_probs = if self.fast_scoring {
                Self::quantized_log_softmax(raw_logits)
            } else {
                Self::log_softmax(raw_logits)
            };
            let raw_id = tokens[pos + 1].0;
            let next_token_id = usize::try_from(raw_id)
                .with_context(|| anyhow!("negative token id {raw_id} at position {}", pos + 1))?;
            let lp = log_probs
                .get(next_token_id)
                .copied()
                .ok_or_else(|| anyhow!("token id {} out of vocab range", next_token_id))?;
            logprobs.push(Some(lp));
        }

        let ids: Vec<u32> = tokens.iter().map(|t| t.0 as u32).collect();
        Ok(ScoredTokens { ids, logprobs })
    }

    #[allow(deprecated)]
    fn token_to_bytes(&self, token_id: u32) -> anyhow::Result<Vec<u8>> {
        let model = self.model()?;
        // `token_to_bytes` (and `Special`) are deprecated in favour of
        // `token_to_piece`, but still functional. We need the raw bytes
        // so that multi-byte UTF-8 sequences split across tokens can be
        // reassembled correctly.
        #[allow(deprecated)]
        model
            .token_to_bytes(LlamaToken(token_id as i32), Special::Tokenize)
            .with_context(|| format!("failed to decode token {token_id}"))
    }
}
