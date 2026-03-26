use std::path::Path;

use anyhow::Context;

use crate::backend::Backend;
use crate::result::CompressResult;
use crate::threshold;
use crate::two_stage;

/// Strategy for deciding which tokens to keep.
#[derive(Clone, Copy)]
pub enum Strategy {
    /// Keep tokens whose logprob is below `threshold`.
    FixedThreshold(f32),
    /// Keep the most surprising `keep_ratio` fraction of scored tokens.
    TargetRatio(f32),
    /// Keep exactly `n` of the most surprising scored tokens.
    TargetCount(usize),
    /// Two-stage filtering: coarse pass removes obvious noise, fine pass
    /// uses context-aware scoring.
    TwoStage {
        coarse_threshold: f32,
        fine_threshold: f32,
        context_alpha: f32,
    },
}

impl Default for Strategy {
    fn default() -> Self {
        Strategy::FixedThreshold(-1.0)
    }
}

/// High-level API: loads a backend and compresses text.
pub struct Compressor {
    backend: Box<dyn Backend>,
    strategy: Strategy,
}

impl Compressor {
    pub fn new(backend: Box<dyn Backend>, strategy: Strategy) -> Self {
        Self { backend, strategy }
    }

    /// Load the model from `model_path` (local GGUF file).
    pub fn load(&mut self, model_path: &Path) -> anyhow::Result<()> {
        self.backend.load(model_path).with_context(|| {
            format!("failed to load model from {}", model_path.display())
        })
    }

    /// Compress a single piece of text.
    pub fn compress(&self, text: &str) -> anyhow::Result<CompressResult> {
        let scored = self.backend.score(text).context("scoring failed")?;

        // Eagerly decode all token bytes so CompressResult is self-contained.
        let token_bytes: Vec<Vec<u8>> = scored
            .ids
            .iter()
            .map(|&id| self.backend.token_to_bytes(id))
            .collect::<anyhow::Result<_>>()
            .context("token decode failed")?;

        let mask = match self.strategy {
            Strategy::FixedThreshold(t) => threshold::fixed_threshold(&scored.logprobs, t),
            Strategy::TargetRatio(r) => threshold::target_ratio(&scored.logprobs, r),
            Strategy::TargetCount(n) => threshold::target_count(&scored.logprobs, n),
            Strategy::TwoStage { coarse_threshold, fine_threshold, context_alpha } => {
                two_stage::two_stage_filter(
                    &scored.logprobs,
                    coarse_threshold,
                    fine_threshold,
                    context_alpha,
                )
            }
        };

        Ok(CompressResult::new(
            text.to_string(),
            scored.ids,
            token_bytes,
            scored.logprobs,
            mask,
        ))
    }

    /// Compress multiple texts.
    pub fn compress_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<CompressResult>> {
        texts.iter().map(|t| self.compress(t)).collect()
    }
}
