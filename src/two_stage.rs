/// Two-stage token filtering for logprob-based compression.
///
/// Stage 1 (Coarse): Removes obviously predictable tokens (logprob > coarse_threshold)
/// to quickly eliminate ~30-50% of tokens.
///
/// Stage 2 (Fine): For remaining tokens, computes a context importance score
/// based on surrounding token surprise. Tokens near other surprising tokens
/// get a bonus, capturing contextual clusters of important information.

/// Compute the context importance score for a token at position `i`.
///
/// The context score is the average logprob (surprise) in a window of +/-3
/// tokens around position `i`. More negative averages mean the token is
/// surrounded by other surprising tokens.
fn context_score(logprobs: &[Option<f32>], i: usize) -> f32 {
    let n = logprobs.len();
    let start = if i >= 3 { i - 3 } else { 0 };
    let end = (i + 4).min(n); // exclusive, so window is [start, end)

    let mut sum = 0.0_f32;
    let mut count = 0u32;
    for lp in &logprobs[start..end] {
        if let Some(v) = lp {
            sum += v;
            count += 1;
        }
    }

    if count == 0 { 0.0 } else { sum / count as f32 }
}

/// Two-stage filtering: returns a keep mask (true = keep the token).
///
/// - `coarse_threshold`: tokens with logprob above this are dropped in stage 1.
///   A value near zero (e.g. -0.01) removes only the most predictable tokens.
/// - `fine_threshold`: applied to the final score (logprob + alpha * context_score)
///   in stage 2.
/// - `context_alpha`: weight for the context score bonus (default 0.3).
///
/// The first token (logprob = None) is always kept.
pub fn two_stage_filter(
    logprobs: &[Option<f32>],
    coarse_threshold: f32,
    fine_threshold: f32,
    context_alpha: f32,
) -> Vec<bool> {
    let n = logprobs.len();
    let mut mask = vec![false; n];

    for (i, lp) in logprobs.iter().enumerate() {
        match lp {
            // First token (no context) is always kept.
            None => {
                mask[i] = true;
            }
            Some(v) => {
                // Stage 1 (Coarse): drop obviously predictable tokens.
                if *v > coarse_threshold {
                    // Predictable — drop.
                    continue;
                }

                // Stage 2 (Fine): compute context-aware final score.
                let ctx = context_score(logprobs, i);
                let final_score = v + context_alpha * ctx;

                mask[i] = final_score < fine_threshold;
            }
        }
    }

    mask
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_token_always_kept() {
        let lps = vec![None, Some(-0.5), Some(-0.001), Some(-2.0)];
        let mask = two_stage_filter(&lps, -0.01, -1.0, 0.3);
        assert!(mask[0], "first token (None) must always be kept");
    }

    #[test]
    fn coarse_stage_drops_predictable() {
        // -0.001 is above the coarse threshold of -0.01 → dropped in stage 1
        let lps = vec![None, Some(-0.001), Some(-3.0)];
        let mask = two_stage_filter(&lps, -0.01, -1.0, 0.3);
        assert!(!mask[1], "very predictable token should be dropped by coarse filter");
    }

    #[test]
    fn surprising_tokens_kept() {
        // All tokens are very surprising
        let lps = vec![None, Some(-5.0), Some(-4.0), Some(-6.0)];
        let mask = two_stage_filter(&lps, -0.01, -1.0, 0.3);
        assert!(mask[1]);
        assert!(mask[2]);
        assert!(mask[3]);
    }

    #[test]
    fn context_score_boosts_clustered_surprises() {
        // Token at index 2 has logprob -1.05 (borderline with threshold -1.0).
        // But it is surrounded by very surprising tokens, so context bonus
        // should push its final score below the fine threshold.
        let lps = vec![
            None,
            Some(-5.0),  // very surprising
            Some(-1.05),  // borderline
            Some(-5.0),  // very surprising
        ];
        let mask = two_stage_filter(&lps, -0.01, -1.0, 0.3);
        // context_score at i=2 ≈ avg(-5.0, -1.05, -5.0) = -3.68
        // final = -1.05 + 0.3 * (-3.68) = -1.05 + (-1.1) = -2.15 < -1.0 → keep
        assert!(mask[2], "token near surprising tokens should be boosted and kept");
    }

    #[test]
    fn context_score_computation() {
        let lps: Vec<Option<f32>> = vec![
            None,
            Some(-2.0),
            Some(-4.0),
            Some(-6.0),
            Some(-1.0),
        ];
        // context_score at i=2, window [0..5]:
        // scored values: -2.0, -4.0, -6.0, -1.0 (None is skipped)
        // avg = (-2 + -4 + -6 + -1) / 4 = -3.25
        let cs = context_score(&lps, 2);
        assert!((cs - (-3.25)).abs() < 0.01);
    }

    #[test]
    fn empty_logprobs() {
        let mask = two_stage_filter(&[], -0.01, -1.0, 0.3);
        assert!(mask.is_empty());
    }

    #[test]
    fn alpha_zero_disables_context() {
        // With alpha=0, two-stage degenerates to a fixed threshold on the
        // fine threshold (after coarse pre-filter).
        let lps = vec![None, Some(-0.5), Some(-2.0)];
        let mask = two_stage_filter(&lps, -0.01, -1.0, 0.0);
        assert!(!mask[1], "-0.5 > -1.0 → drop");
        assert!(mask[2], "-2.0 < -1.0 → keep");
    }
}
