/// Sentence-level, query-relevant context compression.
///
/// This is a zero-dependency compression engine that extracts only the
/// query-relevant sentences from a large context. It does not require an LLM
/// or any model — it uses term-overlap scoring.
///
/// Algorithm (mirrors token-compression.mjs):
/// 1. Split context into sentences on .!? boundaries and newlines.
/// 2. Deduplicate via normalized-sentence hashing.
/// 3. Precompute features for each unique sentence (token count, unique terms,
///    boilerplate flag).
/// 4. Score each sentence by term overlap with the query.
/// 5. Select top-scored sentences within a token budget.
/// 6. Reconstruct in original document order.
use std::collections::{HashMap, HashSet};

// ─── Stop words ──────────────────────────────────────────────────────────────

const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "must", "can", "could", "not",
    "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "our", "their", "which", "who", "what", "how", "when", "where", "as",
    "if", "so", "than", "then", "no", "up", "out", "about", "into", "also",
    "just", "more", "all", "there", "get", "use", "one", "two", "new",
];

const BOILERPLATE_PATTERNS: &[&str] = &[
    "copyright",
    "all rights reserved",
    "disclaimer",
    "terms of use",
    "privacy policy",
    "proprietary",
    "confidential",
    "trademark",
];

// ─── Tokenizer ───────────────────────────────────────────────────────────────

/// Tokenize `text` using word/punctuation boundaries.
///
/// Approximates the JS pattern `[\p{L}\p{N}]+|[^\s]`:
/// - Runs of alphanumeric (Unicode) characters form one token each.
/// - Any single non-whitespace, non-alphanumeric character is its own token.
/// - Whitespace is skipped.
pub fn tokenize(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut tokens = Vec::new();
    let bytes = text.as_bytes();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut i = 0;

    while i < chars.len() {
        let (byte_start, ch) = chars[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        if ch.is_alphanumeric() {
            // consume run of alphanumerics
            let mut j = i + 1;
            while j < chars.len() && chars[j].1.is_alphanumeric() {
                j += 1;
            }
            let byte_end = if j < chars.len() { chars[j].0 } else { text.len() };
            tokens.push(&text[byte_start..byte_end]);
            i = j;
        } else {
            // single non-whitespace character
            let byte_end = if i + 1 < chars.len() { chars[i + 1].0 } else { text.len() };
            let _ = bytes; // suppress unused warning
            tokens.push(&text[byte_start..byte_end]);
            i += 1;
        }
    }
    tokens
}

/// Estimate the number of tokens in `text`.
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    tokenize(text).len()
}

// ─── Sentence splitting ───────────────────────────────────────────────────────

/// Split `text` into sentences on `.`, `!`, `?` boundaries and newlines.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch == '\n' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        } else if ch == '.' || ch == '!' || ch == '?' {
            current.push(ch);
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        } else {
            current.push(ch);
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }
    sentences
}

// ─── Normalization / boilerplate ─────────────────────────────────────────────

fn normalize_sentence(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .flat_map(|c| c.to_lowercase())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_boilerplate(s: &str) -> bool {
    let lower = s.to_lowercase();
    BOILERPLATE_PATTERNS.iter().any(|pat| lower.contains(pat))
}

// ─── FNV-1a hash (deterministic across runs) ─────────────────────────────────

pub(crate) fn fnv1a_hash(s: &str) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut hash = FNV_OFFSET;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ─── Preprocessed context ────────────────────────────────────────────────────

/// Precomputed features for a single sentence.
pub struct SentenceFeatures {
    pub text: String,
    pub token_count: usize,
    /// Content terms (lowercase, len>2, not a stop word).
    pub unique_terms: HashSet<String>,
    pub is_boilerplate: bool,
    /// Position in the original sentence list.
    pub original_index: usize,
}

/// Context split, deduped, and feature-extracted — ready to score against any query.
pub struct PreprocessedContext {
    /// All sentences in original order (may include duplicates).
    pub sentences: Vec<String>,
    /// Unique sentences only (duplicates removed via normalized hash).
    pub unique_features: Vec<SentenceFeatures>,
    pub estimated_total_tokens: usize,
}

/// Preprocess a context string: split → dedup → extract features.
///
/// This step is independent of the query and can be memoized when the same
/// context is used across multiple requests (as `run_benchmark` does).
pub fn preprocess_context(context: &str) -> PreprocessedContext {
    let sentences = split_sentences(context);
    let estimated_total_tokens = estimate_tokens(context);

    let all_features: Vec<SentenceFeatures> = sentences
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let token_count = estimate_tokens(s);
            let unique_terms: HashSet<String> = tokenize(s)
                .into_iter()
                .map(|t| t.to_lowercase())
                .filter(|t| t.len() > 2 && !STOP_WORDS.contains(&t.as_str()))
                .collect();
            SentenceFeatures {
                text: s.clone(),
                token_count,
                unique_terms,
                is_boilerplate: is_boilerplate(s),
                original_index: i,
            }
        })
        .collect();

    // Deduplicate by normalized-sentence hash, preserving first occurrence.
    let mut seen: HashSet<u64> = HashSet::new();
    let unique_features: Vec<SentenceFeatures> = all_features
        .into_iter()
        .filter(|f| {
            let h = fnv1a_hash(&normalize_sentence(&f.text));
            seen.insert(h)
        })
        .collect();

    PreprocessedContext { sentences, unique_features, estimated_total_tokens }
}

// ─── Scoring ─────────────────────────────────────────────────────────────────

fn extract_query_terms(query: &str) -> HashSet<String> {
    tokenize(query)
        .into_iter()
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() > 2 && !STOP_WORDS.contains(&t.as_str()))
        .collect()
}

/// Score a sentence by query-term overlap.
///
/// Formula (mirrors token-compression.mjs):
/// ```text
/// score = (overlap × 2) + (overlap / queryTermCount) + min(tokenCount, 40) / 40
///         − boilerplatePenalty
/// ```
pub fn score_sentence(
    feat: &SentenceFeatures,
    query_terms: &HashSet<String>,
    query_term_count: usize,
) -> f64 {
    let overlap = feat.unique_terms.intersection(query_terms).count() as f64;
    let boilerplate_penalty = if feat.is_boilerplate { 0.5 } else { 0.0 };
    let length_score = (feat.token_count as f64).min(40.0) / 40.0;
    let overlap_ratio =
        if query_term_count > 0 { overlap / query_term_count as f64 } else { 0.0 };
    (overlap * 2.0) + overlap_ratio + length_score - boilerplate_penalty
}

/// Hash-based sketch scoring inspired by QJL random projections.
///
/// Maps each term to a fixed-size vector (64 dimensions) using FNV-1a hashing,
/// then computes cosine similarity between the sentence sketch and query sketch.
/// The final score blends term-overlap (70%) with sketch similarity (30%).
const SKETCH_DIM: usize = 64;

fn build_sketch(terms: &HashSet<String>) -> [f64; SKETCH_DIM] {
    let mut sketch = [0.0_f64; SKETCH_DIM];
    for term in terms {
        let bucket = (fnv1a_hash(term) as usize) % SKETCH_DIM;
        sketch[bucket] += 1.0;
    }
    sketch
}

fn cosine_similarity(a: &[f64; SKETCH_DIM], b: &[f64; SKETCH_DIM]) -> f64 {
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for i in 0..SKETCH_DIM {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

/// Score a sentence using hash-based sketch similarity combined with term overlap.
///
/// Returns `overlap_score * 0.7 + sketch_similarity * 0.3`, capturing both
/// exact term matches and distributional similarity beyond exact overlap.
pub fn score_sentence_sketch(
    feat: &SentenceFeatures,
    query_terms: &HashSet<String>,
    query_term_count: usize,
) -> f64 {
    let overlap_score = score_sentence(feat, query_terms, query_term_count);

    let sentence_sketch = build_sketch(&feat.unique_terms);
    let query_sketch = build_sketch(query_terms);
    let sketch_sim = cosine_similarity(&sentence_sketch, &query_sketch);

    overlap_score * 0.7 + sketch_sim * 0.3
}

// ─── Options & result ────────────────────────────────────────────────────────

/// Options for `compress_context` / `compress_preprocessed`.
pub struct CompressContextOptions {
    /// Fraction of tokens to remove (default: 0.45).
    pub target_reduction: f32,
    /// Minimum number of tokens in output (default: 80).
    pub min_tokens: usize,
    /// Maximum number of sentences to keep (default: 10).
    pub max_sentences: usize,
    /// Use hash-based sketch scoring for distributional similarity (default: false).
    pub use_sketch: bool,
}

impl Default for CompressContextOptions {
    fn default() -> Self {
        Self { target_reduction: 0.45, min_tokens: 80, max_sentences: 10, use_sketch: false }
    }
}

/// Output of `compress_context`.
pub struct CompressContextResult {
    pub compressed: String,
    pub n_original_sentences: usize,
    pub n_kept_sentences: usize,
    pub estimated_original_tokens: usize,
    pub estimated_kept_tokens: usize,
}

// ─── Compression ─────────────────────────────────────────────────────────────

/// Score and select sentences from a pre-processed context.
///
/// Separating preprocessing from scoring lets callers memoize
/// `preprocess_context` when the same context is queried multiple times.
pub fn compress_preprocessed(
    pre: &PreprocessedContext,
    query: &str,
    opts: &CompressContextOptions,
) -> CompressContextResult {
    let n_original_sentences = pre.sentences.len();
    let estimated_original_tokens = pre.estimated_total_tokens;

    if pre.unique_features.is_empty() {
        return CompressContextResult {
            compressed: String::new(),
            n_original_sentences,
            n_kept_sentences: 0,
            estimated_original_tokens,
            estimated_kept_tokens: 0,
        };
    }

    let query_terms = extract_query_terms(query);
    let query_term_count = query_terms.len();

    // Score unique sentences, highest first.
    let scorer: fn(&SentenceFeatures, &HashSet<String>, usize) -> f64 = if opts.use_sketch {
        score_sentence_sketch
    } else {
        score_sentence
    };
    let mut scored: Vec<(f64, usize)> = pre
        .unique_features
        .iter()
        .map(|f| (scorer(f, &query_terms, query_term_count), f.original_index))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Token budget.
    let token_budget = ((estimated_original_tokens as f32 * (1.0 - opts.target_reduction))
        .round() as usize)
        .max(opts.min_tokens);

    // Select top-scored sentences within budget.
    let mut kept_indices: Vec<usize> = Vec::new();
    let mut tokens_used: usize = 0;

    // Build a map from original_index → token_count for fast lookup.
    let tc_by_idx: HashMap<usize, usize> =
        pre.unique_features.iter().map(|f| (f.original_index, f.token_count)).collect();

    for (_, idx) in &scored {
        if kept_indices.len() >= opts.max_sentences {
            break;
        }
        let tc = tc_by_idx.get(idx).copied().unwrap_or(0);
        if tokens_used + tc <= token_budget {
            kept_indices.push(*idx);
            tokens_used += tc;
        }
    }

    // Reconstruct in original document order.
    kept_indices.sort_unstable();
    let compressed: String = kept_indices
        .iter()
        .map(|&i| pre.sentences[i].as_str())
        .collect::<Vec<_>>()
        .join(" ");

    let estimated_kept_tokens = estimate_tokens(&compressed);

    CompressContextResult {
        compressed,
        n_original_sentences,
        n_kept_sentences: kept_indices.len(),
        estimated_original_tokens,
        estimated_kept_tokens,
    }
}

/// Convenience wrapper: preprocess then compress in one call.
pub fn compress_context(
    context: &str,
    query: &str,
    opts: CompressContextOptions,
) -> CompressContextResult {
    let pre = preprocess_context(context);
    compress_preprocessed(&pre, query, &opts)
}

// ─── Prompt builder ──────────────────────────────────────────────────────────

/// Build a prompt with system prompt, context, and query.
pub fn build_prompt(context: &str, query: &str, system_prompt: Option<&str>) -> String {
    let sys =
        system_prompt.unwrap_or("You are a helpful assistant. Answer based on the context.");
    format!("{}\n\nContext:\n{}\n\nQuestion: {}", sys, context, query)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // tokenize ----------------------------------------------------------------

    #[test]
    fn tokenize_words() {
        let t = tokenize("hello world");
        assert_eq!(t, vec!["hello", "world"]);
    }

    #[test]
    fn tokenize_punctuation() {
        let t = tokenize("hello, world!");
        assert_eq!(t, vec!["hello", ",", "world", "!"]);
    }

    #[test]
    fn tokenize_unicode() {
        let t = tokenize("café résumé");
        assert_eq!(t, vec!["café", "résumé"]);
    }

    #[test]
    fn tokenize_numbers() {
        let t = tokenize("42 tokens");
        assert_eq!(t, vec!["42", "tokens"]);
    }

    #[test]
    fn tokenize_whitespace_only() {
        assert!(tokenize("   \t\n").is_empty());
    }

    #[test]
    fn tokenize_empty() {
        assert!(tokenize("").is_empty());
    }

    // estimate_tokens ---------------------------------------------------------

    #[test]
    fn estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn estimate_tokens_simple() {
        assert_eq!(estimate_tokens("hello world"), 2);
    }

    #[test]
    fn estimate_tokens_punctuation() {
        // "hello, world!" → ["hello", ",", "world", "!"] = 4
        assert_eq!(estimate_tokens("hello, world!"), 4);
    }

    // split_sentences ---------------------------------------------------------

    #[test]
    fn split_sentences_dot() {
        let s = split_sentences("Hello. World.");
        assert_eq!(s, vec!["Hello.", "World."]);
    }

    #[test]
    fn split_sentences_newline() {
        let s = split_sentences("Line one\nLine two");
        assert_eq!(s, vec!["Line one", "Line two"]);
    }

    #[test]
    fn split_sentences_mixed() {
        let s = split_sentences("First! Second?\nThird.");
        assert_eq!(s, vec!["First!", "Second?", "Third."]);
    }

    // compress_context --------------------------------------------------------

    #[test]
    fn compress_context_deduplication() {
        let ctx = "The sky is blue. The sky is blue. The grass is green.";
        let result = compress_context(ctx, "sky", CompressContextOptions::default());
        // Duplicate sentence should only appear once
        let count = result.compressed.matches("The sky is blue").count();
        assert_eq!(count, 1);
    }

    #[test]
    fn compress_context_relevance_preserved() {
        // Plant a needle; fill with unrelated filler
        let needle = "The secret code is XYZZY.";
        let filler = "Water boils at 100 degrees. The sun rises in the east. \
                      Plants need sunlight. Rivers flow downhill. Birds can fly.";
        let ctx = format!("{} {}", filler, needle);
        let result = compress_context(
            &ctx,
            "secret code XYZZY",
            CompressContextOptions { max_sentences: 3, ..Default::default() },
        );
        assert!(result.compressed.contains("XYZZY"), "needle must be preserved");
    }

    #[test]
    fn compress_context_empty() {
        let result = compress_context("", "query", CompressContextOptions::default());
        assert!(result.compressed.is_empty());
        assert_eq!(result.n_original_sentences, 0);
    }

    #[test]
    fn compress_context_single_sentence() {
        let result =
            compress_context("Only one sentence.", "one", CompressContextOptions::default());
        assert_eq!(result.n_original_sentences, 1);
    }

    #[test]
    fn compress_context_zero_overlap() {
        // Query has no terms in common with context — result may still contain
        // sentences selected by length score alone.
        let result = compress_context(
            "The quick brown fox jumps.",
            "zzz qqq",
            CompressContextOptions { max_sentences: 1, ..Default::default() },
        );
        // Should not panic and should return some output (length score > 0)
        assert!(result.n_kept_sentences <= 1);
    }

    #[test]
    fn compress_context_ratio_math() {
        // Compression ratio must be ≤ 1
        let ctx = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. \
                   Kappa lambda mu. Nu xi omicron.";
        let result = compress_context(ctx, "alpha delta", CompressContextOptions::default());
        assert!(result.estimated_kept_tokens <= result.estimated_original_tokens);
    }

    #[test]
    fn compress_context_sentence_cap() {
        let ctx = "A. B. C. D. E. F. G. H. I. J. K. L.";
        let result = compress_context(
            ctx,
            "all",
            CompressContextOptions { max_sentences: 3, ..Default::default() },
        );
        assert!(result.n_kept_sentences <= 3);
    }

    #[test]
    fn compress_context_boilerplate_penalty() {
        let ctx = "Copyright 2024 Acme Corp. All rights reserved. \
                   The answer is 42. \
                   This is proprietary information.";
        let result = compress_context(
            ctx,
            "answer",
            CompressContextOptions { max_sentences: 2, ..Default::default() },
        );
        // "The answer is 42" should be preferred over boilerplate sentences
        assert!(result.compressed.contains("42"));
    }

    // build_prompt ------------------------------------------------------------

    #[test]
    fn build_prompt_contains_context_and_query() {
        let p = build_prompt("some context", "my query", None);
        assert!(p.contains("some context"));
        assert!(p.contains("my query"));
    }

    #[test]
    fn build_prompt_custom_system() {
        let p = build_prompt("ctx", "q", Some("Custom system."));
        assert!(p.starts_with("Custom system."));
    }

    // sketch scoring ----------------------------------------------------------

    #[test]
    fn sketch_score_non_negative() {
        let feat = SentenceFeatures {
            text: "Rust is a systems programming language.".to_string(),
            token_count: 6,
            unique_terms: ["rust", "systems", "programming", "language"]
                .iter().map(|s| s.to_string()).collect(),
            is_boilerplate: false,
            original_index: 0,
        };
        let query: HashSet<String> = ["rust", "language"].iter().map(|s| s.to_string()).collect();
        let score = score_sentence_sketch(&feat, &query, 2);
        assert!(score >= 0.0, "sketch score must be non-negative");
    }

    #[test]
    fn sketch_score_identical_terms() {
        let terms: HashSet<String> = ["alpha", "beta", "gamma"]
            .iter().map(|s| s.to_string()).collect();
        let feat = SentenceFeatures {
            text: "alpha beta gamma".to_string(),
            token_count: 3,
            unique_terms: terms.clone(),
            is_boilerplate: false,
            original_index: 0,
        };
        let score = score_sentence_sketch(&feat, &terms, 3);
        // With identical terms, sketch cosine similarity should be 1.0
        // overlap_score = (3*2) + (3/3) + min(3,40)/40 = 6 + 1 + 0.075 = 7.075
        // final = 7.075 * 0.7 + 1.0 * 0.3 = 4.9525 + 0.3 = 5.2525
        assert!(score > 5.0, "identical terms should yield high score, got {score}");
    }

    #[test]
    fn sketch_score_no_overlap() {
        let feat = SentenceFeatures {
            text: "cats dogs".to_string(),
            token_count: 2,
            unique_terms: ["cats", "dogs"].iter().map(|s| s.to_string()).collect(),
            is_boilerplate: false,
            original_index: 0,
        };
        let query: HashSet<String> = ["rust", "language"].iter().map(|s| s.to_string()).collect();
        let score = score_sentence_sketch(&feat, &query, 2);
        // No term overlap; sketch similarity may be 0 if hash buckets don't collide.
        // The only contribution is the length score component.
        assert!(score < 1.0, "no overlap should yield low score, got {score}");
    }

    #[test]
    fn sketch_mode_compresses_without_panic() {
        let ctx = "Rust is fast. Python is slow. Java is verbose. Go is simple.";
        let opts = CompressContextOptions {
            use_sketch: true,
            max_sentences: 2,
            ..Default::default()
        };
        let result = compress_context(ctx, "fast language", opts);
        assert!(result.n_kept_sentences <= 2);
        assert!(!result.compressed.is_empty());
    }

    #[test]
    fn cosine_similarity_unit() {
        let a = build_sketch(&["hello".to_string()].into_iter().collect());
        let b = build_sketch(&["hello".to_string()].into_iter().collect());
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-10, "same vector should have cosine 1.0");
    }
}
