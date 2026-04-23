//! Speculative prefill — sparse prefill optimization (experimental).
//!
//! Selects a subset of prompt tokens for initial prefill, trading off
//! accuracy for reduced TTFT on long sequences.
//!
//! Currently disabled pending optimized `RoPE` implementation.

use higgs_models::error::ModelError;

/// Configuration for speculative prefill.
#[derive(Debug, Clone)]
pub struct SpecPrefillConfig {
    /// Minimum prompt length to trigger speculative prefill.
    pub min_prompt_len: usize,
    /// Fraction of tokens to keep during speculative prefill.
    pub keep_rate: f32,
}

impl Default for SpecPrefillConfig {
    fn default() -> Self {
        Self {
            min_prompt_len: 2048,
            keep_rate: 0.5,
        }
    }
}

/// Engine for speculative (sparse) prefill.
pub struct SpecPrefillEngine {
    config: SpecPrefillConfig,
}

impl SpecPrefillEngine {
    /// Create a new speculative prefill engine.
    pub const fn new(config: SpecPrefillConfig) -> Result<Self, ModelError> {
        Ok(Self { config })
    }

    /// Whether speculative prefill should be used for a given prompt length.
    pub const fn should_use_spec_prefill(&self, prompt_len: usize) -> bool {
        prompt_len >= self.config.min_prompt_len
    }

    /// Get the keep rate for token selection.
    pub const fn get_keep_rate(&self, _prompt_len: usize) -> f32 {
        self.config.keep_rate
    }
}

#[cfg(test)]
mod tests {
    use super::{SpecPrefillConfig, SpecPrefillEngine};

    #[test]
    fn default_config_matches_expected_thresholds() {
        let config = SpecPrefillConfig::default();
        assert_eq!(config.min_prompt_len, 2048);
        assert_eq!(config.keep_rate, 0.5);
    }

    #[test]
    fn engine_uses_threshold_and_keep_rate_from_config() {
        let engine = SpecPrefillEngine::new(SpecPrefillConfig {
            min_prompt_len: 128,
            keep_rate: 0.25,
        })
        .unwrap();

        assert!(!engine.should_use_spec_prefill(127));
        assert!(engine.should_use_spec_prefill(128));
        assert_eq!(engine.get_keep_rate(32), 0.25);
        assert_eq!(engine.get_keep_rate(4096), 0.25);
    }
}
