//! Model manifest loader for `benchmarks/models.toml`.
//!
//! Each `[[models]]` entry describes one model that the benches can target.
//! Adding a new model is one TOML entry; benches can filter by tag.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// One model entry from `benchmarks/models.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// Unique key passed to `--model` on bench CLIs.
    pub key: String,
    /// Human-readable display name (used in summary tables).
    pub label: String,
    /// Model path: `HuggingFace` repo ID or absolute local path.
    pub path: String,
    /// Quantization format, e.g. `4bit`, `3bit`.
    pub quantization: String,
    /// Approximate model size in gigabytes.
    pub approx_size_gb: f64,
    /// Maximum context length in tokens.
    pub context: u32,
    /// Optional tags for filtering, e.g. `small`, `dense`, `moe`, `h2h`.
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Top-level structure of `benchmarks/models.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// All model entries declared in the manifest.
    #[serde(default)]
    pub models: Vec<Model>,
}

impl Manifest {
    #[must_use]
    pub fn find_by_key(&self, key: &str) -> Option<&Model> {
        self.models.iter().find(|m| m.key == key)
    }

    #[must_use]
    pub fn find_by_tag(&self, tag: &str) -> Vec<&Model> {
        self.models
            .iter()
            .filter(|m| m.tags.iter().any(|t| t == tag))
            .collect()
    }
}

pub fn load_manifest(path: &Path) -> Result<Manifest> {
    let body = fs::read_to_string(path)
        .with_context(|| format!("read model manifest at {}", path.display()))?;
    let manifest: Manifest = toml::from_str(&body).context("parse model manifest TOML")?;
    Ok(manifest)
}

/// Convenience for binaries: load the manifest at `path` and look up `key`.
pub fn find_by_key(path: &Path, key: &str) -> Result<Model> {
    let manifest = load_manifest(path)?;
    manifest
        .find_by_key(key)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("model key '{key}' not found in {}", path.display()))
}

/// Convenience for binaries: load the manifest at `path` and filter by `tag`.
pub fn find_by_tag(path: &Path, tag: &str) -> Result<Vec<Model>> {
    let manifest = load_manifest(path)?;
    Ok(manifest.find_by_tag(tag).into_iter().cloned().collect())
}
