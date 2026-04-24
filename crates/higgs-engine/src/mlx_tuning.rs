use std::path::Path;

/// Conservative default for chunked prefill on 27B-class models.
const DEFAULT_CHUNKED_PREFILL_THRESHOLD: i32 = 512;

/// Conservative default chunk size for long-prefill chunking.
const DEFAULT_CHUNKED_PREFILL_CHUNK_SIZE: i32 = 512;

const DEFAULT_PAGED_KV_TARGET_BYTES: usize = 512 * 1024 * 1024;
const MIN_PAGED_KV_TARGET_BYTES: usize = 256 * 1024 * 1024;
const MAX_PAGED_KV_TARGET_BYTES: usize = 2 * 1024 * 1024 * 1024;

fn parse_positive_chunked_prefill_value(raw: Option<&str>, default: i32) -> i32 {
    raw.and_then(|s| s.parse::<i32>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn parse_enabled_flag(raw: Option<&str>) -> Option<bool> {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("1" | "true" | "on" | "yes") => Some(true),
        Some("0" | "false" | "off" | "no") => Some(false),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RequestedMlxProfile {
    Baseline,
    #[default]
    Auto,
    Latency,
    Balanced,
    Throughput,
}

impl RequestedMlxProfile {
    pub fn from_env_raw(raw: Option<&str>) -> Result<Option<Self>, String> {
        match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
            None => Ok(None),
            Some("baseline" | "default" | "off") => Ok(Some(Self::Baseline)),
            Some("auto" | "mlx") => Ok(Some(Self::Auto)),
            Some("latency" | "ttft") => Ok(Some(Self::Latency)),
            Some("balanced") => Ok(Some(Self::Balanced)),
            Some("throughput" | "tps") => Ok(Some(Self::Throughput)),
            Some(other) => Err(format!(
                "invalid HIGGS_MLX_PROFILE '{other}'; expected auto, latency, balanced, throughput, or legacy aliases baseline/ttft/tps"
            )),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Auto => "auto",
            Self::Latency => "latency",
            Self::Balanced => "balanced",
            Self::Throughput => "throughput",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedMlxProfile {
    Baseline,
    Latency,
    Balanced,
    Throughput,
}

impl ResolvedMlxProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Latency => "latency",
            Self::Balanced => "balanced",
            Self::Throughput => "throughput",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelSizeClass {
    Small,
    Medium,
    Large,
    Huge,
}

#[derive(Debug, Clone, Default)]
struct ModelMetadata {
    model_type: Option<String>,
    num_hidden_layers: Option<usize>,
    hidden_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    weight_bytes: Option<u64>,
}

impl ModelMetadata {
    fn from_model_dir(model_dir: &Path) -> Self {
        let config_path = model_dir.join("config.json");
        let config = std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .unwrap_or(serde_json::Value::Null);

        Self {
            model_type: config_lookup_str(&config, "model_type").map(str::to_owned),
            num_hidden_layers: config_lookup_u64(&config, "num_hidden_layers")
                .and_then(|v| usize::try_from(v).ok()),
            hidden_size: config_lookup_u64(&config, "hidden_size")
                .and_then(|v| usize::try_from(v).ok()),
            max_position_embeddings: config_lookup_u64(&config, "max_position_embeddings")
                .and_then(|v| usize::try_from(v).ok()),
            weight_bytes: model_weight_bytes(model_dir),
        }
    }

    fn size_class(&self) -> ModelSizeClass {
        const SMALL_BYTES: u64 = 4 * 1024 * 1024 * 1024;
        const MEDIUM_BYTES: u64 = 8 * 1024 * 1024 * 1024;
        const LARGE_BYTES: u64 = 16 * 1024 * 1024 * 1024;

        self.weight_bytes.map_or_else(
            || match (
                self.num_hidden_layers.unwrap_or_default(),
                self.hidden_size.unwrap_or_default(),
            ) {
                (layers, hidden) if layers >= 80 || hidden >= 8192 => ModelSizeClass::Huge,
                (layers, hidden) if layers >= 48 || hidden >= 5120 => ModelSizeClass::Large,
                (layers, hidden) if layers >= 32 || hidden >= 3072 => ModelSizeClass::Medium,
                _ => ModelSizeClass::Small,
            },
            |weight_bytes| {
                if weight_bytes >= LARGE_BYTES {
                    ModelSizeClass::Huge
                } else if weight_bytes >= MEDIUM_BYTES {
                    ModelSizeClass::Large
                } else if weight_bytes >= SMALL_BYTES {
                    ModelSizeClass::Medium
                } else {
                    ModelSizeClass::Small
                }
            },
        )
    }

    fn is_moe(&self) -> bool {
        matches!(
            self.model_type.as_deref(),
            Some("qwen3_moe" | "qwen3_5_moe" | "deepseek_v2")
        )
    }

    fn is_long_context(&self) -> bool {
        self.max_position_embeddings.unwrap_or_default() >= 65_536
    }
}

#[derive(Debug, Clone)]
pub struct MlxRuntimeTuning {
    requested_profile: RequestedMlxProfile,
    resolved_profile: ResolvedMlxProfile,
    chunked_prefill_threshold: i32,
    chunked_prefill_chunk_size: i32,
    clear_cache_after_prefill: bool,
    enable_mtp: bool,
    paged_kv_target_bytes: usize,
}

impl MlxRuntimeTuning {
    pub fn from_model_dir(model_dir: &Path, requested_profile: RequestedMlxProfile) -> Self {
        let metadata = ModelMetadata::from_model_dir(model_dir);
        let resolved_profile = resolve_profile_from_metadata(requested_profile, &metadata);
        let mut tuning = Self::from_profile(requested_profile, resolved_profile, &metadata);

        tuning.chunked_prefill_threshold = parse_positive_chunked_prefill_value(
            std::env::var("HIGGS_CHUNKED_PREFILL_THRESHOLD")
                .ok()
                .as_deref(),
            tuning.chunked_prefill_threshold,
        );
        tuning.chunked_prefill_chunk_size = parse_positive_chunked_prefill_value(
            std::env::var("HIGGS_CHUNKED_PREFILL_CHUNK_SIZE")
                .ok()
                .as_deref(),
            tuning.chunked_prefill_chunk_size,
        );
        tuning.clear_cache_after_prefill = parse_enabled_flag(
            std::env::var("HIGGS_CLEAR_CACHE_AFTER_PREFILL")
                .ok()
                .as_deref(),
        )
        .unwrap_or(tuning.clear_cache_after_prefill);
        tuning.enable_mtp = parse_enabled_flag(std::env::var("HIGGS_MTP").ok().as_deref())
            .unwrap_or(tuning.enable_mtp);

        tuning
    }

    fn from_profile(
        requested_profile: RequestedMlxProfile,
        resolved_profile: ResolvedMlxProfile,
        metadata: &ModelMetadata,
    ) -> Self {
        let size_class = metadata.size_class();
        let is_long_context = metadata.is_long_context();
        let is_moe = metadata.is_moe();
        let (balanced_threshold, balanced_chunk) =
            balanced_chunked_prefill(size_class, is_long_context, is_moe);
        let balanced_paged_kv = heuristic_paged_kv_target_bytes(metadata, size_class, is_moe);

        match resolved_profile {
            ResolvedMlxProfile::Baseline => Self {
                requested_profile,
                resolved_profile,
                chunked_prefill_threshold: DEFAULT_CHUNKED_PREFILL_THRESHOLD,
                chunked_prefill_chunk_size: DEFAULT_CHUNKED_PREFILL_CHUNK_SIZE,
                clear_cache_after_prefill: false,
                enable_mtp: false,
                paged_kv_target_bytes: DEFAULT_PAGED_KV_TARGET_BYTES,
            },
            ResolvedMlxProfile::Latency => Self {
                requested_profile,
                resolved_profile,
                chunked_prefill_threshold: (balanced_threshold.saturating_mul(2)).min(4096),
                chunked_prefill_chunk_size: balanced_chunk.max(768),
                clear_cache_after_prefill: false,
                enable_mtp: true,
                paged_kv_target_bytes: clamp_paged_kv_target_bytes(
                    balanced_paged_kv.saturating_mul(9) / 8,
                ),
            },
            ResolvedMlxProfile::Balanced => Self {
                requested_profile,
                resolved_profile,
                chunked_prefill_threshold: balanced_threshold,
                chunked_prefill_chunk_size: balanced_chunk,
                clear_cache_after_prefill: false,
                enable_mtp: true,
                paged_kv_target_bytes: balanced_paged_kv,
            },
            ResolvedMlxProfile::Throughput => Self {
                requested_profile,
                resolved_profile,
                chunked_prefill_threshold: balanced_threshold.max(1024),
                chunked_prefill_chunk_size: balanced_chunk.max(1024),
                clear_cache_after_prefill: false,
                enable_mtp: true,
                paged_kv_target_bytes: clamp_paged_kv_target_bytes(
                    balanced_paged_kv.saturating_mul(5) / 4,
                ),
            },
        }
    }

    pub const fn requested_profile(&self) -> RequestedMlxProfile {
        self.requested_profile
    }

    pub const fn resolved_profile(&self) -> ResolvedMlxProfile {
        self.resolved_profile
    }

    pub const fn chunked_prefill_threshold(&self) -> i32 {
        self.chunked_prefill_threshold
    }

    pub const fn chunked_prefill_chunk_size(&self) -> i32 {
        self.chunked_prefill_chunk_size
    }

    pub const fn clear_cache_after_prefill(&self) -> bool {
        self.clear_cache_after_prefill
    }

    pub const fn enable_mtp(&self) -> bool {
        self.enable_mtp
    }

    pub const fn paged_kv_target_bytes(&self) -> usize {
        self.paged_kv_target_bytes
    }
}

pub fn resolve_runtime_tuning(
    model_dir: &Path,
    requested_profile: RequestedMlxProfile,
) -> MlxRuntimeTuning {
    MlxRuntimeTuning::from_model_dir(model_dir, requested_profile)
}

pub fn resolve_effective_mlx_profile(
    model_dir: &Path,
    requested_profile: RequestedMlxProfile,
) -> ResolvedMlxProfile {
    let metadata = ModelMetadata::from_model_dir(model_dir);
    resolve_profile_from_metadata(requested_profile, &metadata)
}

fn resolve_profile_from_metadata(
    requested_profile: RequestedMlxProfile,
    metadata: &ModelMetadata,
) -> ResolvedMlxProfile {
    match requested_profile {
        RequestedMlxProfile::Baseline => ResolvedMlxProfile::Baseline,
        RequestedMlxProfile::Latency => ResolvedMlxProfile::Latency,
        RequestedMlxProfile::Balanced => ResolvedMlxProfile::Balanced,
        RequestedMlxProfile::Throughput => ResolvedMlxProfile::Throughput,
        RequestedMlxProfile::Auto => match metadata.size_class() {
            ModelSizeClass::Small | ModelSizeClass::Medium => ResolvedMlxProfile::Balanced,
            ModelSizeClass::Large | ModelSizeClass::Huge => ResolvedMlxProfile::Throughput,
        },
    }
}

fn balanced_chunked_prefill(
    size_class: ModelSizeClass,
    is_long_context: bool,
    is_moe: bool,
) -> (i32, i32) {
    let (base_threshold, base_chunk) = match size_class {
        ModelSizeClass::Small => (2048, 1024),
        ModelSizeClass::Medium => (1536, 768),
        ModelSizeClass::Large => (1024, 512),
        ModelSizeClass::Huge => (768, 384),
    };

    let tuned_threshold = if is_long_context {
        base_threshold.min(1024)
    } else {
        base_threshold
    };
    let tuned_chunk = if is_moe {
        base_chunk.min(512)
    } else {
        base_chunk
    };
    (tuned_threshold, tuned_chunk)
}

fn heuristic_paged_kv_target_bytes(
    metadata: &ModelMetadata,
    size_class: ModelSizeClass,
    is_moe: bool,
) -> usize {
    let Some(max_recommended) = mlx_max_recommended_working_set_size() else {
        return DEFAULT_PAGED_KV_TARGET_BYTES;
    };

    let available = metadata
        .weight_bytes
        .and_then(|bytes| usize::try_from(bytes).ok())
        .map_or(max_recommended, |weight_bytes| {
            max_recommended.saturating_sub(weight_bytes)
        });

    if available == 0 {
        return DEFAULT_PAGED_KV_TARGET_BYTES;
    }

    let divisor = if is_moe {
        8
    } else {
        match size_class {
            ModelSizeClass::Small => 4,
            ModelSizeClass::Medium => 5,
            ModelSizeClass::Large => 6,
            ModelSizeClass::Huge => 8,
        }
    };

    clamp_paged_kv_target_bytes(available / divisor)
}

fn clamp_paged_kv_target_bytes(bytes: usize) -> usize {
    bytes.clamp(MIN_PAGED_KV_TARGET_BYTES, MAX_PAGED_KV_TARGET_BYTES)
}

fn config_lookup<'a>(config: &'a serde_json::Value, key: &str) -> Option<&'a serde_json::Value> {
    config
        .get(key)
        .filter(|v| !v.is_null())
        .or_else(|| config.get("text_config").and_then(|tc| tc.get(key)))
}

fn config_lookup_str<'a>(config: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    config_lookup(config, key).and_then(serde_json::Value::as_str)
}

fn config_lookup_u64(config: &serde_json::Value, key: &str) -> Option<u64> {
    config_lookup(config, key).and_then(serde_json::Value::as_u64)
}

fn model_weight_bytes(model_dir: &Path) -> Option<u64> {
    let mut total: u64 = 0;
    let mut stack = vec![model_dir.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).ok()?;
        for entry_result in entries {
            let dir_entry = entry_result.ok()?;
            let path = dir_entry.path();
            let file_type = dir_entry.file_type().ok()?;
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            if file_type.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
            {
                total = total.saturating_add(dir_entry.metadata().ok()?.len());
            }
        }
    }

    Some(total).filter(|sum| *sum > 0)
}

#[allow(unsafe_code)]
fn mlx_max_recommended_working_set_size() -> Option<usize> {
    unsafe {
        let mut info = mlx_sys::mlx_device_info_new();
        let mut dev = mlx_sys::mlx_device_new();
        mlx_sys::mlx_get_default_device(&raw mut dev);
        let mut max_rec = None;
        if mlx_sys::mlx_device_info_get(&raw mut info, dev) == 0 {
            let mut value: usize = 0;
            let key = c"max_recommended_working_set_size";
            if mlx_sys::mlx_device_info_get_size(&raw mut value, info, key.as_ptr()) == 0
                && value > 0
            {
                max_rec = Some(value);
            }
        }
        mlx_sys::mlx_device_info_free(info);
        mlx_sys::mlx_device_free(dev);
        max_rec
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ModelMetadata, RequestedMlxProfile, ResolvedMlxProfile, resolve_profile_from_metadata,
    };

    #[test]
    fn test_requested_profile_env_aliases_parse() {
        assert_eq!(
            RequestedMlxProfile::from_env_raw(Some("ttft")),
            Ok(Some(RequestedMlxProfile::Latency))
        );
        assert_eq!(
            RequestedMlxProfile::from_env_raw(Some("tps")),
            Ok(Some(RequestedMlxProfile::Throughput))
        );
        assert_eq!(
            RequestedMlxProfile::from_env_raw(Some("baseline")),
            Ok(Some(RequestedMlxProfile::Baseline))
        );
        assert_eq!(RequestedMlxProfile::from_env_raw(None), Ok(None));
    }

    #[test]
    fn test_requested_profile_invalid_env_fails() {
        assert!(RequestedMlxProfile::from_env_raw(Some("unknown")).is_err());
    }

    #[test]
    fn test_auto_profile_resolves_to_balanced_for_small_models() {
        let metadata = ModelMetadata {
            hidden_size: Some(2048),
            num_hidden_layers: Some(24),
            ..ModelMetadata::default()
        };
        assert_eq!(
            resolve_profile_from_metadata(RequestedMlxProfile::Auto, &metadata),
            ResolvedMlxProfile::Balanced
        );
    }

    #[test]
    fn test_auto_profile_resolves_to_throughput_for_large_models() {
        let metadata = ModelMetadata {
            weight_bytes: Some(20 * 1024 * 1024 * 1024),
            ..ModelMetadata::default()
        };
        assert_eq!(
            resolve_profile_from_metadata(RequestedMlxProfile::Auto, &metadata),
            ResolvedMlxProfile::Throughput
        );
    }
}
