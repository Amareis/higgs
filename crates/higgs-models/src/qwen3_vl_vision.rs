//! Qwen3 VL Vision Encoder
//!
//! Ported from Python mlx-vlm's `qwen3_vl/vision.py`.
//! Processes pre-patchified pixel values through Conv3d patch embedding,
//! learned positional embeddings with bilinear interpolation, 2D RoPE,
//! transformer blocks, and a patch merger.

use std::collections::HashMap;

use mlx_rs::ops::PadMode;
use mlx_rs::{
    Array, Dtype, arange,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn, ops,
    ops::indexing::IndexOp,
    transforms::eval,
};
use serde::Deserialize;

use crate::error::ModelError;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Vision encoder configuration from `config.json` -> `vision_config`.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3VisionConfig {
    pub depth: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub out_hidden_size: i32,
    pub num_heads: i32,
    pub patch_size: i32,
    pub spatial_merge_size: i32,
    pub temporal_patch_size: i32,
    pub in_channels: i32,
    pub num_position_embeddings: i32,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<i32>,
}

fn default_layer_norm_eps() -> f32 {
    1e-6
}

impl Qwen3VisionConfig {
    pub const fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_heads
    }
}

// ---------------------------------------------------------------------------
// Patch Embed (Conv3d)
// ---------------------------------------------------------------------------

/// 3D convolutional patch embedding.
///
/// Input:  `[num_patches, in_channels * temporal_patch_size * patch_size * patch_size]`
/// Output: `[num_patches, hidden_size]`
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct PatchEmbed {
    #[param]
    proj: nn::Conv3d,
    in_channels: i32,
    temporal_patch_size: i32,
    patch_size: i32,
    hidden_size: i32,
}

impl PatchEmbed {
    fn new(config: &Qwen3VisionConfig) -> Result<Self, Exception> {
        let kernel_size = mlx_rs::utils::SingleOrTriple::Triple(
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        );
        let stride = mlx_rs::utils::SingleOrTriple::Triple(
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        );
        // MLX Conv3d: weight shape [out_channels, kD, kH, kW, in_channels]
        let proj = nn::Conv3dBuilder::new(config.in_channels, config.hidden_size, kernel_size)
            .stride(stride)
            .bias(true)
            .build()?;

        Ok(Self {
            proj,
            in_channels: config.in_channels,
            temporal_patch_size: config.temporal_patch_size,
            patch_size: config.patch_size,
            hidden_size: config.hidden_size,
        })
    }

    fn forward(&mut self, hidden_states: &Array) -> Result<Array, Exception> {
        // hidden_states shape: [num_patches, C * T * H * W]
        let _flat_size =
            self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size;
        let num_patches = hidden_states.shape()[0];

        // Reshape to [num_patches, C, T, H, W]
        let reshaped = hidden_states.reshape(&[
            num_patches,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ])?;

        // Move axis 1 -> 4 for MLX Conv3d (expects [N, D, H, W, C])
        let moved = reshaped.transpose_axes(&[0, 2, 3, 4, 1])?;

        // Conv3d
        let conv_out = self.proj.forward(&moved)?;

        // Flatten to [num_patches, hidden_size]
        conv_out.reshape(&[num_patches, self.hidden_size])
    }
}

// ---------------------------------------------------------------------------
// Vision Rotary Embedding (2D)
// ---------------------------------------------------------------------------

/// 2D Rotary Position Embedding for vision attention.
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct VisionRotaryEmbedding {
    dim: i32,
    theta: f32,
}

impl VisionRotaryEmbedding {
    fn new(dim: i32, theta: f32) -> Self {
        Self { dim, theta }
    }

    /// Compute rotary frequencies for a given max grid size.
    /// Returns `[max_grid_size, dim // 2]` frequency table.
    fn compute_freqs(&self, max_grid_size: i32) -> Result<Array, Exception> {
        let inv_freq = ops::power(
            &Array::from_f32(self.theta),
            &(arange!(stop = self.dim as f32, step = 2.0)?
                .negative()?
                .divide(&Array::from_f32(self.dim as f32))?),
        )?;

        let positions = arange!(stop = max_grid_size as f32)?;
        let freqs = positions
            .reshape(&[max_grid_size, 1])?
            .multiply(&inv_freq.reshape(&[1, self.dim / 2])?)?;

        Ok(freqs)
    }
}

// ---------------------------------------------------------------------------
// Merge image features into text embeddings
// ---------------------------------------------------------------------------

pub fn merge_input_ids_with_image_features(
    image_features: &Array,
    inputs_embeds: &Array,
    input_ids: &Array,
    image_token_index: i32,
) -> Result<Array, Exception> {
    let special_image_mask = input_ids.eq(&Array::from(image_token_index))?;
    let n_image_tokens = special_image_mask.sum(None)?.item::<i32>();

    let _special_image_mask_3d = special_image_mask.expand_dims(2)?; // [1, seq, 1]
    let n_image_features = image_features.shape()[0];
    if n_image_tokens != n_image_features as i32 {
        return Err(Exception::custom(format!(
            "Image features and image tokens do not match: tokens: {n_image_tokens}, features: {n_image_features}"
        )));
    }

    let final_embedding_shape = inputs_embeds.shape().to_vec();
    let batch = final_embedding_shape[0];
    let seq_len = final_embedding_shape[1];
    let hidden_size = final_embedding_shape[2];

    // Build a 1D index into image_features for each sequence position:
    // positions with image token get 0,1,2,...; text positions get 0 (clipped).
    let mask_1d = special_image_mask.reshape(&[seq_len])?;
    let cumsum = mask_1d.cumsum(0, None, None)?;
    let indices = cumsum.subtract(&Array::from(1i32))?;
    let max_idx = (image_features.shape()[0] - 1).max(0);
    let indices_clipped = ops::clip(&indices, (Array::from(0i32), Array::from(max_idx)))?;

    // Gather features: [seq, hidden]
    let image_features_2d = image_features.reshape(&[-1, hidden_size])?;
    let gathered = mlx_rs::ops::indexing::take_axis(&image_features_2d, &indices_clipped, 0)?;
    let gathered_3d = gathered.reshape(&[batch, seq_len, hidden_size])?;

    // Broadcast mask to [batch, seq, hidden]
    let mask_2d = mask_1d.expand_dims(1)?;
    let mask_3d = mlx_rs::ops::broadcast_to(&mask_2d, &[batch, seq_len, hidden_size])?;
    mlx_rs::ops::r#where(&mask_3d, &gathered_3d, inputs_embeds)
}

// ---------------------------------------------------------------------------
// Fast Positional Embedding Interpolation
// ---------------------------------------------------------------------------

/// Bilinear interpolation of learned position embeddings.
///
/// Input: `grid_thw` shape `[batch, 3]` with `[T, H, W]` in patches.
/// Output: `[total_patches, hidden_size]` position embeddings.
fn fast_pos_embed_interpolate(
    pos_embed: &nn::Embedding,
    grid_thw: &Array,
    spatial_merge_size: i32,
    num_grid_per_side: i32,
) -> Result<Array, Exception> {
    let grid_slice = grid_thw.as_slice::<i32>();
    let grid_list: Vec<Vec<i32>> = grid_slice.chunks(3).map(|c| c.to_vec()).collect();

    let mut idx_list: Vec<Vec<i32>> = vec![Vec::new(); 4];
    let mut weight_list: Vec<Vec<f32>> = vec![Vec::new(); 4];

    for item in &grid_list {
        let _t = item[0];
        let h = item[1];
        let w = item[2];

        let h_idxs = ops::linspace::<_, f32>(0.0, (num_grid_per_side - 1) as f32, h)?;
        let w_idxs = ops::linspace::<_, f32>(0.0, (num_grid_per_side - 1) as f32, w)?;

        let h_floor = ops::floor(&h_idxs)?.as_dtype(Dtype::Int32)?;
        let w_floor = ops::floor(&w_idxs)?.as_dtype(Dtype::Int32)?;
        let h_ceil = ops::minimum(
            &h_floor.add(&Array::from(1i32))?,
            &Array::from(num_grid_per_side - 1),
        )?;
        let w_ceil = ops::minimum(
            &w_floor.add(&Array::from(1i32))?,
            &Array::from(num_grid_per_side - 1),
        )?;

        let dh = h_idxs.subtract(&h_floor.as_dtype(h_idxs.dtype())?)?;
        let dw = w_idxs.subtract(&w_floor.as_dtype(w_idxs.dtype())?)?;

        let base_h = h_floor.multiply(&Array::from(num_grid_per_side as f32))?;
        let base_h_ceil = h_ceil.multiply(&Array::from(num_grid_per_side as f32))?;

        let one = Array::from_f32(1.0);
        let one_m_dh = one.subtract(&dh)?;
        let one_m_dw = one.subtract(&dw)?;

        let idx0 = base_h
            .reshape(&[h, 1])?
            .add(&w_floor.as_dtype(Dtype::Float32)?.reshape(&[1, w])?)?
            .reshape(&[-1])?
            .as_dtype(Dtype::Int32)?;
        let idx1 = base_h
            .reshape(&[h, 1])?
            .add(&w_ceil.as_dtype(Dtype::Float32)?.reshape(&[1, w])?)?
            .reshape(&[-1])?
            .as_dtype(Dtype::Int32)?;
        let idx2 = base_h_ceil
            .reshape(&[h, 1])?
            .add(&w_floor.as_dtype(Dtype::Float32)?.reshape(&[1, w])?)?
            .reshape(&[-1])?
            .as_dtype(Dtype::Int32)?;
        let idx3 = base_h_ceil
            .reshape(&[h, 1])?
            .add(&w_ceil.as_dtype(Dtype::Float32)?.reshape(&[1, w])?)?
            .reshape(&[-1])?
            .as_dtype(Dtype::Int32)?;

        let w0 = one_m_dh
            .reshape(&[h, 1])?
            .multiply(&one_m_dw.reshape(&[1, w])?)?
            .reshape(&[-1])?;
        let w1 = one_m_dh
            .reshape(&[h, 1])?
            .multiply(&dw.reshape(&[1, w])?)?
            .reshape(&[-1])?;
        let w2 = dh
            .reshape(&[h, 1])?
            .multiply(&one_m_dw.reshape(&[1, w])?)?
            .reshape(&[-1])?;
        let w3 = dh
            .reshape(&[h, 1])?
            .multiply(&dw.reshape(&[1, w])?)?
            .reshape(&[-1])?;

        idx_list[0].extend(idx0.as_slice::<i32>().to_vec());
        idx_list[1].extend(idx1.as_slice::<i32>().to_vec());
        idx_list[2].extend(idx2.as_slice::<i32>().to_vec());
        idx_list[3].extend(idx3.as_slice::<i32>().to_vec());
        weight_list[0].extend(w0.as_slice::<f32>().to_vec());
        weight_list[1].extend(w1.as_slice::<f32>().to_vec());
        weight_list[2].extend(w2.as_slice::<f32>().to_vec());
        weight_list[3].extend(w3.as_slice::<f32>().to_vec());
    }

    let total_positions = idx_list[0].len() as i32;
    let idx_flat: Vec<i32> = idx_list.into_iter().flatten().collect();
    let weight_flat: Vec<f32> = weight_list.into_iter().flatten().collect();

    let idx_tensor = Array::from_slice(&idx_flat, &[4, total_positions]);
    let weight_tensor = Array::from_slice(&weight_flat, &[4, total_positions]);

    let pos_embeds = pos_embed
        .clone()
        .forward(&idx_tensor)?
        .multiply(&weight_tensor.expand_dims(2)?)?;

    let pos0 = pos_embeds.index((0,));
    let pos1 = pos_embeds.index((1,));
    let pos2 = pos_embeds.index((2,));
    let pos3 = pos_embeds.index((3,));
    let patch_pos_embeds = pos0.add(&pos1)?.add(&pos2)?.add(&pos3)?;

    // Split by grid_thw entries
    let mut splits = Vec::new();
    let mut start = 0;
    for (i, item) in grid_list.iter().enumerate() {
        let size = item[1] * item[2];
        let end = start + size;
        if i < grid_list.len() - 1 {
            splits.push(patch_pos_embeds.index((start..end,)));
        } else {
            splits.push(patch_pos_embeds.index((start..,)));
        }
        start = end;
    }

    let mut patch_pos_embeds_permute = Vec::new();
    for (pos_embed_arr, item) in splits.iter().zip(grid_list.iter()) {
        let t = item[0];
        let h = item[1];
        let w = item[2];
        let feature_dim = pos_embed_arr.shape()[pos_embed_arr.shape().len() - 1];

        let tiled = if t > 1 {
            ops::tile(pos_embed_arr, &[t, 1])?
        } else {
            pos_embed_arr.clone()
        };
        let reshaped = tiled.reshape(&[t, h, w, feature_dim])?;
        let permuted = reshaped
            .reshape(&[
                t,
                h / spatial_merge_size,
                spatial_merge_size,
                w / spatial_merge_size,
                spatial_merge_size,
                feature_dim,
            ])?
            .transpose_axes(&[0, 1, 3, 2, 4, 5])?
            .reshape(&[-1, feature_dim])?;
        patch_pos_embeds_permute.push(permuted);
    }

    ops::concatenate_axis(&patch_pos_embeds_permute.iter().collect::<Vec<_>>(), 0)
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
struct VisionAttention {
    #[param]
    qkv: nn::Linear,
    #[param]
    proj: nn::Linear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl VisionAttention {
    fn new(config: &Qwen3VisionConfig) -> Result<Self, Exception> {
        let head_dim = config.head_dim();
        Ok(Self {
            qkv: nn::LinearBuilder::new(config.hidden_size, config.hidden_size * 3)
                .bias(true)
                .build()?,
            proj: nn::LinearBuilder::new(config.hidden_size, config.hidden_size)
                .bias(true)
                .build()?,
            num_heads: config.num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        x: &Array,
        _cu_seqlens: &Array,
        rotary_pos_emb: &Array,
    ) -> Result<Array, Exception> {
        let seq_len = x.shape()[0];
        let qkv = self
            .qkv
            .forward(x)?
            .reshape(&[seq_len, 3, self.num_heads, self.head_dim])?;

        // Split QKV
        let q = qkv.index((.., 0));
        let k = qkv.index((.., 1));
        let v = qkv.index((.., 2));
        // Apply RoPE (add batch dim as in Python)
        let q_rot_raw = apply_rotary_pos_emb_vision(&q.expand_dims(0)?, rotary_pos_emb)?;
        let k_rot_raw = apply_rotary_pos_emb_vision(&k.expand_dims(0)?, rotary_pos_emb)?;

        let q_rot = q_rot_raw.index(0);
        let k_rot = k_rot_raw.index(0);

        // SDPA
        // For batch=1, full sequence attention
        let q_t = q_rot.transpose_axes(&[1, 0, 2])?; // [heads, seq, head_dim]
        let k_t = k_rot.transpose_axes(&[1, 0, 2])?;
        let v_t = v.transpose_axes(&[1, 0, 2])?;
        let k_t_T = k_t.transpose_axes(&[0, 2, 1])?;
        let scores = ops::matmul(&q_t, &k_t_T)?;
        let scale_arr = Array::from_f32(self.scale).as_dtype(scores.dtype())?;
        let scores = scores.multiply(&scale_arr)?;
        let weights = ops::softmax_axis(&scores, -1, None)?;
        let out = ops::matmul(&weights, &v_t)?;
        let out = out.transpose_axes(&[1, 0, 2])?;
        let out = out.reshape(&[seq_len, self.num_heads * self.head_dim])?;

        self.proj.forward(&out)
    }
}

fn apply_rotary_pos_emb_vision(tensor: &Array, freqs: &Array) -> Result<Array, Exception> {
    let orig_dtype = tensor.dtype();

    let cos = &ops::tile(&ops::cos(freqs)?.expand_dims(1)?, &[1, 1, 2])?.expand_dims(0)?;
    let sin = ops::expand_dims(
        &ops::tile(&ops::sin(freqs)?.expand_dims(1)?, &[1, 1, 2])?,
        0,
    )?;

    let output = tensor
        .multiply(&cos)?
        .add(&rotate_half(tensor)?.multiply(&sin)?)?;
    output.as_dtype(orig_dtype)
}

fn rotate_half(x: &Array) -> Result<Array, Exception> {
    let half = x.shape()[x.shape().len() - 1] / 2;
    // Tensor is 4D: [batch, seq, heads, head_dim]
    let x1 = x.index((.., .., .., ..half));
    let x2 = x.index((.., .., .., half..));
    ops::concatenate_axis(&[&x2.negative()?, &x1], -1)
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
struct VisionMlp {
    #[param]
    linear_fc1: nn::Linear,
    #[param]
    linear_fc2: nn::Linear,
}

impl VisionMlp {
    fn new(config: &Qwen3VisionConfig) -> Result<Self, Exception> {
        Ok(Self {
            linear_fc1: nn::LinearBuilder::new(config.hidden_size, config.intermediate_size)
                .bias(true)
                .build()?,
            linear_fc2: nn::LinearBuilder::new(config.intermediate_size, config.hidden_size)
                .bias(true)
                .build()?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let hidden = self.linear_fc1.forward(x)?;
        let activated = nn::gelu_approximate(&hidden)?;
        self.linear_fc2.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// Vision Block
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
struct VisionBlock {
    #[param]
    norm1: nn::LayerNorm,
    #[param]
    norm2: nn::LayerNorm,
    #[param]
    attn: VisionAttention,
    #[param]
    mlp: VisionMlp,
}

impl VisionBlock {
    fn new(config: &Qwen3VisionConfig) -> Result<Self, Exception> {
        Ok(Self {
            norm1: nn::LayerNormBuilder::new(config.hidden_size)
                .eps(config.layer_norm_eps)
                .build()?,
            norm2: nn::LayerNormBuilder::new(config.hidden_size)
                .eps(config.layer_norm_eps)
                .build()?,
            attn: VisionAttention::new(config)?,
            mlp: VisionMlp::new(config)?,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        cu_seqlens: &Array,
        rotary_pos_emb: &Array,
    ) -> Result<Array, Exception> {
        let normed = self.norm1.forward(x)?;
        let attn_out = self.attn.forward(&normed, cu_seqlens, rotary_pos_emb)?;
        let h = x.add(&attn_out)?;

        let normed2 = self.norm2.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed2)?;
        h.add(&mlp_out)
    }
}

// ---------------------------------------------------------------------------
// Patch Merger
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
struct PatchMerger {
    #[param]
    norm: nn::LayerNorm,
    #[param]
    linear_fc1: nn::Linear,
    #[param]
    linear_fc2: nn::Linear,
    hidden_size: i32,
}

impl PatchMerger {
    fn new(config: &Qwen3VisionConfig) -> Result<Self, Exception> {
        let hidden_size =
            config.hidden_size * (config.spatial_merge_size * config.spatial_merge_size);
        Ok(Self {
            norm: nn::LayerNormBuilder::new(config.hidden_size)
                .eps(config.layer_norm_eps)
                .build()?,
            linear_fc1: nn::LinearBuilder::new(hidden_size, hidden_size)
                .bias(true)
                .build()?,
            linear_fc2: nn::LinearBuilder::new(hidden_size, config.out_hidden_size)
                .bias(true)
                .build()?,
            hidden_size,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let x = self.norm.forward(x)?;
        let x = x.reshape(&[-1, self.hidden_size])?;
        let x = nn::gelu_approximate(&self.linear_fc1.forward(&x)?)?;
        self.linear_fc2.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// Vision Model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct Qwen3VisionModel {
    pub config: Qwen3VisionConfig,
    #[param]
    patch_embed: PatchEmbed,
    #[param]
    pos_embed: nn::Embedding,
    #[param]
    rotary_pos_emb: VisionRotaryEmbedding,
    #[param]
    blocks: Vec<VisionBlock>,
    #[param]
    merger: PatchMerger,
    num_grid_per_side: i32,
}

impl Qwen3VisionModel {
    pub fn new(config: &Qwen3VisionConfig) -> Result<Self, Exception> {
        let num_grid_per_side = (config.num_position_embeddings as f32).sqrt() as i32;

        Ok(Self {
            patch_embed: PatchEmbed::new(config)?,
            pos_embed: nn::Embedding::new(config.num_position_embeddings, config.hidden_size)?,
            rotary_pos_emb: VisionRotaryEmbedding::new(config.head_dim() / 2, 10000.0),
            blocks: (0..config.depth)
                .map(|_| VisionBlock::new(config))
                .collect::<Result<Vec<_>, _>>()?,
            merger: PatchMerger::new(config)?,
            num_grid_per_side,
            config: config.clone(),
        })
    }

    pub fn forward(&mut self, hidden_states: &Array, grid_thw: &Array) -> Result<Array, Exception> {
        // Patch embedding
        let mut hidden_states = self.patch_embed.forward(hidden_states)?;

        // Positional embeddings
        let pos_embeds = fast_pos_embed_interpolate(
            &self.pos_embed,
            grid_thw,
            self.config.spatial_merge_size,
            self.num_grid_per_side,
        )?;
        hidden_states = hidden_states.add(&pos_embeds)?;

        // Rotary position embeddings
        let max_hw = grid_thw.index((.., 1..)).max(None)?;
        let max_hw = max_hw.item::<i32>();
        let rotary_freqs = self.rotary_pos_emb.compute_freqs(max_hw)?;

        // Compute 2D position IDs from grid_thw
        let rotary_pos_emb = self.compute_rot_pos_emb(grid_thw, &rotary_freqs)?;

        // Flatten sequence
        let seq_len = hidden_states.shape()[0];
        hidden_states = hidden_states.reshape(&[seq_len, -1])?;
        let rotary_pos_emb = rotary_pos_emb.reshape(&[seq_len, -1])?;

        // cu_seqlens for batch
        let mut cu_seqlens_vec = Vec::new();
        let batch_size = grid_thw.shape()[0];
        for i in 0..batch_size {
            let h = grid_thw.index((i, 1)).item::<i32>();
            let w = grid_thw.index((i, 2)).item::<i32>();
            let seq_len = h * w;
            let t = grid_thw.index((i, 0)).item::<i32>();
            for _ in 0..t {
                cu_seqlens_vec.push(seq_len);
            }
        }
        let cu_seqlens_arr = Array::from_slice(&cu_seqlens_vec, &[cu_seqlens_vec.len() as i32]);
        let cu_seqlens = cu_seqlens_arr.cumsum(None, None, None)?;
        let cu_seqlens = ops::pad(
            &cu_seqlens,
            &[(1, 0)],
            Array::from_f32(0.0),
            PadMode::Constant,
        )?;

        // Transformer blocks
        for block in &mut self.blocks {
            hidden_states = block.forward(&hidden_states, &cu_seqlens, &rotary_pos_emb)?;
        }

        // Merger
        self.merger.forward(&hidden_states)
    }

    fn compute_rot_pos_emb(
        &self,
        grid_thw: &Array,
        freq_table: &Array,
    ) -> Result<Array, Exception> {
        let merge_size = self.config.spatial_merge_size;
        let mut pos_ids_list = Vec::new();

        // grid_thw tolist not available — iterate via as_slice
        let grid_slice = grid_thw.as_slice::<i32>();
        let grid_list: Vec<Vec<i32>> = grid_slice.chunks(3).map(|c| c.to_vec()).collect();
        for item in grid_list {
            let t = item[0];
            let h = item[1];
            let w = item[2];
            let merged_h = h / merge_size;
            let merged_w = w / merge_size;

            // Create block and intra-block indices
            let block_rows = arange!(stop = merged_h)?.as_dtype(Dtype::Int32)?;
            let block_cols = arange!(stop = merged_w)?.as_dtype(Dtype::Int32)?;
            let intra_row = arange!(stop = merge_size)?.as_dtype(Dtype::Int32)?;
            let intra_col = arange!(stop = merge_size)?.as_dtype(Dtype::Int32)?;

            // Compute full-resolution positions
            let row_idx = block_rows
                .reshape(&[merged_h, 1, 1, 1])?
                .multiply(&Array::from_f32(merge_size as f32))?
                .add(&intra_row.reshape(&[1, 1, merge_size, 1])?)?;
            let col_idx = block_cols
                .reshape(&[1, merged_w, 1, 1])?
                .multiply(&Array::from_f32(merge_size as f32))?
                .add(&intra_col.reshape(&[1, 1, 1, merge_size])?)?;

            // Broadcast and flatten
            let row_idx =
                ops::broadcast_to(&row_idx, &[merged_h, merged_w, merge_size, merge_size])?
                    .reshape(&[-1])?;
            let col_idx =
                ops::broadcast_to(&col_idx, &[merged_h, merged_w, merge_size, merge_size])?
                    .reshape(&[-1])?;

            // Stack into coordinate pairs
            let coords = ops::stack_axis(&[&row_idx, &col_idx], -1)?;

            // Repeat for temporal dimension
            let coords = if t > 1 {
                ops::tile(&coords, &[t, 1])?
            } else {
                coords
            };

            pos_ids_list.push(coords);
        }

        let pos_ids = ops::concatenate_axis(&pos_ids_list.iter().collect::<Vec<_>>(), 0)?;

        // Lookup embeddings
        let pos_ids = pos_ids.as_dtype(Dtype::Int32)?;
        let h_embeddings = freq_table.index(pos_ids.index((.., 0)));
        let w_embeddings = freq_table.index(pos_ids.index((.., 1)));
        let emb = ops::concatenate_axis(&[&h_embeddings, &w_embeddings], -1)?;
        Ok(emb)
    }
}

// ---------------------------------------------------------------------------
// Weight Loading
// ---------------------------------------------------------------------------

/// Load Qwen3 vision model weights from safetensors.
///
/// Expects weights prefixed with `vision_tower.` in the MLX-converted model.
pub fn load_qwen3_vision_weights(
    model: &mut Qwen3VisionModel,
    weights: &HashMap<String, Array>,
) -> Result<(), ModelError> {
    let get = |name: &str| -> Result<Array, ModelError> {
        weights
            .get(name)
            .cloned()
            .ok_or_else(|| ModelError::MissingWeight(format!("Missing vision weight: {name}")))
    };

    let prefix = "vision_tower";

    // Patch embedding
    // MLX Conv3d expects [out_channels, kD, kH, kW, in_channels]
    // PyTorch weight is [out_channels, in_channels, kD, kH, kW]
    // Need transpose: (0, 2, 3, 4, 1)
    let pe_weight = get(&format!("{prefix}.patch_embed.proj.weight"))?;
    let pe_weight =
        if pe_weight.shape().len() == 5 && pe_weight.shape()[1] == model.config.in_channels {
            pe_weight.transpose_axes(&[0, 2, 3, 4, 1])?
        } else {
            pe_weight
        };
    model.patch_embed.proj.weight = Param::new(pe_weight);
    model.patch_embed.proj.bias =
        Param::new(Some(get(&format!("{prefix}.patch_embed.proj.bias"))?));

    // Position embedding
    model.pos_embed.weight = Param::new(get(&format!("{prefix}.pos_embed.weight"))?);

    // Blocks
    for (i, block) in model.blocks.iter_mut().enumerate() {
        let bp = format!("{prefix}.blocks.{i}");

        // Norm1
        block.norm1.weight = Param::new(Some(get(&format!("{bp}.norm1.weight"))?));
        block.norm1.bias = Param::new(Some(get(&format!("{bp}.norm1.bias"))?));

        // Norm2
        block.norm2.weight = Param::new(Some(get(&format!("{bp}.norm2.weight"))?));
        block.norm2.bias = Param::new(Some(get(&format!("{bp}.norm2.bias"))?));

        // Attention QKV
        block.attn.qkv.weight = Param::new(get(&format!("{bp}.attn.qkv.weight"))?);
        block.attn.qkv.bias = Param::new(Some(get(&format!("{bp}.attn.qkv.bias"))?));

        // Attention proj
        block.attn.proj.weight = Param::new(get(&format!("{bp}.attn.proj.weight"))?);
        block.attn.proj.bias = Param::new(Some(get(&format!("{bp}.attn.proj.bias"))?));

        // MLP
        let fc1_w = get(&format!("{bp}.mlp.linear_fc1.weight"))?;
        block.mlp.linear_fc1.weight = Param::new(fc1_w);
        block.mlp.linear_fc1.bias = Param::new(Some(get(&format!("{bp}.mlp.linear_fc1.bias"))?));
        block.mlp.linear_fc2.weight = Param::new(get(&format!("{bp}.mlp.linear_fc2.weight"))?);
        block.mlp.linear_fc2.bias = Param::new(Some(get(&format!("{bp}.mlp.linear_fc2.bias"))?));
    }

    // Merger
    let mp = format!("{prefix}.merger");
    model.merger.norm.weight = Param::new(Some(get(&format!("{mp}.norm.weight"))?));
    model.merger.norm.bias = Param::new(Some(get(&format!("{mp}.norm.bias"))?));
    model.merger.linear_fc1.weight = Param::new(get(&format!("{mp}.linear_fc1.weight"))?);
    model.merger.linear_fc1.bias = Param::new(Some(get(&format!("{mp}.linear_fc1.bias"))?));
    model.merger.linear_fc2.weight = Param::new(get(&format!("{mp}.linear_fc2.weight"))?);
    model.merger.linear_fc2.bias = Param::new(Some(get(&format!("{mp}.linear_fc2.bias"))?));

    // Force eval to GPU
    let mut all_params: Vec<&Array> = Vec::new();
    all_params.push(model.patch_embed.proj.weight.as_ref());
    if let Some(ref b) = model.patch_embed.proj.bias.value {
        all_params.push(b);
    }
    all_params.push(model.pos_embed.weight.as_ref());
    eval(all_params).map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;

    Ok(())
}
