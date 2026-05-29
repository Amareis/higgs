//! Image preprocessing for Qwen3-VL / Qwen3.5 vision models.
//!
//! Ports the logic from `mlx-vlm/mlx_vlm/models/qwen3_vl/processing_qwen3_vl.py`.

use image::{imageops, DynamicImage, GenericImageView};
use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;
use crate::Exception;
use std::path::Path;

/// Smart resize for Qwen3-VL images.
///
/// Returns `(resized_height, resized_width)` where both dimensions are
/// multiples of `factor`, clamped between `min_pixels` and `max_pixels`.
pub fn smart_resize_image(
    height: i32,
    width: i32,
    factor: i32,
    min_pixels: i32,
    max_pixels: i32,
) -> Result<(i32, i32), String> {
    if height < factor || width < factor {
        return Err(format!(
            "height:{height} or width:{width} must be larger than factor:{factor}"
        ));
    }
    let ratio = height.max(width) as f32 / height.min(width) as f32;
    if ratio > 200.0 {
        return Err(format!(
            "absolute aspect ratio must be smaller than 200, got {ratio}"
        ));
    }
    let mut h_bar = ((height as f32 / factor as f32).round() * factor as f32) as i32;
    let mut w_bar = ((width as f32 / factor as f32).round() * factor as f32) as i32;

    if h_bar * w_bar > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h_bar = (factor as f64)
            .max((height as f64 / beta / factor as f64).floor() * factor as f64)
            as i32;
        w_bar = (factor as f64)
            .max((width as f64 / beta / factor as f64).floor() * factor as f64)
            as i32;
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h_bar = ((height as f64 * beta / factor as f64).ceil() * factor as f64) as i32;
        w_bar = ((width as f64 * beta / factor as f64).ceil() * factor as f64) as i32;
    }
    Ok((h_bar, w_bar))
}

/// Load an image from disk and convert it to RGB8.
pub fn load_image(path: &Path) -> Result<DynamicImage, Exception> {
    image::open(path)
        .map(|img| img.to_rgb8().into())
        .map_err(|e| Exception::custom(format!("Failed to open image: {e}")))
}

/// Preprocess a single image for the Qwen3-VL vision encoder.
///
/// # Arguments
/// * `img` – loaded RGB image
/// * `patch_size` – e.g. 16
/// * `temporal_patch_size` – e.g. 2
/// * `merge_size` – spatial merge size, e.g. 2
/// * `min_pixels` – minimum pixel budget (default 56*56 = 3136)
/// * `max_pixels` – maximum pixel budget (default 14*14*4*1280 = 1003520)
///
/// # Returns
/// `ProcessedImage` where:
/// - `pixel_values` has shape `(num_patches, C * tps * ps * ps)` e.g. `(576, 1536)`
/// - `grid_thw` has shape `(1, 3)` with values `[grid_t, grid_h, grid_w]`
pub fn process_image(
    img: &DynamicImage,
    patch_size: i32,
    temporal_patch_size: i32,
    merge_size: i32,
    min_pixels: i32,
    max_pixels: i32,
) -> Result<crate::ProcessedImage, Exception> {
    let (orig_h, orig_w) = (img.height() as i32, img.width() as i32);
    let factor = patch_size * merge_size;

    let (resized_h, resized_w) = smart_resize_image(
        orig_h,
        orig_w,
        factor,
        min_pixels,
        max_pixels,
    )
    .map_err(Exception::custom)?;

    // Bicubic resize via image crate (CatmullRom ≈ PIL BICUBIC)
    let resized = img.resize_exact(
        resized_w as u32,
        resized_h as u32,
        imageops::FilterType::CatmullRom,
    );
    let rgb = resized.to_rgb8();
    let (w, h) = (rgb.width() as i32, rgb.height() as i32);
    let c = 3i32;

    // HWC -> CHW and convert to f32
    let mut data = Vec::with_capacity((c * h * w) as usize);
    for ch in 0..c {
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * c + ch) as usize;
                data.push(rgb.as_raw()[idx] as f32);
            }
        }
    }

    let mut arr = Array::from_slice(&data, &[c, h, w]);

    // Rescale [0, 255] -> [0, 1]
    arr = arr.multiply(&Array::from(1.0f32 / 255.0))?;

    // Normalize with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    let mean = Array::from_slice(&[0.5f32, 0.5f32, 0.5f32], &[3])
        .expand_dims_axes(&[1, 2])?;
    let std = Array::from_slice(&[0.5f32, 0.5f32, 0.5f32], &[3])
        .expand_dims_axes(&[1, 2])?;
    arr = arr.subtract(&mean)?;
    arr = arr.divide(&std)?;

    // Add batch and temporal dims: [C, H, W] -> [1, 1, C, H, W]
    arr = arr.expand_dims_axes(&[0, 1])?;

    // Temporal duplication: [1, 1, C, H, W] -> [1, tps, C, H, W]
    let dup = arr.clone();
    arr = mlx_rs::ops::concatenate_axis(&[&arr, &dup], 1)?;

    let grid_t = 1i32;
    let grid_h = resized_h / patch_size;
    let grid_w = resized_w / patch_size;

    // Patchify + spatial merge
    let patches = arr.reshape(&[
        1,
        grid_t,
        temporal_patch_size,
        c,
        grid_h / merge_size,
        merge_size,
        patch_size,
        grid_w / merge_size,
        merge_size,
        patch_size,
    ])?;
    let patches = patches.transpose_axes(&[0, 1, 4, 7, 5, 8, 3, 2, 6, 9])?;
    let flatten = patches.reshape(&[
        1,
        grid_t * grid_h * grid_w,
        c * temporal_patch_size * patch_size * patch_size,
    ])?;
    let pixel_values = flatten.index(0);

    let grid_thw = Array::from_slice(&[grid_t, grid_h, grid_w], &[1, 3]);

    Ok(crate::ProcessedImage {
        pixel_values,
        grid_thw: Some(grid_thw),
    })
}

/// Convenience: load + preprocess a single image file.
pub fn process_image_file(
    path: &Path,
    patch_size: i32,
    temporal_patch_size: i32,
    merge_size: i32,
    min_pixels: i32,
    max_pixels: i32,
) -> Result<crate::ProcessedImage, Exception> {
    let img = load_image(path)?;
    process_image(
        &img,
        patch_size,
        temporal_patch_size,
        merge_size,
        min_pixels,
        max_pixels,
    )
}
