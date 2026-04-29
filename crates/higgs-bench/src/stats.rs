#![allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
//! Tiny statistics helpers for bench result summaries.

/// Returns the arithmetic mean, or `0.0` if the slice is empty.
#[must_use]
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let sum: f64 = xs.iter().sum();
    sum / xs.len() as f64
}

/// Returns the median (p50). For even counts, averages the two midpoints.
#[must_use]
pub fn median(xs: &[f64]) -> f64 {
    percentile(xs, 50.0)
}

/// Returns the p95.
#[must_use]
pub fn p95(xs: &[f64]) -> f64 {
    percentile(xs, 95.0)
}

/// Returns the population standard deviation, or `0.0` if `xs.len() < 2`.
#[must_use]
pub fn stdev(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64;
    var.sqrt()
}

/// Linear-interpolated percentile in `[0, 100]`. Returns `0.0` on empty input.
#[must_use]
pub fn percentile(xs: &[f64], pct: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = pct.clamp(0.0, 100.0);
    let rank = p / 100.0 * (sorted.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let lo_v = sorted.get(lo).copied().unwrap_or(0.0);
    let hi_v = sorted.get(hi).copied().unwrap_or(lo_v);
    let frac = rank - rank.floor();
    frac.mul_add(hi_v - lo_v, lo_v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_slice_returns_zero() {
        assert!((mean(&[]) - 0.0).abs() < f64::EPSILON);
        assert!((median(&[]) - 0.0).abs() < f64::EPSILON);
        assert!((p95(&[]) - 0.0).abs() < f64::EPSILON);
        assert!((stdev(&[]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn small_sample() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&xs) - 3.0).abs() < 1e-9);
        assert!((median(&xs) - 3.0).abs() < 1e-9);
    }
}
