#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::shadow_unrelated,
    clippy::shadow_reuse,
    clippy::shadow_same
)]
//! `bench_decode` drives a running higgs server over HTTP and reports
//! per-trial decode tok/s and TTFT (time to first token).

use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use futures::StreamExt;
use higgs_bench::{
    BenchOutput, ModelInfo, OutputFormat, RunMetadata, default_manifest_path, format_json,
    format_markdown, models, persist_result, server, stats,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(
    name = "bench_decode",
    about = "Measure decode tok/s and TTFT against a running higgs server",
    version
)]
struct Args {
    /// Port the higgs server is listening on.
    #[arg(long, default_value_t = 8899)]
    port: u16,

    /// Host the higgs server is listening on.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Model key from `benchmarks/models.toml`.
    #[arg(long)]
    model: String,

    /// Override the manifest path.
    #[arg(long)]
    manifest: Option<std::path::PathBuf>,

    /// Maximum tokens to generate per trial.
    #[arg(long, default_value_t = 200)]
    max_tokens: u32,

    /// Number of warmup trials (not measured).
    #[arg(long, default_value_t = 1)]
    warmup: u32,

    /// Number of measured trials.
    #[arg(long, default_value_t = 5)]
    trials: u32,

    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    #[arg(long)]
    top_k: Option<u32>,

    #[arg(long)]
    top_p: Option<f32>,

    #[arg(long)]
    repetition_penalty: Option<f32>,

    #[arg(long)]
    frequency_penalty: Option<f32>,

    /// Prompt to send. Defaults to a short fixed prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Output format (json | markdown).
    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,

    /// Skip the readiness probe and start measuring immediately.
    #[arg(long)]
    no_wait: bool,
}

#[derive(Debug, Serialize)]
struct Params {
    host: String,
    port: u16,
    model_key: String,
    model_path: String,
    max_tokens: u32,
    warmup: u32,
    trials: u32,
    temperature: f32,
    top_k: Option<u32>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    prompt: String,
}

#[derive(Debug, Serialize, Clone)]
struct TrialResult {
    ttft_ms: f64,
    decode_tokps: f64,
    tokens_after_first: u32,
    total_tokens: u32,
    wall_ms: f64,
}

#[derive(Debug, Serialize)]
struct Results {
    trials: Vec<TrialResult>,
    ttft_ms_mean: f64,
    ttft_ms_median: f64,
    ttft_ms_p95: f64,
    ttft_ms_stdev: f64,
    decode_tokps_mean: f64,
    decode_tokps_median: f64,
    decode_tokps_p95: f64,
    decode_tokps_stdev: f64,
}

const DEFAULT_PROMPT: &str = "What is 2+2? Answer in one word.";

fn main() -> ExitCode {
    let args = Args::parse();
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: failed to start tokio runtime: {e}");
            return ExitCode::from(2);
        }
    };
    match runtime.block_on(run(args)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}

async fn run(args: Args) -> Result<()> {
    let mut metadata = RunMetadata::capture("bench_decode");
    let started = Instant::now();

    let manifest_path = args.manifest.clone().unwrap_or_else(default_manifest_path);
    let model = models::find_by_key(&manifest_path, &args.model)?;
    metadata.model = Some(ModelInfo {
        key: model.key.clone(),
        path: model.path.clone(),
        quantization: model.quantization.clone(),
        approx_size_gb: model.approx_size_gb,
    });

    let base_url = format!("http://{}:{}", args.host, args.port);
    if !args.no_wait {
        server::wait_until_ready(&base_url, Duration::from_secs(30))
            .await
            .with_context(|| {
                format!(
                    "higgs server not reachable at {base_url}; start it with `higgs serve --port {}`",
                    args.port
                )
            })?;
    }

    let prompt = args
        .prompt
        .clone()
        .unwrap_or_else(|| DEFAULT_PROMPT.to_owned());
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;

    for i in 0..args.warmup {
        eprintln!("[warmup {}/{}]", i + 1, args.warmup);
        let _ = run_trial(&client, &base_url, &model.path, &prompt, &args).await?;
    }

    let mut trials = Vec::with_capacity(args.trials as usize);
    for i in 0..args.trials {
        eprintln!("[trial  {}/{}]", i + 1, args.trials);
        let trial = run_trial(&client, &base_url, &model.path, &prompt, &args).await?;
        trials.push(trial);
    }

    let ttft: Vec<f64> = trials.iter().map(|t| t.ttft_ms).collect();
    let dec: Vec<f64> = trials.iter().map(|t| t.decode_tokps).collect();
    let results = Results {
        trials: trials.clone(),
        ttft_ms_mean: stats::mean(&ttft),
        ttft_ms_median: stats::median(&ttft),
        ttft_ms_p95: stats::p95(&ttft),
        ttft_ms_stdev: stats::stdev(&ttft),
        decode_tokps_mean: stats::mean(&dec),
        decode_tokps_median: stats::median(&dec),
        decode_tokps_p95: stats::p95(&dec),
        decode_tokps_stdev: stats::stdev(&dec),
    };

    metadata.duration_ms = started.elapsed().as_millis() as u64;

    let params = Params {
        host: args.host.clone(),
        port: args.port,
        model_key: model.key.clone(),
        model_path: model.path.clone(),
        max_tokens: args.max_tokens,
        warmup: args.warmup,
        trials: args.trials,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        frequency_penalty: args.frequency_penalty,
        prompt,
    };

    let output = BenchOutput {
        metadata,
        params,
        results,
    };

    let path = persist_result(&output)?;
    eprintln!("[persisted] {}", path.display());

    let rendered = match args.format {
        OutputFormat::Json => format_json(&output)?,
        OutputFormat::Markdown => format_markdown(&output)?,
    };
    println!("{rendered}");

    Ok(())
}

async fn run_trial(
    client: &reqwest::Client,
    base_url: &str,
    model_path: &str,
    prompt: &str,
    args: &Args,
) -> Result<TrialResult> {
    let mut body = serde_json::json!({
        "model": model_path,
        "messages": [{"role": "user", "content": prompt}],
        "stream": true,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    });
    if let Some(p) = args.top_p {
        body["top_p"] = serde_json::json!(p);
    }
    if let Some(k) = args.top_k {
        body["top_k"] = serde_json::json!(k);
    }
    if let Some(rp) = args.repetition_penalty {
        body["repetition_penalty"] = serde_json::json!(rp);
    }
    if let Some(fp) = args.frequency_penalty {
        body["frequency_penalty"] = serde_json::json!(fp);
    }

    let url = format!("{base_url}/v1/chat/completions");
    let started = Instant::now();
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("{url} returned HTTP {status}: {text}");
    }

    let mut stream = resp.bytes_stream();
    let mut first_token_at: Option<Instant> = None;
    let mut tokens_seen: u32 = 0;
    let mut buf = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("read SSE chunk")?;
        buf.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(idx) = buf.find('\n') {
            let line: String = buf.drain(..=idx).collect();
            let line = line.trim();
            if line.is_empty() || !line.starts_with("data:") {
                continue;
            }
            let data = line[5..].trim();
            if data == "[DONE]" {
                continue;
            }
            let value: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let delta = value
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("delta"))
                .and_then(|d| d.get("content"))
                .and_then(|s| s.as_str());
            if let Some(s) = delta {
                if !s.is_empty() {
                    if first_token_at.is_none() {
                        first_token_at = Some(Instant::now());
                    } else {
                        tokens_seen += 1;
                    }
                }
            }
        }
    }

    let total_elapsed = started.elapsed();
    let first_at =
        first_token_at.ok_or_else(|| anyhow::anyhow!("no streamed tokens received from server"))?;
    let ttft_ms = first_at.duration_since(started).as_secs_f64() * 1000.0;
    let after_first = total_elapsed.saturating_sub(first_at.duration_since(started));
    let decode_tokps = if after_first.as_secs_f64() > 0.0 && tokens_seen > 0 {
        f64::from(tokens_seen) / after_first.as_secs_f64()
    } else {
        0.0
    };

    Ok(TrialResult {
        ttft_ms,
        decode_tokps,
        tokens_after_first: tokens_seen,
        total_tokens: tokens_seen + 1,
        wall_ms: total_elapsed.as_secs_f64() * 1000.0,
    })
}
