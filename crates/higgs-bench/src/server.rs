//! Helpers for waiting on a higgs server to become ready.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};

/// Polls `GET /health` (and falls back to `/v1/models`) until the server
/// responds with success or `timeout` elapses.
pub async fn wait_until_ready(base_url: &str, timeout: Duration) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .context("build readiness probe client")?;

    let deadline = Instant::now() + timeout;
    let probes = [
        format!("{base_url}/health"),
        format!("{base_url}/v1/models"),
    ];

    let mut last_err: Option<String> = None;
    while Instant::now() < deadline {
        for url in &probes {
            match client.get(url).send().await {
                Ok(resp) if resp.status().is_success() => return Ok(()),
                Ok(resp) => last_err = Some(format!("{url} -> HTTP {}", resp.status())),
                Err(e) => last_err = Some(format!("{url} -> {e}")),
            }
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }

    Err(anyhow::anyhow!(
        "server at {base_url} did not become ready within {:?}: {}",
        timeout,
        last_err.unwrap_or_else(|| "no probes attempted".into())
    ))
}
