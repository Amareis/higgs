//! Helpers for waiting on a higgs server to become ready.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};

/// Polls `GET /health` (and falls back to `/v1/models`) until the server
/// responds with success or `timeout` elapses.
///
/// Each probe is bounded by `min(remaining_deadline, 2s)` so a small overall
/// `timeout` is honored even when the server hangs mid-handshake.
pub async fn wait_until_ready(base_url: &str, timeout: Duration) -> Result<()> {
    let client = reqwest::Client::builder()
        .build()
        .context("build readiness probe client")?;

    let deadline = Instant::now() + timeout;
    let probes = [
        format!("{base_url}/health"),
        format!("{base_url}/v1/models"),
    ];
    let max_probe = Duration::from_secs(2);

    let mut last_err: Option<String> = None;
    while Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        for url in &probes {
            let req_timeout = remaining.min(max_probe);
            match tokio::time::timeout(req_timeout, client.get(url).send()).await {
                Ok(Ok(resp)) if resp.status().is_success() => return Ok(()),
                Ok(Ok(resp)) => last_err = Some(format!("{url} -> HTTP {}", resp.status())),
                Ok(Err(e)) => last_err = Some(format!("{url} -> {e}")),
                Err(_) => {
                    last_err = Some(format!("{url} -> timed out after {req_timeout:?}"));
                }
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
