#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::as_conversions,
    clippy::doc_markdown,
    clippy::map_unwrap_or,
    clippy::unnecessary_map_or
)]
//! `bench_summarize` walks `target/bench-results/` and emits a Markdown
//! table grouped by model, with one row per (bench_name, model) pair.

use std::collections::BTreeMap;
use std::process::ExitCode;

use anyhow::Result;
use clap::Parser;
use higgs_bench::{OutputFormat, collect_results, results_dir};

#[derive(Debug, Parser)]
#[command(
    name = "bench_summarize",
    about = "Summarize persisted bench results from target/bench-results/",
    version
)]
struct Args {
    /// Filter to one bench name (e.g., `bench_decode`). Default: all.
    #[arg(long)]
    bench: Option<String>,

    /// Filter to a specific commit (short or full SHA). Default: all.
    #[arg(long)]
    commit: Option<String>,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutputFormat::Markdown)]
    format: OutputFormat,
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}

#[derive(Debug)]
struct Row {
    bench_name: String,
    model_key: String,
    model_label: String,
    commit_short: String,
    started_at: String,
    headline: String,
}

#[allow(clippy::too_many_lines)]
fn run(args: &Args) -> Result<()> {
    let dir = results_dir();
    let entries = collect_results(&dir)?;

    if entries.is_empty() {
        match args.format {
            OutputFormat::Markdown => {
                println!("# Bench summary\n");
                println!("_No results found in `{}`._\n", dir.display());
                println!("Run `bench_decode` (or another bench) first to populate this directory.");
            }
            OutputFormat::Json => {
                println!("{}", serde_json::json!({"results": []}));
            }
        }
        return Ok(());
    }

    let mut latest: BTreeMap<(String, String), (String, Row)> = BTreeMap::new();

    for entry in entries {
        let v = &entry.value;
        let Some(metadata) = v.get("metadata") else {
            continue;
        };
        let bench_name = metadata
            .get("bench_name")
            .and_then(|s| s.as_str())
            .unwrap_or("unknown")
            .to_owned();
        if let Some(filter) = &args.bench {
            if &bench_name != filter {
                continue;
            }
        }
        let commit = metadata
            .get("git_commit")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_owned();
        let commit_short = metadata
            .get("git_commit_short")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_owned();
        if let Some(filter) = &args.commit {
            if !commit.starts_with(filter.as_str()) && !commit_short.starts_with(filter.as_str()) {
                continue;
            }
        }
        let model_key = metadata
            .get("model")
            .and_then(|m| m.get("key"))
            .and_then(|s| s.as_str())
            .unwrap_or("no-model")
            .to_owned();
        let model_label = metadata
            .get("model")
            .and_then(|m| m.get("path"))
            .and_then(|s| s.as_str())
            .unwrap_or(&model_key)
            .to_owned();
        let started_at = metadata
            .get("started_at")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_owned();

        let headline = headline_for(&bench_name, v);

        let row = Row {
            bench_name: bench_name.clone(),
            model_key: model_key.clone(),
            model_label,
            commit_short,
            started_at: started_at.clone(),
            headline,
        };

        let key = (model_key, bench_name);
        let keep = latest
            .get(&key)
            .map_or(true, |(prev_ts, _)| started_at > *prev_ts);
        if keep {
            latest.insert(key, (started_at, row));
        }
    }

    let mut by_model: BTreeMap<String, Vec<Row>> = BTreeMap::new();
    for ((model, _bench), (_ts, row)) in latest {
        by_model.entry(model).or_default().push(row);
    }

    if by_model.is_empty() {
        match args.format {
            OutputFormat::Markdown => {
                println!("# Bench summary\n");
                println!("_No results matched the provided filters._\n");
            }
            OutputFormat::Json => {
                println!("{}", serde_json::json!({"results": []}));
            }
        }
        return Ok(());
    }

    match args.format {
        OutputFormat::Markdown => render_markdown(&by_model),
        OutputFormat::Json => render_json(&by_model)?,
    }
    Ok(())
}

fn headline_for(bench_name: &str, value: &serde_json::Value) -> String {
    let Some(results) = value.get("results") else {
        return String::new();
    };
    if bench_name == "bench_decode" {
        let ttft = results
            .get("ttft_ms_median")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        let dec = results
            .get("decode_tokps_median")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        return format!("{dec:.1} tok/s decode, {ttft:.0} ms TTFT");
    }
    String::new()
}

fn render_markdown(by_model: &BTreeMap<String, Vec<Row>>) {
    println!("# Bench summary\n");
    for (model, rows) in by_model {
        let label = rows
            .first()
            .map(|r| r.model_label.as_str())
            .unwrap_or(model.as_str());
        println!("## {model}");
        println!("\n_path: `{label}`_\n");
        println!("| Bench | Headline | Commit | When |");
        println!("|---|---|---|---|");
        for row in rows {
            println!(
                "| {} | {} | {} | {} |",
                row.bench_name, row.headline, row.commit_short, row.started_at
            );
        }
        println!();
    }
}

fn render_json(by_model: &BTreeMap<String, Vec<Row>>) -> Result<()> {
    let json = serde_json::json!({
        "models": by_model.iter().map(|(model, rows)| {
            serde_json::json!({
                "model_key": model,
                "rows": rows.iter().map(|r| serde_json::json!({
                    "bench_name": r.bench_name,
                    "model_key": r.model_key,
                    "commit": r.commit_short,
                    "started_at": r.started_at,
                    "headline": r.headline,
                })).collect::<Vec<_>>(),
            })
        }).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}
