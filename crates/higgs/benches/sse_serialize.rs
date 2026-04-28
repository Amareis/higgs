// Microbench for SSE chunk serialization. Compares the baseline
// `serde_json::to_string(&full_chunk)` path against the pre-serialized prefix
// path in `crate::sse`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use serde::Serialize;

// We can't reach the private `sse` module from a `benches/` target, so we
// re-implement the writer locally. It is a 1:1 copy of `crate::sse` and is
// kept in sync by the unit tests in that module which assert byte-equivalence
// with `serde_json::to_string`.

#[derive(Debug, Clone, Serialize)]
struct ChunkChoice<'a> {
    index: u32,
    delta: Delta<'a>,
    finish_reason: Option<&'a str>,
}

#[derive(Debug, Clone, Serialize)]
struct Delta<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
}

#[derive(Debug, Clone, Serialize)]
struct Chunk<'a> {
    id: &'a str,
    object: &'static str,
    created: i64,
    model: &'a str,
    choices: Vec<ChunkChoice<'a>>,
}

fn push_json_string(out: &mut String, s: &str) {
    if let Ok(encoded) = serde_json::to_string(s) {
        out.push_str(&encoded);
    }
}

struct Writer {
    prefix: String,
    buf: String,
}

impl Writer {
    fn new(id: &str, created: i64, model: &str) -> Self {
        let mut prefix = String::with_capacity(96 + id.len() + model.len());
        prefix.push_str(r#"{"id":"#);
        push_json_string(&mut prefix, id);
        prefix.push_str(r#","object":"chat.completion.chunk","created":"#);
        prefix.push_str(&created.to_string());
        prefix.push_str(r#","model":"#);
        push_json_string(&mut prefix, model);
        prefix.push_str(r#","choices":[{"index":0,"delta":"#);
        Self {
            prefix,
            buf: String::with_capacity(256),
        }
    }

    fn write_delta(&mut self, delta: &Delta) -> &str {
        self.buf.clear();
        self.buf.push_str(&self.prefix);
        let dj = serde_json::to_string(delta).unwrap();
        self.buf.push_str(&dj);
        self.buf.push_str(r#","finish_reason":null}]}"#);
        &self.buf
    }
}

fn bench_chunk_serialize(c: &mut Criterion) {
    let id = "chatcmpl-1234567890abcdef";
    let model = "mlx-community/Qwen3-1.7B-4bit";
    let created = 1_700_000_000_i64;
    let token_text = " hello";

    let mut group = c.benchmark_group("sse_chunk");

    group.bench_function("baseline_full_serde", |b| {
        b.iter(|| {
            let chunk = Chunk {
                id,
                object: "chat.completion.chunk",
                created,
                model,
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: Some(token_text),
                    },
                    finish_reason: None,
                }],
            };
            let s = serde_json::to_string(black_box(&chunk)).unwrap();
            black_box(s);
        });
    });

    group.bench_function("prefix_writer", |b| {
        let mut w = Writer::new(id, created, model);
        b.iter(|| {
            let d = Delta {
                role: None,
                content: Some(black_box(token_text)),
            };
            let out = w.write_delta(&d);
            black_box(out.len());
        });
    });

    group.finish();
}

criterion_group!(benches, bench_chunk_serialize);
criterion_main!(benches);
