//! Microbench for SSE chunk serialization.
//!
//! Compares the baseline `serde_json::to_string(&full_chunk)` path against the
//! production pre-serialized prefix path in `crate::sse::ChatChunkWriter`. By
//! calling the production writer directly (rather than re-implementing it
//! locally), the bench numbers stay tied to the code that actually serves
//! requests. Future changes to `crate::sse` show up here automatically.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use higgs::sse::ChatChunkWriter;
use higgs::types::openai::{ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionDelta};

fn bench_chunk_serialize(c: &mut Criterion) {
    let id = "chatcmpl-1234567890abcdef";
    let model = "mlx-community/Qwen3-1.7B-4bit";
    let created = 1_700_000_000_i64;
    let token_text = " hello";

    let mut group = c.benchmark_group("sse_chunk");

    group.bench_function("baseline_full_serde", |b| {
        b.iter(|| {
            let chunk = ChatCompletionChunk {
                id: id.to_owned(),
                object: "chat.completion.chunk",
                created,
                model: model.to_owned(),
                choices: vec![ChatCompletionChunkChoice {
                    index: 0,
                    delta: ChatCompletionDelta {
                        role: None,
                        content: Some(token_text.to_owned()),
                        reasoning_content: None,
                        tool_calls: None,
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
                usage: None,
            };
            let s = serde_json::to_string(black_box(&chunk)).unwrap();
            black_box(s);
        });
    });

    group.bench_function("prefix_writer", |b| {
        let mut w = ChatChunkWriter::new(id, created, model);
        b.iter(|| {
            let d = ChatCompletionDelta {
                role: None,
                content: Some(black_box(token_text).to_owned()),
                reasoning_content: None,
                tool_calls: None,
            };
            let out = w.write_delta(&d, None, None).unwrap();
            black_box(out.len());
        });
    });

    group.finish();
}

criterion_group!(benches, bench_chunk_serialize);
criterion_main!(benches);
