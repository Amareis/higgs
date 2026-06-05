use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use higgs_engine::chat_template::ChatMessage;
use higgs_engine::mlx_tuning::{MlxRuntimeTuning, RequestedMlxProfile};
use higgs_engine::simple::SimpleEngine;
use higgs_models::SamplingParams;
use higgs_models::turboquant::KvCacheConfig;
use tokio::sync::mpsc;

const MAX_TOKENS: u32 = 2000;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let enable_thinking = !std::env::args().any(|a| a == "--no-thinking");

    let model_dir = Path::new(
        // "/Users/joe/.cache/huggingface/hub/models--mlx-community--Qwen3.5-4B-MLX-4bit/snapshots/32f3e8ecf65426fc3306969496342d504bfa13f3",
        "/Users/joe/work/kudach/mltry/mlx_models/Qwen3.5-4B-MLX-8bit",
    );

    println!("Loading model...");
    let tuning = MlxRuntimeTuning::from_model_dir(model_dir, RequestedMlxProfile::Auto);
    let engine = Arc::new(SimpleEngine::load(
        model_dir,
        KvCacheConfig::default(),
        tuning,
        true,
    )?);
    println!(
        "Model: {} | Thinking: {enable_thinking}\n",
        engine.model_name()
    );

    let params = SamplingParams {
        temperature: 0.7,
        top_p: 0.9,
        top_k: Some(40),
        ..Default::default()
    };

    // --- Turn 1 ---
    let messages1 = vec![
        ChatMessage {
            role: "system".into(),
            content: "You are a helpful assistant.".into(),
            tool_calls: None,
            num_images: 0,
        },
        ChatMessage {
            role: "user".into(),
            content: "Write a short story about a robot learning to paint.".into(),
            tool_calls: None,
            num_images: 0,
        },
    ];
    let assistant1 = generate_turn(&engine, &messages1, &params, enable_thinking, 1).await?;

    // --- Turn 2 (continues with REAL assistant response from turn 1) ---
    let messages2 = vec![
        ChatMessage {
            role: "system".into(),
            content: "You are a helpful assistant.".into(),
            tool_calls: None,
            num_images: 0,
        },
        ChatMessage {
            role: "user".into(),
            content: "Write a short story about a robot learning to paint.".into(),
            tool_calls: None,
            num_images: 0,
        },
        ChatMessage {
            role: "assistant".into(),
            content: assistant1,
            tool_calls: None,
            num_images: 0,
        },
        ChatMessage {
            role: "user".into(),
            content: "Now rewrite it as a haiku.".into(),
            tool_calls: None,
            num_images: 0,
        },
    ];
    let _assistant2 = generate_turn(&engine, &messages2, &params, enable_thinking, 2).await?;

    Ok(())
}

async fn generate_turn(
    engine: &Arc<SimpleEngine>,
    messages: &[ChatMessage],
    params: &SamplingParams,
    enable_thinking: bool,
    turn: u32,
) -> Result<String, Box<dyn std::error::Error>> {
    let prompt_tokens = if enable_thinking {
        engine.prepare_chat_prompt(messages, None)?
    } else {
        engine.prepare_chat_prompt_with_thinking(messages, None, false)?
    };

    println!("--- Turn {turn} ---");
    println!("Prompt tokens: {}", prompt_tokens.len());

    let (tx, mut rx) = mpsc::channel::<higgs_engine::engine::StreamingOutput>(32);
    let engine_cloned = Arc::clone(engine);
    let params_cloned = params.clone();
    let prompt_tokens_cloned = prompt_tokens.clone();

    let start_total = Instant::now();
    let handle = tokio::task::spawn_blocking(move || {
        engine_cloned.generate_streaming_with_thinking(
            &prompt_tokens_cloned,
            MAX_TOKENS,
            &params_cloned,
            &[],
            false,
            None,
            &tx,
            enable_thinking,
            None,
            None,
            None,
        )
    });

    let mut assistant = String::new();
    let mut prompt_tok = 0;
    let mut completion_tok = 0;
    let mut finish_reason = String::new();

    let mut first_chunk_time: Option<Instant> = None;
    let mut last_chunk_time = Instant::now();

    while let Some(chunk) = rx.recv().await {
        if first_chunk_time.is_none() {
            first_chunk_time = Some(Instant::now());
        }
        last_chunk_time = Instant::now();
        print!("{}", chunk.new_text);
        assistant.push_str(&chunk.new_text);
        if chunk.finished {
            prompt_tok = chunk.prompt_tokens;
            completion_tok = chunk.completion_tokens;
            finish_reason = chunk.finish_reason.unwrap_or_default();
            break;
        }
    }

    handle.await??;
    let total_duration = start_total.elapsed();

    let prefill_duration = first_chunk_time
        .map(|t| t.duration_since(start_total))
        .unwrap_or_default();
    let decode_duration = first_chunk_time
        .map(|t| last_chunk_time.duration_since(t))
        .unwrap_or_default();

    let prefill_tps = if prefill_duration.as_secs_f32() > 0.0 {
        prompt_tok as f32 / prefill_duration.as_secs_f32()
    } else {
        0.0
    };
    let decode_tps = if decode_duration.as_secs_f32() > 0.0 {
        completion_tok as f32 / decode_duration.as_secs_f32()
    } else {
        0.0
    };

    println!(
        "\n\n[turn {turn}: {} prompt + {} completion = {} total, finish={}]",
        prompt_tok,
        completion_tok,
        prompt_tok + completion_tok,
        finish_reason
    );
    println!(
        "  prefill:  {:.2}s ({:.1} tok/s)",
        prefill_duration.as_secs_f32(),
        prefill_tps
    );
    println!(
        "  decode:   {:.2}s ({:.1} tok/s)",
        decode_duration.as_secs_f32(),
        decode_tps
    );
    println!("  total:    {:.2}s\n", total_duration.as_secs_f32());

    Ok(assistant)
}
