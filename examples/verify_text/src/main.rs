use higgs_engine::chat_template::ChatMessage;
use higgs_engine::mlx_tuning::{MlxRuntimeTuning, RequestedMlxProfile};
use higgs_engine::simple::SimpleEngine;
use higgs_models::turboquant::KvCacheConfig;
use higgs_models::SamplingParams;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir =
        PathBuf::from(std::env::var("MODEL_DIR").unwrap_or_else(|_| {
            "/Users/joe/work/kudach/mltry/mlx_models/Qwen3.5-4B-MLX-8bit".into()
        }));

    let args: Vec<String> = std::env::args().collect();
    let prompt_text = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "Say ONLY 20 random words, each on new line".into());
    let enable_thinking = args.get(2).map(|s| s == "think").unwrap_or(false);

    println!("Loading model from {:?}...", model_dir);
    let tuning = MlxRuntimeTuning::from_model_dir(&model_dir, RequestedMlxProfile::Auto);
    let engine = SimpleEngine::load(&model_dir, KvCacheConfig::default(), tuning, true)?;
    println!("Model loaded: {}", engine.model_name());
    println!("Prompt: {:?}", prompt_text);
    println!("Thinking: {}", enable_thinking);

    let messages = vec![ChatMessage {
        role: "user".into(),
        content: prompt_text,
        tool_calls: None,
        num_images: 0,
    }];

    let prompt_tokens = if enable_thinking {
        engine.prepare_chat_prompt(&messages, None)?
    } else {
        engine.prepare_chat_prompt_with_thinking(&messages, None, false)?
    };
    println!("Prompt tokens: {:?}", prompt_tokens);

    let params = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: None,
        ..Default::default()
    };

    let (tx, mut rx) = tokio::sync::mpsc::channel::<higgs_engine::engine::StreamingOutput>(32);

    let handle = tokio::task::spawn_blocking({
        let engine = engine;
        let prompt_tokens = prompt_tokens.clone();
        move || {
            engine.generate_streaming_with_thinking(
                &prompt_tokens,
                50,
                &params,
                &[],
                false,
                None,
                &tx,
                enable_thinking,
                None,
                None,
            )
        }
    });

    println!("\nGenerated tokens:");
    let mut token_ids = Vec::new();
    let mut full_text = String::new();
    while let Some(chunk) = rx.recv().await {
        if !chunk.new_text.is_empty() {
            print!("{}", chunk.new_text);
            full_text.push_str(&chunk.new_text);
        }
        token_ids.push(chunk.current_token);
        if chunk.finished {
            break;
        }
    }
    println!("\n\nToken IDs: {:?}", token_ids);
    println!("Full text: {:?}", full_text);

    drop(handle);
    Ok(())
}
