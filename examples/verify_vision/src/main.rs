use clap::Parser;
use higgs_engine::chat_template::ChatMessage;
use higgs_engine::mlx_tuning::{MlxRuntimeTuning, RequestedMlxProfile};
use higgs_engine::simple::SimpleEngine;
use higgs_models::turboquant::KvCacheConfig;
use higgs_models::SamplingParams;

#[derive(Parser, Debug)]
#[command(name = "verify_vision")]
#[command(about = "Run vision-language inference on an image")]
struct Args {
    /// Path(s) to raw image files (jpg/png/etc). Can be specified multiple times.
    /// If given, the built-in Qwen3-VL preprocessor is used. Otherwise falls back
    /// to --baseline-dir.
    #[arg(short, long, action = clap::ArgAction::Append)]
    image: Vec<String>,

    /// Directory containing preprocessed baseline NPY files
    /// (pixel_values.npy, grid_thw.npy produced by verify_vision_baseline.py)
    #[arg(short, long, default_value = "/tmp/qwen3_vl_baseline")]
    baseline_dir: String,

    /// Path to the MLX model directory
    #[arg(
        short,
        long,
        default_value = "/Users/joe/work/kudach/mltry/mlx_models/Qwen3.5-4B-MLX-8bit"
    )]
    model: String,

    /// Prompt text to send along with the image
    #[arg(short, long, default_value = "What do you see on this image?")]
    prompt: String,

    /// Maximum number of tokens to generate
    #[arg(short = 't', long, default_value_t = 500)]
    max_tokens: u32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Load processor params from model's processor_config.json
    let proc = higgs_models::load_image_processor_params(&args.model);
    println!(
        "Processor params: patch_size={}, temporal_patch_size={}, merge_size={}, min_pixels={}, max_pixels={}",
        proc.patch_size, proc.temporal_patch_size, proc.merge_size, proc.min_pixels, proc.max_pixels
    );

    // Load or preprocess images
    let mut images: Vec<higgs_models::ProcessedImage> = Vec::with_capacity(args.image.len());
    for path in &args.image {
        println!("Processing image {:?}...", path);
        let img = higgs_models::qwen3_vl_processor::process_image_file(
            std::path::Path::new(path),
            proc.patch_size,
            proc.temporal_patch_size,
            proc.merge_size,
            proc.min_pixels,
            proc.max_pixels,
        )
        .map_err(|e| std::io::Error::other(format!("Image processing failed: {e}")))?;
        images.push(img);
    }

    // Compute number of image tokens from grid_thw
    let mut total_image_tokens = 0usize;
    for img in &images {
        let grid = img.grid_thw.as_ref().unwrap();
        let grid_slice = grid.as_slice::<i32>();
        let num_patches = grid_slice[0] * grid_slice[1] * grid_slice[2];
        total_image_tokens += (num_patches / (proc.merge_size * proc.merge_size)) as usize;
    }
    println!("total_image_tokens: {}", total_image_tokens);

    // Load engine
    println!("Loading model from {:?}...", args.model);
    let tuning = MlxRuntimeTuning::from_model_dir(
        std::path::Path::new(&args.model),
        RequestedMlxProfile::Auto,
    );
    let engine = SimpleEngine::load(&args.model, KvCacheConfig::default(), tuning, true)?;
    println!("Model loaded: {}", engine.model_name());

    // Prepare multimodal prompt through the engine's chat template
    let message = ChatMessage {
        role: "user".to_owned(),
        content: args.prompt,
        tool_calls: None,
        num_images: images.len(),
    };

    let (prompt_tokens, images_data) = engine
        .prepare_multimodal_prompt(&[message], &images, None, false)
        .map_err(|e| std::io::Error::other(format!("Prompt preparation failed: {e}")))?;

    println!("Prompt tokens: {} tokens", prompt_tokens.len());

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
                args.max_tokens,
                &params,
                &[],
                false,
                None,
                &tx,
                false, // enable_thinking
                None,
                images_data,
            )
        }
    });

    println!("\nGenerated text:");
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

    drop(handle);
    Ok(())
}
