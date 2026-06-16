//! CLI tool for generating audio with deterministic seed
//!
//! This tool generates WAV audio files using the Qwen3-TTS pipeline with a
//! specific seed, allowing direct comparison with Python output.
//!
//! Usage:
//!     cargo run --features cli --bin generate_audio -- --text "Hello" --seed 42
//!     cargo run --features cli --bin generate_audio -- --text "Hello" --seed 42 --duration 10.0

use anyhow::Result;
use byteorder::{LittleEndian, WriteBytesExt};
use candle_core::{DType, Device, IndexOp, Tensor};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use candle_transformers::quantized_var_builder::VarBuilder;
use models::talker::{Language, Speaker, TalkerConfig};
use qwen3_tts::{
    device_info, generation, models, parse_device, tokenizer, AudioBuffer, ModelType,
    ParsedModelConfig, QuantizedCodePredictor, QuantizedTalkerModel, Qwen3TTS, SynthesisOptions,
};

/// Wrapper enum for F32 vs quantized code predictor
enum CpVariant {
    F32(models::CodePredictor),
    Q(QuantizedCodePredictor),
}

impl CpVariant {
    fn new_kv_caches(&self) -> Vec<models::AnyKVCache> {
        match self {
            Self::F32(cp) => cp.new_kv_caches(),
            Self::Q(cp) => cp.new_kv_caches(),
        }
    }

    fn generate_acoustic_codes(
        &self,
        talker_hidden: &Tensor,
        semantic_embed: &Tensor,
        cp_kv_caches: &mut [models::AnyKVCache],
    ) -> anyhow::Result<Tensor> {
        match self {
            Self::F32(cp) => {
                cp.generate_acoustic_codes(talker_hidden, semantic_embed, cp_kv_caches)
            }
            Self::Q(cp) => cp.generate_acoustic_codes(talker_hidden, semantic_embed, cp_kv_caches),
        }
    }

    fn get_acoustic_embeddings_sum_from_tensor(&self, codes: &Tensor) -> anyhow::Result<Tensor> {
        match self {
            Self::F32(cp) => cp.get_acoustic_embeddings_sum_from_tensor(codes),
            Self::Q(cp) => cp.get_acoustic_embeddings_sum_from_tensor(codes),
        }
    }
}

/// Generate reference audio with deterministic seed for comparison
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Text to synthesize
    #[arg(short, long, default_value = "Hello")]
    text: String,

    /// Random seed for reproducible generation
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Maximum number of frames to generate (default: 2048, ~164s). Generation
    /// stops early when the model emits an end-of-sequence token.
    #[arg(short, long, default_value_t = 2048)]
    frames: usize,

    /// Maximum duration in seconds (overrides --frames if specified).
    /// Generation stops early when the model emits an end-of-sequence token.
    #[arg(short, long)]
    duration: Option<f64>,

    /// Sampling temperature
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Top-k sampling parameter
    #[arg(long, default_value_t = 50)]
    top_k: usize,

    /// Top-p (nucleus) sampling parameter
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// Model directory containing model.safetensors
    #[arg(short, long, default_value = "test_data/model")]
    model_dir: String,

    /// Output directory for generated files
    #[arg(short, long, default_value = "test_data/rust_audio")]
    output_dir: String,

    /// Compare with Python reference output (if exists)
    #[arg(short, long)]
    compare: bool,

    /// Python reference directory
    #[arg(long, default_value = "test_data/reference_audio")]
    reference_dir: String,

    /// Tokenizer directory (defaults to model_dir/../tokenizer)
    #[arg(long)]
    tokenizer_dir: Option<String>,

    /// Speaker name for CustomVoice (ryan, serena, vivian, aiden, etc.)
    #[arg(long, default_value = "ryan")]
    speaker: String,

    /// Language for TTS (english, chinese, japanese, etc.)
    #[arg(long, default_value = "english")]
    language: String,

    /// Force 1.7B config (hidden=2048). Usually auto-detected from config.json.
    #[arg(long)]
    custom_voice: bool,

    /// Voice description for VoiceDesign model (e.g. "A cheerful young female voice")
    #[arg(long)]
    instruct: Option<String>,

    /// Reference audio WAV file for voice cloning (x_vector_only mode)
    #[arg(long)]
    ref_audio: Option<String>,

    /// Reference text for ICL voice cloning (requires --ref-audio)
    #[arg(long)]
    ref_text: Option<String>,

    /// Use x_vector_only mode (speaker embedding only, no ICL)
    #[arg(long)]
    x_vector_only: bool,

    /// Repetition penalty (1.0 = disabled, 1.05 = Python default)
    #[arg(long, default_value_t = 1.05)]
    repetition_penalty: f64,

    /// Output WAV file path (overrides default naming in --output-dir)
    #[arg(long)]
    output: Option<String>,

    /// Device for inference (auto, cpu, cuda, cuda:N, metal)
    #[arg(long, default_value = "auto")]
    device: String,

    /// Use quantized GGUF model instead of safetensors
    #[arg(long)]
    quantized: bool,

    /// Path to quantized GGUF model file (required when --quantized is set)
    #[arg(long)]
    gguf_path: Option<String>,
}

/// Metadata for generated audio
#[derive(Debug, Serialize, Deserialize)]
struct GenerationMetadata {
    text: String,
    seed: u64,
    num_frames: usize,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    input_ids: Vec<u32>,
    codes_shape: Vec<usize>,
    audio_samples: usize,
    sample_rate: u32,
}

/// Calculate max frames from --duration or --frames.
fn max_frames_from_args(args: &Args) -> usize {
    if let Some(duration) = args.duration {
        (duration * 12.5) as usize
    } else {
        args.frames
    }
}

/// Resolve output WAV path from --output or default naming in --output-dir.
fn resolve_output_path(args: &Args, max_frames: usize) -> Result<std::path::PathBuf> {
    let path = if let Some(ref out) = args.output {
        std::path::PathBuf::from(out)
    } else {
        let output_dir = Path::new(&args.output_dir);
        fs::create_dir_all(output_dir)?;
        output_dir.join(format!("audio_seed{}_frames{}.wav", args.seed, max_frames))
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(path)
}

/// Validate CLI arg combinations and bail early on contradictory flags.
fn validate_args(args: &Args) -> Result<()> {
    // --instruct and --ref-audio are mutually exclusive
    if args.instruct.is_some() && args.ref_audio.is_some() {
        anyhow::bail!(
            "--instruct and --ref-audio are mutually exclusive.\n  \
             --instruct is for VoiceDesign models (text-described voices).\n  \
             --ref-audio is for Base models (voice cloning from reference audio)."
        );
    }

    // --ref-text requires --ref-audio (ref text is an ICL transcript for voice cloning)
    if args.ref_text.is_some() && args.ref_audio.is_none() {
        anyhow::bail!("--ref-text requires --ref-audio (reference text is the transcript of the reference audio for ICL voice cloning)");
    }

    // --x-vector-only requires --ref-audio (it's a voice cloning mode)
    if args.x_vector_only && args.ref_audio.is_none() {
        anyhow::bail!(
            "--x-vector-only requires --ref-audio (x_vector_only is a voice cloning mode)"
        );
    }

    // --x-vector-only + --ref-text is contradictory: x_vector_only disables ICL
    if args.x_vector_only && args.ref_text.is_some() {
        anyhow::bail!(
            "--x-vector-only and --ref-text are mutually exclusive.\n  \
             x_vector_only uses only the speaker embedding (no ICL).\n  \
             Remove --x-vector-only to use ICL mode with reference text."
        );
    }

    // --custom-voice + --ref-audio: CustomVoice models don't have a speaker encoder
    if args.custom_voice && args.ref_audio.is_some() {
        anyhow::bail!(
            "--custom-voice and --ref-audio are incompatible.\n  \
             CustomVoice models use preset speakers (--speaker), not voice cloning.\n  \
             Use a Base model for voice cloning with reference audio."
        );
    }

    // --compare only works in the preset speaker path (low-level generation)
    if args.compare && args.ref_audio.is_some() {
        anyhow::bail!(
            "--compare is only supported in the preset speaker code path.\n  \
             Voice cloning uses the high-level API which doesn't emit comparison data."
        );
    }

    // --quantized requires --gguf-path
    if args.quantized && args.gguf_path.is_none() {
        anyhow::bail!("--quantized requires --gguf-path");
    }

    // --gguf-path without --quantized is a no-op warning
    if !args.quantized && args.gguf_path.is_some() {
        anyhow::bail!("--gguf-path requires --quantized");
    }

    // --quantized is incompatible with voice cloning and voice design (for now)
    if args.quantized && args.ref_audio.is_some() {
        anyhow::bail!("--quantized is not yet supported with --ref-audio (voice cloning)");
    }
    if args.quantized && args.instruct.is_some() {
        anyhow::bail!("--quantized is not yet supported with --instruct (VoiceDesign)");
    }

    Ok(())
}

/// Voice cloning path: uses the high-level Qwen3TTS API when --ref-audio is provided.
fn run_voice_clone(args: &Args) -> Result<()> {
    let ref_audio_path = args.ref_audio.as_ref().expect("--ref-audio required");

    println!("=== Voice Clone Generation ===");
    println!("Text: {}", args.text);
    println!("Reference audio: {}", ref_audio_path);
    if let Some(ref rt) = args.ref_text {
        println!("Reference text (ICL): {}", rt);
    } else {
        println!("Mode: x_vector_only (no reference text)");
    }

    println!("Seed: {}", args.seed);

    let device = parse_device(&args.device)?;
    println!("Device: {}", device_info(&device));
    let model = Qwen3TTS::from_pretrained_with_tokenizer(
        &args.model_dir,
        args.tokenizer_dir.as_deref(),
        device,
    )?;

    // Load reference audio
    let ref_audio = AudioBuffer::load(ref_audio_path)?;
    println!(
        "Loaded reference audio: {:.2}s, {} samples",
        ref_audio.duration(),
        ref_audio.len()
    );

    // Create voice clone prompt (x_vector_only or ICL depending on --ref-text)
    let prompt = model.create_voice_clone_prompt(&ref_audio, args.ref_text.as_deref())?;

    // Debug: print ref_codes info if ICL mode
    if let Some(ref codes) = &prompt.ref_codes {
        let shape = codes.shape();
        println!("ref_codes shape: {:?}", shape);
        // Print first frame
        let first_frame: Vec<u32> = codes.i(0)?.to_vec1()?;
        println!("ref_codes[0] (first frame): {:?}", first_frame);
        // Print first 10 semantic codes
        let semantic: Vec<u32> = codes.i((.., 0))?.to_vec1::<u32>()?;
        println!(
            "First 10 semantic codes: {:?}",
            &semantic[..10.min(semantic.len())]
        );
    }

    let language: Language = args.language.parse()?;
    let max_frames = max_frames_from_args(args);

    let options = SynthesisOptions {
        max_length: max_frames,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        seed: Some(args.seed),
        ..Default::default()
    };

    println!("Generating up to {} frames...", max_frames);
    let (audio, codes) =
        model.synthesize_voice_clone_debug(&args.text, &prompt, language, Some(options))?;
    println!(
        "Generated: {:.2}s, {} samples, {} frames",
        audio.duration(),
        audio.len(),
        codes.len()
    );

    // Print semantic tokens for debugging
    let semantic_tokens: Vec<u32> = codes.iter().map(|f| f[0]).collect();
    println!("Semantic tokens: {:?}", &semantic_tokens);
    let has_eos = semantic_tokens.contains(&qwen3_tts::CODEC_EOS_TOKEN_ID);
    println!(
        "Contains EOS ({}): {}",
        qwen3_tts::CODEC_EOS_TOKEN_ID,
        has_eos
    );

    let output_path = resolve_output_path(args, max_frames)?;
    audio.save(&output_path)?;
    println!("Saved WAV to: {}", output_path.display());

    println!("Generation complete!");
    Ok(())
}

/// VoiceDesign path: uses text-described voice conditioning when --instruct is provided.
fn run_voice_design(args: &Args) -> Result<()> {
    let instruct = args.instruct.as_ref().expect("--instruct required");

    println!("=== VoiceDesign Generation ===");
    println!("Text: {}", args.text);
    println!("Instruct: {}", instruct);

    println!("Seed: {}", args.seed);

    let device = parse_device(&args.device)?;
    println!("Device: {}", device_info(&device));
    let model = Qwen3TTS::from_pretrained_with_tokenizer(
        &args.model_dir,
        args.tokenizer_dir.as_deref(),
        device,
    )?;

    if !model.supports_voice_design() {
        eprintln!(
            "  WARNING: This model is not a VoiceDesign variant. \
             Voice design synthesis may produce unpredictable results."
        );
    }

    let language: Language = args.language.parse()?;
    let max_frames = max_frames_from_args(args);

    let options = SynthesisOptions {
        max_length: max_frames,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        seed: Some(args.seed),
        ..Default::default()
    };

    println!("Generating up to {} frames...", max_frames);
    let audio = model.synthesize_voice_design(&args.text, instruct, language, Some(options))?;
    println!(
        "Generated: {:.2}s, {} samples",
        audio.duration(),
        audio.len()
    );

    let output_path = resolve_output_path(args, max_frames)?;
    audio.save(&output_path)?;
    println!("Saved WAV to: {}", output_path.display());

    println!("Generation complete!");
    Ok(())
}

/// Quantized GGUF path: loads talker from GGUF, code predictor + decoder from safetensors.
///
/// Since QuantizedCodePredictor doesn't exist yet, the small code predictor
/// (5 layers, 1024 hidden) is loaded from model.safetensors as regular weights.
fn run_quantized(args: &Args) -> Result<()> {
    let start_all = std::time::Instant::now();
    let mut phase = std::time::Instant::now();

    let gguf_path = args.gguf_path.as_deref().expect("--gguf-path required");

    println!("=== Quantized GGUF Generation ===");
    println!("Text: {}", args.text);
    println!("GGUF: {}", gguf_path);
    println!("Seed: {}", args.seed);

    let device = parse_device(&args.device)?;
    println!("Device: {}", device_info(&device));

    // ── Load GGUF model ──────────────────────────────────────────────────────
    println!("\nLoading quantized model from GGUF...");
    let vb = VarBuilder::from_gguf(gguf_path, &device)?;
    let talker_config = if args.custom_voice {
        println!("Using CustomVoice config (hidden=2048, MRoPE)");
        TalkerConfig::custom_voice()
    } else {
        TalkerConfig::default()
    };
    let hidden_size = talker_config.hidden_size;
    let talker = QuantizedTalkerModel::from_gguf(vb.pp("talker"), talker_config, &device)?;
    let t_gguf = phase.elapsed().as_secs_f64();
    println!(
        "QuantizedTalkerModel loaded (hidden_size={}) — {:.2}s",
        hidden_size, t_gguf
    );
    phase = std::time::Instant::now();

    // ── Load decoder ──────────────────────────────────────────────────────────
    let decoder_path = Path::new(&args.model_dir).join("speech_tokenizer/model.safetensors");
    let decoder_weights = load_weights(&decoder_path, &device)?;
    println!("Creating Decoder12Hz...");
    let decoder = models::codec::Decoder12Hz::from_weights(&decoder_weights, Default::default())?;
    let t_decoder = phase.elapsed().as_secs_f64();
    println!("Decoder loaded — {:.2}s", t_decoder);
    phase = std::time::Instant::now();

    // ── Load tokenizer & tokenize ─────────────────────────────────────────────
    let tokenizer_dir = args
        .tokenizer_dir
        .clone()
        .unwrap_or_else(|| args.model_dir.clone());
    println!("Loading tokenizer from {}...", tokenizer_dir);
    let text_tokenizer = tokenizer::TextTokenizer::from_pretrained(&tokenizer_dir)?;
    let input_ids = text_tokenizer.encode(&args.text)?;
    println!("Input IDs: {:?}", input_ids);
    let speaker: Speaker = args.speaker.parse()?;
    let language: Language = args.language.parse()?;
    println!("\nSpeaker: {:?}, Language: {:?}", speaker, language);
    let num_frames = max_frames_from_args(args);
    println!("\nGenerating up to {} frames...", num_frames);
    let t_tokenizer = phase.elapsed().as_secs_f64();
    phase = std::time::Instant::now();

    // ── Load code predictor ──────────────────────────────────────────────────
    let cp_config = if args.custom_voice {
        models::CodePredictorConfig::custom_voice()
    } else {
        models::CodePredictorConfig::default()
    };
    let code_predictor = if vb.contains_key("talker.code_predictor.model.norm.weight") {
        println!("Loading code predictor from GGUF (quantized)...");
        CpVariant::Q(QuantizedCodePredictor::new(
            cp_config,
            vb.pp("talker.code_predictor"),
        )?)
    } else {
        println!("Loading code predictor weights from model.safetensors...");
        let model_path = Path::new(&args.model_dir).join("model.safetensors");
        let weights = load_weights(&model_path, &device)?;
        let cp_weights = filter_weights(&weights, "talker.code_predictor.");
        let cp_vb = candle_nn::VarBuilder::from_tensors(cp_weights, DType::F32, &device);
        let cp = models::CodePredictor::new(cp_config, cp_vb)?;
        CpVariant::F32(cp)
    };
    let t_cp = phase.elapsed().as_secs_f64();
    println!("Code predictor loaded — {:.2}s", t_cp);
    phase = std::time::Instant::now();

    // Build trailing text embeddings
    let trailing_text_hidden = if input_ids.len() > 1 {
        let remaining_proj = talker.get_projected_text_embeddings(&input_ids[1..])?;
        let tts_eos_embed = talker.get_tts_eos_embed()?;
        Tensor::cat(&[&remaining_proj, &tts_eos_embed], 1)?
    } else {
        talker.get_tts_eos_embed()?
    };
    let trailing_text_len = trailing_text_hidden.dim(1)?;
    let tts_pad_embed = talker.get_tts_pad_embed()?;
    println!("Trailing text length: {} positions", trailing_text_len);

    // Create sampling context with deterministic seed
    let mut sampling_ctx = generation::SamplingContext::new(Some(args.seed));
    let gen_config = generation::GenerationConfig {
        max_new_tokens: num_frames,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        eos_token_id: Some(qwen3_tts::CODEC_EOS_TOKEN_ID),
        min_new_tokens: 2,
    };

    // Initialize KV caches
    let mut kv_caches = talker.new_kv_caches(gen_config.max_new_tokens + 256);

    // Prefill
    println!("Running prefill...");
    let (hidden, logits) =
        talker.prefill_custom_voice(&input_ids, speaker, language, &mut kv_caches)?;
    let prefill_len = hidden.dim(1)?;
    let mut offset = prefill_len;
    let mut last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;
    let t_prefill = phase.elapsed().as_secs_f64();
    println!(
        "Prefill done — {:.2}s ({} positions)",
        t_prefill, prefill_len
    );
    phase = std::time::Instant::now();

    // Sample first semantic token
    let logits_2d = logits.squeeze(1)?;
    let logits_suppressed = generation::apply_token_suppression(
        &logits_2d,
        qwen3_tts::codec_tokens::CODEC_VOCAB_SIZE,
        qwen3_tts::CODEC_EOS_TOKEN_ID,
    )?;
    let first_token = generation::sample(&logits_suppressed, &gen_config, &mut sampling_ctx)?;
    let mut semantic_token_tensor = first_token.flatten_all()?; // [1] tensor on GPU
    let _first_semantic: u32 = semantic_token_tensor.to_vec1::<u32>()?[0];
    println!("First semantic token: {}", _first_semantic);

    // Generation loop — keep everything on GPU, single per-frame sync for EOS check
    let mut gpu_frames: Vec<Tensor> = Vec::with_capacity(num_frames);
    let mut cp_kv_caches = code_predictor.new_kv_caches();
    let progress = ProgressBar::new(num_frames as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} frames",
            )?
            .progress_chars("#>-"),
    );

    for frame_idx in 0..num_frames {
        // EOS check — one scalar sync per frame (~0.01ms on CUDA)
        let semantic_token: u32 = semantic_token_tensor.to_vec1::<u32>()?[0];
        if let Some(eos_id) = gen_config.eos_token_id {
            if semantic_token == eos_id {
                println!("\nEOS token {} at frame {} — stopping", eos_id, frame_idx);
                break;
            }
        }

        // GPU-resident embedding lookup (no CPU→GPU transfer)
        let semantic_embed = talker.get_codec_embedding_from_tensor(&semantic_token_tensor)?;

        let acoustic_codes_tensor = code_predictor.generate_acoustic_codes(
            &last_hidden,
            &semantic_embed,
            &mut cp_kv_caches,
        )?;

        // Accumulate frame as GPU tensor [semantic, 15 acoustics] — no CPU→GPU sync
        gpu_frames.push(Tensor::cat(
            &[&semantic_token_tensor.reshape(1)?, &acoustic_codes_tensor],
            0,
        )?);
        progress.inc(1);

        if frame_idx == num_frames - 1 {
            break;
        }

        // GPU tensor embedding sum (avoids 15 individual CPU→GPU transfers)
        let acoustic_embed_sum =
            code_predictor.get_acoustic_embeddings_sum_from_tensor(&acoustic_codes_tensor)?;
        let summed = semantic_embed.add(&acoustic_embed_sum)?;

        let text_addition = if frame_idx < trailing_text_len {
            trailing_text_hidden.i((.., frame_idx..frame_idx + 1, ..))?
        } else {
            tts_pad_embed.clone()
        };
        let step_input = summed.add(&text_addition)?;

        let (h, new_logits) =
            talker.generate_step_with_embed(&step_input, &mut kv_caches, offset)?;
        offset += 1;
        last_hidden = h;

        let logits_2d = new_logits.squeeze(1)?;
        let logits_suppressed = generation::apply_token_suppression(
            &logits_2d,
            qwen3_tts::codec_tokens::CODEC_VOCAB_SIZE,
            qwen3_tts::CODEC_EOS_TOKEN_ID,
        )?;
        semantic_token_tensor =
            generation::sample(&logits_suppressed, &gen_config, &mut sampling_ctx)?;
    }

    let t_gen = phase.elapsed().as_secs_f64();
    progress.finish_with_message("Done generating codes");
    phase = std::time::Instant::now();

    // Single GPU→CPU transfer + build [1, 16, num_frames] decoder tensor
    let num_final_frames = gpu_frames.len();
    println!("\nDecoding {} frames to audio...", num_final_frames);
    let codes_tensor = if gpu_frames.is_empty() {
        Tensor::zeros((1, 16, 0), DType::I64, &device)?
    } else {
        let stacked = Tensor::stack(&gpu_frames, 0)?; // [n, 16] U32 on GPU
        stacked
            .transpose(0, 1)? // [16, n]
            .unsqueeze(0)? // [1, 16, n]
            .to_dtype(DType::I64)?
    };
    let waveform = decoder.decode(&codes_tensor)?;
    let audio_samples: Vec<f32> = waveform.flatten_all()?.to_vec1()?;
    println!(
        "Audio: {} samples ({:.3}s at 24kHz)",
        audio_samples.len(),
        audio_samples.len() as f64 / 24000.0
    );

    // Save WAV
    let output_path = resolve_output_path(args, num_frames)?;
    let audio_buffer = AudioBuffer::new(audio_samples, 24000);
    audio_buffer.save(&output_path)?;
    let t_decode_save = phase.elapsed().as_secs_f64();

    println!();
    println!("=== Timing Summary ===");
    println!("  Load GGUF:        {:>8.2}s", t_gguf);
    println!("  Load decoder:     {:>8.2}s", t_decoder);
    println!("  Tokenizer:        {:>8.2}s", t_tokenizer);
    println!("  Load CP:          {:>8.2}s", t_cp);
    println!("  Prefill:          {:>8.2}s", t_prefill);
    println!(
        "  Generation:       {:>8.2}s  ({} frames, {:.1}s/frame)",
        t_gen,
        num_final_frames,
        t_gen / num_final_frames.max(1) as f64
    );
    println!("  Decode + save:    {:>8.2}s", t_decode_save);
    println!("  ──────────────────────────────────────");
    println!(
        "  Total:            {:>8.2}s",
        start_all.elapsed().as_secs_f64()
    );

    Ok(())
}

fn main() -> Result<()> {
    // Use chrome tracing when `profiling` feature is active, otherwise plain fmt.
    let _profiling_guard = qwen3_tts::profiling::init();
    if _profiling_guard.is_none() {
        tracing_subscriber::fmt::init();
    }

    let args = Args::parse();
    validate_args(&args)?;

    // Quantized GGUF path: loads talker from GGUF + code predictor from safetensors
    if args.quantized {
        return run_quantized(&args);
    }

    // Voice clone path: when --ref-audio is provided, use the high-level API
    if args.ref_audio.is_some() {
        return run_voice_clone(&args);
    }

    // VoiceDesign path: when --instruct is provided, use text-described voice conditioning
    if args.instruct.is_some() {
        return run_voice_design(&args);
    }

    let num_frames = max_frames_from_args(&args);

    println!("=== Generating Audio (Rust) ===");
    println!("Text: {}", args.text);
    println!("Seed: {}", args.seed);
    println!("Frames: {}", num_frames);
    println!("Temperature: {}", args.temperature);
    println!("Top-k: {}", args.top_k);
    println!("Top-p: {}", args.top_p);

    // Create sampling context with deterministic seed
    let mut sampling_ctx = generation::SamplingContext::new(Some(args.seed));
    println!("\nSeed: {}", args.seed);

    let device = parse_device(&args.device)?;
    println!("Device: {}", device_info(&device));

    // Create output directory
    let output_dir = Path::new(&args.output_dir);
    fs::create_dir_all(output_dir)?;

    // Load weights
    println!("\nLoading model weights...");
    let model_path = Path::new(&args.model_dir).join("model.safetensors");
    let weights = load_weights(&model_path, &device)?;

    let decoder_path = Path::new(&args.model_dir).join("speech_tokenizer/model.safetensors");
    let decoder_weights = load_weights(&decoder_path, &device)?;

    // Load tokenizer (defaults to model_dir, which has vocab.json + merges.txt)
    let tokenizer_dir = args.tokenizer_dir.unwrap_or_else(|| args.model_dir.clone());
    println!("Loading tokenizer from {}...", tokenizer_dir);
    let text_tokenizer = tokenizer::TextTokenizer::from_pretrained(&tokenizer_dir)?;

    // Tokenize text
    let input_ids = text_tokenizer.encode(&args.text)?;
    println!("Input IDs: {:?}", input_ids);
    // Create models — auto-detect from config.json, fall back to --custom-voice flag
    let config_path = Path::new(&args.model_dir).join("config.json");
    let parsed_config = if config_path.exists() {
        match ParsedModelConfig::from_file(&config_path) {
            Ok(cfg) => {
                println!("Detected model variant: {}", cfg.label());
                Some(cfg)
            }
            Err(e) => {
                eprintln!("Warning: failed to parse config.json: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Validate CLI args against detected model variant
    if let Some(ref cfg) = parsed_config {
        match cfg.model_type {
            ModelType::Base => {
                if args.ref_audio.is_none() {
                    eprintln!();
                    eprintln!(
                        "  WARNING: This is a {} model (trained for voice cloning).",
                        cfg.label()
                    );
                    eprintln!("  Without --ref-audio, the output voice will be unpredictable.");
                    eprintln!("  Recommended usage:");
                    eprintln!("    --ref-audio <path.wav>                 (x_vector_only mode)");
                    eprintln!("    --ref-audio <path.wav> --ref-text ...  (ICL mode)");
                    eprintln!();
                }
            }
            ModelType::CustomVoice => {
                if args.ref_audio.is_some() {
                    eprintln!();
                    eprintln!(
                        "  WARNING: This is a {} model (preset speakers only).",
                        cfg.label()
                    );
                    eprintln!("  --ref-audio is ignored — CustomVoice models don't have a speaker encoder.");
                    eprintln!(
                        "  Use --speaker <name> instead. Available: ryan, serena, vivian, aiden,"
                    );
                    eprintln!("  uncle_fu, ono_anna, sohee, eric, dylan");
                    eprintln!();
                }
            }
            ModelType::VoiceDesign => {
                if args.instruct.is_none() {
                    eprintln!();
                    eprintln!(
                        "  WARNING: This is a {} model (text-described voices).",
                        cfg.label()
                    );
                    eprintln!(
                        "  Without --instruct, falling back to preset speaker prefill — voice will be unpredictable."
                    );
                    eprintln!("  Recommended usage:");
                    eprintln!("    --instruct \"A cheerful young female voice with high pitch\"");
                    eprintln!();
                }
            }
        }
    }

    let (talker_config, cp_config) = if let Some(ref cfg) = parsed_config {
        (
            TalkerConfig::from_parsed(cfg),
            models::CodePredictorConfig::from_parsed(cfg),
        )
    } else if args.custom_voice {
        println!("Using CustomVoice config (hidden=2048, MRoPE)");
        (
            TalkerConfig::custom_voice(),
            models::CodePredictorConfig::custom_voice(),
        )
    } else {
        (
            TalkerConfig::default(),
            models::CodePredictorConfig::default(),
        )
    };

    println!(
        "Creating TalkerModel (hidden={})...",
        talker_config.hidden_size
    );
    let talker = models::TalkerModel::from_weights_with_config(&weights, talker_config, &device)?;

    println!("Creating CodePredictor...");
    let cp_weights = filter_weights(&weights, "talker.code_predictor.");
    let cp_vb = candle_nn::VarBuilder::from_tensors(cp_weights, DType::F32, &device);
    let code_predictor = models::CodePredictor::new(cp_config, cp_vb)?;

    println!("Creating Decoder12Hz...");
    let decoder = models::codec::Decoder12Hz::from_weights(&decoder_weights, Default::default())?;

    // Parse speaker and language for CustomVoice prefill
    let speaker: Speaker = args.speaker.parse()?;
    let language: Language = args.language.parse()?;
    println!("\nSpeaker: {:?}, Language: {:?}", speaker, language);

    // Build trailing text embeddings:
    //   remaining text tokens (all except first) projected + tts_eos
    //   After trailing text is exhausted, tts_pad is used for each subsequent step
    let trailing_text_hidden = if input_ids.len() > 1 {
        let remaining_proj = talker.get_projected_text_embeddings(&input_ids[1..])?;
        let tts_eos_embed = talker.get_tts_eos_embed()?;
        Tensor::cat(&[&remaining_proj, &tts_eos_embed], 1)?
    } else {
        talker.get_tts_eos_embed()?
    };
    let trailing_text_len = trailing_text_hidden.dim(1)?;
    let tts_pad_embed = talker.get_tts_pad_embed()?;
    println!("Trailing text length: {} positions", trailing_text_len);

    // Generate codes
    println!("\nGenerating {} frames...", num_frames);
    let progress = ProgressBar::new(num_frames as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} frames",
            )?
            .progress_chars("#>-"),
    );

    let gen_config = generation::GenerationConfig {
        max_new_tokens: num_frames,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        eos_token_id: if !args.compare {
            Some(qwen3_tts::CODEC_EOS_TOKEN_ID)
        } else {
            None // Don't stop early when comparing with Python reference
        },
        min_new_tokens: 2,
    };

    // Initialize KV caches
    let mut kv_caches = talker.new_kv_caches(gen_config.max_new_tokens + 256);

    // Prefill with CustomVoice format:
    //   [role_prefix (im_start, assistant, newline)]
    //   [tts_pad×5 + tts_bos ADDED with codec_control (think, think_bos, lang, think_eos, speaker, pad)]
    //   [first_text_proj + codec_bos]
    let (hidden, logits) =
        talker.prefill_custom_voice(&input_ids, speaker, language, &mut kv_caches)?;
    let prefill_len = hidden.dim(1)?;
    let mut offset = prefill_len;
    println!("Prefill length: {} positions", prefill_len);

    // Get last hidden state (for code predictor input)
    let mut last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

    // Track generated tokens for repetition penalty
    let mut generated_tokens: Vec<u32> = Vec::with_capacity(num_frames);

    // Apply repetition penalty + token suppression and sample first semantic token
    let logits_2d = logits.squeeze(1)?;
    let logits_2d = if gen_config.repetition_penalty != 1.0 && !generated_tokens.is_empty() {
        let prev = Tensor::new(generated_tokens.as_slice(), &device)?;
        generation::apply_repetition_penalty(&logits_2d, &prev, gen_config.repetition_penalty)?
    } else {
        logits_2d
    };
    let logits_suppressed = generation::apply_token_suppression(
        &logits_2d,
        qwen3_tts::codec_tokens::CODEC_VOCAB_SIZE,
        qwen3_tts::CODEC_EOS_TOKEN_ID,
    )?;
    let first_token = generation::sample(&logits_suppressed, &gen_config, &mut sampling_ctx)?;
    let mut semantic_token_tensor = first_token.flatten_all()?; // [1] tensor on GPU
    let _first_semantic: u32 = semantic_token_tensor.to_vec1::<u32>()?[0];
    generated_tokens.push(_first_semantic);
    println!("First semantic token: {}", _first_semantic);

    // Generation loop — keep semantic_token on GPU, single sync per frame for EOS
    let mut gpu_frames: Vec<Tensor> = Vec::with_capacity(num_frames);
    let mut cp_kv_caches = code_predictor.new_kv_caches();

    for frame_idx in 0..num_frames {
        // EOS check — one scalar sync per frame
        let semantic_token: u32 = semantic_token_tensor.to_vec1::<u32>()?[0];
        if let Some(eos_id) = gen_config.eos_token_id {
            if semantic_token == eos_id {
                println!(
                    "EOS token {} at frame {} — stopping generation",
                    eos_id, frame_idx
                );
                break;
            }
        }

        // GPU-resident embedding lookup (no CPU→GPU transfer)
        let semantic_embed = talker.get_codec_embedding_from_tensor(&semantic_token_tensor)?;

        let acoustic_codes_tensor = code_predictor.generate_acoustic_codes(
            &last_hidden,
            &semantic_embed,
            &mut cp_kv_caches,
        )?;

        // Accumulate frame as GPU tensor
        gpu_frames.push(Tensor::cat(
            &[&semantic_token_tensor.reshape(1)?, &acoustic_codes_tensor],
            0,
        )?);

        if frame_idx < 5 || frame_idx == num_frames - 1 {
            let debug_codes: Vec<u32> = acoustic_codes_tensor.narrow(0, 0, 3)?.to_vec1::<u32>()?;
            println!(
                "Frame {}: semantic={}, acoustics={:?}...",
                frame_idx, semantic_token, &debug_codes
            );
        } else if frame_idx == 5 {
            println!("...");
        }

        progress.inc(1);

        if frame_idx == num_frames - 1 {
            break;
        }

        // GPU tensor embedding sum (avoids 15 individual CPU→GPU transfers)
        let acoustic_embed_sum =
            code_predictor.get_acoustic_embeddings_sum_from_tensor(&acoustic_codes_tensor)?;
        let summed = semantic_embed.add(&acoustic_embed_sum)?;

        let text_addition = if frame_idx < trailing_text_len {
            trailing_text_hidden.i((.., frame_idx..frame_idx + 1, ..))?
        } else {
            tts_pad_embed.clone()
        };
        let step_input = summed.add(&text_addition)?;

        let (h, new_logits) =
            talker.generate_step_with_embed(&step_input, &mut kv_caches, offset)?;
        offset += 1;
        last_hidden = h;

        // Sample next semantic token with repetition penalty (CPU-generated_tokens path)
        let logits_2d = new_logits.squeeze(1)?;
        let logits_2d = if gen_config.repetition_penalty != 1.0 {
            let prev = Tensor::new(generated_tokens.as_slice(), &device)?;
            generation::apply_repetition_penalty(&logits_2d, &prev, gen_config.repetition_penalty)?
        } else {
            logits_2d
        };
        let logits_suppressed = generation::apply_token_suppression(
            &logits_2d,
            qwen3_tts::codec_tokens::CODEC_VOCAB_SIZE,
            qwen3_tts::CODEC_EOS_TOKEN_ID,
        )?;
        semantic_token_tensor =
            generation::sample(&logits_suppressed, &gen_config, &mut sampling_ctx)?;
        let new_token: u32 = semantic_token_tensor.to_vec1::<u32>()?[0];
        generated_tokens.push(new_token);
    }

    progress.finish_with_message("Done generating codes");

    // Single GPU→CPU transfer + build [1, 16, num_frames] decoder tensor
    let num_final_frames = gpu_frames.len();
    let codes_tensor = if gpu_frames.is_empty() {
        Tensor::zeros((1, 16, 0), DType::I64, &device)?
    } else {
        let stacked = Tensor::stack(&gpu_frames, 0)?; // [n, 16] U32 on GPU
        stacked
            .transpose(0, 1)? // [16, n]
            .unsqueeze(0)? // [1, 16, n]
            .to_dtype(DType::I64)?
    };
    println!("\nCodes tensor shape: {:?}", codes_tensor.shape());

    // Save codes as binary (single to_vec1 after generation completes)
    let all_codes_flat: Vec<u32> = if num_final_frames > 0 {
        Tensor::stack(&gpu_frames, 0)?
            .flatten_all()?
            .to_vec1::<u32>()?
    } else {
        Vec::new()
    };
    let mut all_codes: Vec<Vec<u32>> = Vec::with_capacity(num_final_frames);
    for f in 0..num_final_frames {
        let start = f * 16;
        all_codes.push(all_codes_flat[start..start + 16].to_vec());
    }
    let codes_bin_path =
        output_dir.join(format!("codes_seed{}_frames{}.bin", args.seed, num_frames));
    save_codes_binary(&all_codes, &codes_bin_path)?;
    println!("Saved binary codes to: {:?}", codes_bin_path);

    // Decode to audio (GPU tensor already on device)
    println!("\nDecoding to audio...");
    let waveform = decoder.decode(&codes_tensor)?;
    let audio_samples: Vec<f32> = waveform.flatten_all()?.to_vec1()?;
    println!(
        "Audio samples: {} ({:.3}s at 24kHz)",
        audio_samples.len(),
        audio_samples.len() as f64 / 24000.0
    );

    // Save audio as WAV
    let wav_path = if let Some(ref out) = args.output {
        let p = std::path::PathBuf::from(out);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent)?;
        }
        p
    } else {
        output_dir.join(format!("audio_seed{}_frames{}.wav", args.seed, num_frames))
    };
    let audio_buffer = AudioBuffer::new(audio_samples.clone(), 24000);
    audio_buffer.save(&wav_path)?;
    println!("Saved WAV to: {:?}", wav_path);

    // Save audio as binary
    let audio_bin_path =
        output_dir.join(format!("audio_seed{}_frames{}.bin", args.seed, num_frames));
    save_audio_binary(&audio_samples, &audio_bin_path)?;
    println!("Saved binary audio to: {:?}", audio_bin_path);

    // Save metadata
    let metadata = GenerationMetadata {
        text: args.text.clone(),
        seed: args.seed,
        num_frames,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        input_ids,
        codes_shape: vec![1, 16, num_frames],
        audio_samples: audio_samples.len(),
        sample_rate: 24000,
    };
    let metadata_path = output_dir.join(format!(
        "metadata_seed{}_frames{}.json",
        args.seed, num_frames
    ));
    let metadata_file = File::create(&metadata_path)?;
    serde_json::to_writer_pretty(metadata_file, &metadata)?;
    println!("Saved metadata to: {:?}", metadata_path);

    // Compare with Python reference if requested
    if args.compare {
        println!("\n=== Comparing with Python Reference ===");
        compare_with_reference(
            &args.reference_dir,
            args.seed,
            num_frames,
            &all_codes,
            &audio_samples,
        )?;
    }

    println!("\nGeneration complete!");

    Ok(())
}

/// Load weights from safetensors file
fn load_weights(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let tensors: HashMap<String, Tensor> = candle_core::safetensors::load(path, device)?;
    let mut converted = HashMap::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        let t = if tensor.dtype() == DType::BF16 {
            tensor.to_dtype(DType::F32)?
        } else {
            tensor
        };
        converted.insert(name, t);
    }
    Ok(converted)
}

/// Filter weights by prefix, removing the prefix from keys.
fn filter_weights(weights: &HashMap<String, Tensor>, prefix: &str) -> HashMap<String, Tensor> {
    weights
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix(prefix)
                .map(|stripped| (stripped.to_string(), v.clone()))
        })
        .collect()
}

/// Save codes as binary (row-major: frame0_q0, frame0_q1, ..., frame1_q0, ...)
fn save_codes_binary(codes: &[Vec<u32>], path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write as i64 to match Python format
    for frame_codes in codes {
        for &code in frame_codes {
            writer.write_i64::<LittleEndian>(code as i64)?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Save audio samples as binary f32
fn save_audio_binary(samples: &[f32], path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    for &sample in samples {
        writer.write_f32::<LittleEndian>(sample)?;
    }
    writer.flush()?;
    Ok(())
}

/// Compare Rust output with Python reference
fn compare_with_reference(
    reference_dir: &str,
    seed: u64,
    num_frames: usize,
    rust_codes: &[Vec<u32>],
    rust_audio: &[f32],
) -> Result<()> {
    let ref_dir = Path::new(reference_dir);

    // Load Python codes
    let py_codes_path = ref_dir.join(format!("codes_seed{}_frames{}.bin", seed, num_frames));
    if py_codes_path.exists() {
        let py_codes_data = fs::read(&py_codes_path)?;
        let py_codes: Vec<i64> = py_codes_data
            .chunks(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        // Flatten Rust codes for comparison
        let rust_codes_flat: Vec<i64> = rust_codes
            .iter()
            .flat_map(|frame| frame.iter().map(|&c| c as i64))
            .collect();

        // Compare codes
        let codes_match = py_codes.len() == rust_codes_flat.len()
            && py_codes
                .iter()
                .zip(rust_codes_flat.iter())
                .all(|(a, b)| a == b);

        if codes_match {
            println!("Codes: MATCH (all {} values identical)", py_codes.len());
        } else {
            println!("Codes: MISMATCH");
            println!("  Python: {} values", py_codes.len());
            println!("  Rust:   {} values", rust_codes_flat.len());

            // Show first differences
            let min_len = py_codes.len().min(rust_codes_flat.len());
            let mut diff_count = 0;
            for i in 0..min_len {
                if py_codes[i] != rust_codes_flat[i] {
                    if diff_count < 5 {
                        println!(
                            "  Index {}: Python={}, Rust={}",
                            i, py_codes[i], rust_codes_flat[i]
                        );
                    }
                    diff_count += 1;
                }
            }
            if diff_count > 5 {
                println!("  ... and {} more differences", diff_count - 5);
            }
            println!("  Total differences: {}", diff_count);
        }
    } else {
        println!("Codes: Python reference not found at {:?}", py_codes_path);
    }

    // Load Python audio
    let py_audio_path = ref_dir.join(format!("audio_seed{}_frames{}.bin", seed, num_frames));
    if py_audio_path.exists() {
        let py_audio_data = fs::read(&py_audio_path)?;
        let py_audio: Vec<f32> = py_audio_data
            .chunks(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        // Calculate audio difference statistics
        let min_len = py_audio.len().min(rust_audio.len());
        if min_len > 0 {
            let mut max_diff = 0.0f32;
            let mut sum_diff = 0.0f64;
            let mut sum_sq_diff = 0.0f64;

            for i in 0..min_len {
                let diff = (py_audio[i] - rust_audio[i]).abs();
                max_diff = max_diff.max(diff);
                sum_diff += diff as f64;
                sum_sq_diff += (diff * diff) as f64;
            }

            let mean_diff = sum_diff / min_len as f64;
            let rmse = (sum_sq_diff / min_len as f64).sqrt();

            println!("\nAudio comparison ({} samples):", min_len);
            println!("  Python samples: {}", py_audio.len());
            println!("  Rust samples:   {}", rust_audio.len());
            println!("  Max difference: {:.6}", max_diff);
            println!("  Mean difference: {:.6}", mean_diff);
            println!("  RMSE: {:.6}", rmse);

            // Check if audio is essentially identical
            if max_diff < 1e-5 {
                println!("  Status: MATCH (max diff < 1e-5)");
            } else if max_diff < 1e-3 {
                println!("  Status: CLOSE (max diff < 1e-3)");
            } else {
                println!("  Status: DIFFERENT");
            }
        }
    } else {
        println!("Audio: Python reference not found at {:?}", py_audio_path);
    }

    // Load and compare metadata
    let py_meta_path = ref_dir.join(format!("metadata_seed{}_frames{}.json", seed, num_frames));
    if py_meta_path.exists() {
        let py_meta: serde_json::Value = serde_json::from_reader(File::open(&py_meta_path)?)?;
        println!("\nPython metadata: {:?}", py_meta);
    }

    Ok(())
}
