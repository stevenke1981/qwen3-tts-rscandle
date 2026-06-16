//! Quantized TalkerModel for autoregressive semantic token generation
//!
//! Same architecture as `TalkerModel` but uses `candle_transformers::quantized_nn`
//! primitives (QMatMul-backed Linear, quantized RmsNorm, quantized Embedding)
//! and `candle_transformers::quantized_var_builder::VarBuilder` for GGUF inference.
//!
//! Forward pass logic is identical to the regular `talker.rs`.

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_transformers::quantized_nn::{self, Embedding, Linear, RmsNorm};
use candle_transformers::quantized_var_builder::VarBuilder;

use super::super::kv_cache::{AnyKVCache, KVCache, PreAllocKVCache};
pub use super::super::talker::{
    codec_tokens, special_tokens, tts_tokens, Language, Speaker, TalkerConfig,
};
use super::super::transformer::{MRoPE, RoPEType, RotaryEmbedding};
use super::transformer::QuantizedDecoderLayer;

/// Quantized text projection with SwiGLU activation.
///
/// Maps text embeddings (2048) to hidden dimension (1024).
/// Identical forward logic to `TextProjection` but backed by QTensor projections.
pub struct QuantizedTextProjection {
    fc1: Linear,
    fc2: Linear,
}

impl QuantizedTextProjection {
    /// Create from VarBuilder with config dimensions.
    pub fn new(config: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let fc1 = quantized_nn::linear(
            config.text_embed_dim,
            config.text_proj_intermediate,
            vb.pp("linear_fc1"),
        )?;
        let fc2 = quantized_nn::linear(
            config.text_proj_intermediate,
            config.hidden_size,
            vb.pp("linear_fc2"),
        )?;
        Ok(Self { fc1, fc2 })
    }

    /// Forward pass: fc1 -> silu -> fc2
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.fc1.forward(x)?;
        let hidden = candle_nn::ops::silu(&hidden)?;
        Ok(self.fc2.forward(&hidden)?)
    }
}

/// Quantized TalkerModel for autoregressive semantic token generation.
///
/// Same architecture as `TalkerModel` but uses:
/// - `quantized_nn::Embedding` instead of `candle_nn::Embedding`
/// - `QuantizedTextProjection` instead of `TextProjection`
/// - `QuantizedDecoderLayer` instead of `DecoderLayer`
/// - `quantized_nn::RmsNorm` instead of `candle_nn::RmsNorm`
/// - `quantized_nn::Linear` instead of `candle_nn::Linear`
pub struct QuantizedTalkerModel {
    /// Text embedding [text_vocab_size, text_embed_dim]
    text_embedding: Embedding,
    /// Text projection (2048 -> 1024)
    text_projection: QuantizedTextProjection,
    /// Codec embedding [codec_vocab_size, hidden_size]
    codec_embedding: Embedding,
    /// Transformer decoder layers
    layers: Vec<QuantizedDecoderLayer>,
    /// Final RMS norm
    norm: RmsNorm,
    /// Codec head (hidden_size -> codec_vocab_size, no bias)
    codec_head: Linear,
    /// Rotary position embedding (standard or MRoPE)
    rope: RoPEType,
    /// Configuration
    config: TalkerConfig,
    /// Device
    device: Device,
    /// Cached tts_pad projected embedding [1, 1, hidden_size]
    cached_tts_pad: Tensor,
    /// Cached tts_eos projected embedding [1, 1, hidden_size]
    cached_tts_eos: Tensor,
    /// Cached tts_bos projected embedding [1, 1, hidden_size]
    cached_tts_bos: Tensor,
}

impl QuantizedTalkerModel {
    /// Load model from a pre-built quantized VarBuilder scoped to `talker.`.
    ///
    /// The `VarBuilder` should be constructed from quantized (QTensor) weights
    /// via `candle_transformers::quantized_var_builder::VarBuilder::from_gguf()`
    /// and then *pp()'d to the `talker` prefix (the caller is responsible for
    /// the top-level scope — we do NOT prepend another `talker.`).
    pub fn from_gguf(vb: VarBuilder, config: TalkerConfig, device: &Device) -> Result<Self> {
        // Naming: the caller already scopes us to `talker.` so `vb` points at the
        // `talker` prefix level.  The GGUF file contains tensors named:
        //   talker.model.*           (text_embedding, codec_embedding, norm, layers)
        //   talker.codec_head
        //   talker.text_projection.*
        let model = vb.pp("model");
        let layer_config = config.to_layer_config();

        let text_embedding = Embedding::new(
            config.text_vocab_size,
            config.text_embed_dim,
            model.pp("text_embedding"),
        )?;
        let text_projection = QuantizedTextProjection::new(&config, vb.pp("text_projection"))?;
        let codec_embedding = Embedding::new(
            config.codec_vocab_size,
            config.hidden_size,
            model.pp("codec_embedding"),
        )?;
        let norm = RmsNorm::new(config.hidden_size, config.rms_norm_eps, model.pp("norm"))?;
        let codec_head = quantized_nn::linear_no_bias(
            config.hidden_size,
            config.codec_vocab_size,
            vb.pp("codec_head"),
        )?;

        let layers = (0..config.num_hidden_layers)
            .map(|i| QuantizedDecoderLayer::new(&layer_config, model.pp(format!("layers.{}", i))))
            .collect::<Result<Vec<_>>>()?;

        // RoPE - use MRoPE if mrope_section is configured
        let rope = if let Some(mrope_section) = config.mrope_section {
            RoPEType::Multimodal(MRoPE::new(
                config.head_dim,
                config.rope_theta,
                mrope_section,
                device,
            )?)
        } else {
            RoPEType::Standard(RotaryEmbedding::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                device,
            )?)
        };

        // Cache constant projected embeddings for generation
        let cached_tts_pad = {
            let id = Tensor::new(&[tts_tokens::TTS_PAD], device)?;
            let embed = text_embedding.forward(&id)?.unsqueeze(0)?;
            text_projection.forward(&embed)?
        };
        let cached_tts_eos = {
            let id = Tensor::new(&[tts_tokens::TTS_EOS], device)?;
            let embed = text_embedding.forward(&id)?.unsqueeze(0)?;
            text_projection.forward(&embed)?
        };
        let cached_tts_bos = {
            let id = Tensor::new(&[tts_tokens::TTS_BOS], device)?;
            let embed = text_embedding.forward(&id)?.unsqueeze(0)?;
            text_projection.forward(&embed)?
        };

        Ok(Self {
            text_embedding,
            text_projection,
            codec_embedding,
            layers,
            norm,
            codec_head,
            rope,
            config,
            device: device.clone(),
            cached_tts_pad,
            cached_tts_eos,
            cached_tts_bos,
        })
    }

    // ─── Prefill Methods ────────────────────────────────────────────────────

    /// Prefill for CustomVoice model with speaker and language.
    ///
    /// Constructs the full input sequence matching the Python implementation:
    /// - Positions 0-2: role prefix (text_proj of im_start, assistant, newline)
    /// - Positions 3-8: tts_pad/tts_bos ADDED with codec embeddings
    ///   - 3: tts_pad + codec_think
    ///   - 4: tts_pad + codec_think_bos
    ///   - 5: tts_pad + language_id
    ///   - 6: tts_pad + codec_think_eos
    ///   - 7: tts_pad + speaker
    ///   - 8: tts_bos + codec_pad
    /// - Position 9: first_text_proj + codec_bos
    ///
    /// Returns (hidden_states, logits) for generation.
    pub fn prefill_custom_voice(
        &self,
        text_tokens: &[u32],
        speaker: Speaker,
        language: Language,
        kv_caches: &mut [AnyKVCache],
    ) -> Result<(Tensor, Tensor)> {
        use codec_tokens::*;

        let role_prefix_hidden = self.build_role_prefix()?;

        // Codec: [think, think_bos, lang, think_eos, speaker, pad, bos]
        let codec_ids = Tensor::new(
            &[
                CODEC_THINK,
                CODEC_THINK_BOS,
                language.token_id(),
                CODEC_THINK_EOS,
                speaker.token_id(),
                CODEC_PAD,
                CODEC_BOS,
            ],
            &self.device,
        )?;
        let codec_embed = self.codec_embedding.forward(&codec_ids)?.unsqueeze(0)?;

        // 5 × tts_pad + 1 × tts_bos overlaid on first 6 codec tokens
        let tts_text_embed = self.build_tts_pad_bos(5)?;
        let codec_first6 = codec_embed.i((.., ..6, ..))?;
        let codec_hidden = tts_text_embed.add(&codec_first6)?;

        let mut hidden = Tensor::cat(&[&role_prefix_hidden, &codec_hidden], 1)?;

        // First text token + codec_bos
        let codec_bos_embed = codec_embed.i((.., 6..7, ..))?;
        if let Some(combined) = self.build_first_text_combined(text_tokens, &codec_bos_embed)? {
            hidden = Tensor::cat(&[&hidden, &combined], 1)?;
        }

        self.run_prefill_layers(hidden, kv_caches)
    }

    /// Prefill for voice cloning (x_vector_only mode).
    ///
    /// Same structure as `prefill_custom_voice` but replaces the discrete speaker
    /// token with a continuous speaker embedding from the ECAPA-TDNN encoder.
    ///
    /// The codec sequence becomes:
    /// `[think, think_bos, lang, think_eos, SPEAKER_EMBED, pad, bos]`
    ///
    /// When `icl_mode` is `true`, the final position (first_text + codec_bos) is
    /// omitted — that content moves into the ICL prompt instead, matching the
    /// Python reference implementation (9 positions vs 10).
    ///
    /// # Arguments
    /// * `text_tokens` — tokenized target text
    /// * `speaker_embed` — speaker embedding from the encoder, shape `[hidden_size]`
    /// * `language` — target language
    /// * `icl_mode` — if true, omit position 9 (first_text + codec_bos)
    /// * `kv_caches` — KV caches to populate
    pub fn prefill_voice_clone(
        &self,
        text_tokens: &[u32],
        speaker_embed: &Tensor,
        language: Language,
        icl_mode: bool,
        kv_caches: &mut [AnyKVCache],
    ) -> Result<(Tensor, Tensor)> {
        use codec_tokens::*;

        let role_prefix_hidden = self.build_role_prefix()?;

        // Codec: [think, think_bos, lang, think_eos] + speaker_embed + [pad, bos]
        let codec_prefix_ids = Tensor::new(
            &[
                CODEC_THINK,
                CODEC_THINK_BOS,
                language.token_id(),
                CODEC_THINK_EOS,
            ],
            &self.device,
        )?;
        let codec_prefix_embed = self
            .codec_embedding
            .forward(&codec_prefix_ids)?
            .unsqueeze(0)?;

        let speaker = speaker_embed.reshape((1, 1, self.config.hidden_size))?;

        let codec_suffix_ids = Tensor::new(&[CODEC_PAD, CODEC_BOS], &self.device)?;
        let codec_suffix_embed = self
            .codec_embedding
            .forward(&codec_suffix_ids)?
            .unsqueeze(0)?;

        let codec_embed = Tensor::cat(&[&codec_prefix_embed, &speaker, &codec_suffix_embed], 1)?;

        // 5 × tts_pad + 1 × tts_bos overlaid on first 6 codec tokens
        let tts_text_embed = self.build_tts_pad_bos(5)?;
        let codec_first6 = codec_embed.i((.., ..6, ..))?;
        let codec_hidden = tts_text_embed.add(&codec_first6)?;

        let mut hidden = Tensor::cat(&[&role_prefix_hidden, &codec_hidden], 1)?;

        // First text token + codec_bos (skipped in ICL mode)
        if !icl_mode {
            let codec_bos_embed = codec_embed.i((.., 6..7, ..))?;
            if let Some(combined) = self.build_first_text_combined(text_tokens, &codec_bos_embed)? {
                hidden = Tensor::cat(&[&hidden, &combined], 1)?;
            }
        }

        self.run_prefill_layers(hidden, kv_caches)
    }

    /// Prefill for VoiceDesign model with text-described voice conditioning.
    ///
    /// Constructs the full input sequence matching the Python implementation:
    /// - Instruct embedding: text_proj(instruct_tokens) — variable length (N positions)
    /// - Positions 0-2 (relative): role prefix (text_proj of im_start, assistant, newline)
    /// - Positions 3-7: tts_pad/tts_bos ADDED with codec embeddings (no speaker token)
    ///   - 3: tts_pad + codec_think
    ///   - 4: tts_pad + codec_think_bos
    ///   - 5: tts_pad + language_id
    ///   - 6: tts_pad + codec_think_eos
    ///   - 7: tts_bos + codec_pad
    /// - Position 8: first_text_proj + codec_bos
    ///
    /// Key differences from CustomVoice:
    /// - Instruct embedding prepended before role prefix
    /// - No speaker token -> codec prefix is 6 tokens not 7
    /// - TTS pad overlay is 4 copies not 5
    ///
    /// Returns (hidden_states, logits) for generation.
    pub fn prefill_voice_design(
        &self,
        text_tokens: &[u32],
        instruct_tokens: &[u32],
        language: Language,
        kv_caches: &mut [AnyKVCache],
    ) -> Result<(Tensor, Tensor)> {
        use codec_tokens::*;

        // Instruct text prefix
        let instruct_embed = self.get_projected_text_embeddings(instruct_tokens)?;

        let role_prefix_hidden = self.build_role_prefix()?;

        // Codec (no speaker): [think, think_bos, lang, think_eos, pad, bos]
        let codec_ids = Tensor::new(
            &[
                CODEC_THINK,
                CODEC_THINK_BOS,
                language.token_id(),
                CODEC_THINK_EOS,
                CODEC_PAD,
                CODEC_BOS,
            ],
            &self.device,
        )?;
        let codec_embed = self.codec_embedding.forward(&codec_ids)?.unsqueeze(0)?;

        // 4 × tts_pad + 1 × tts_bos overlaid on first 5 codec tokens
        let tts_text_embed = self.build_tts_pad_bos(4)?;
        let codec_first5 = codec_embed.i((.., ..5, ..))?;
        let codec_hidden = tts_text_embed.add(&codec_first5)?;

        let mut hidden = Tensor::cat(&[&instruct_embed, &role_prefix_hidden, &codec_hidden], 1)?;

        // First text token + codec_bos (index 5)
        let codec_bos_embed = codec_embed.i((.., 5..6, ..))?;
        if let Some(combined) = self.build_first_text_combined(text_tokens, &codec_bos_embed)? {
            hidden = Tensor::cat(&[&hidden, &combined], 1)?;
        }

        self.run_prefill_layers(hidden, kv_caches)
    }

    // ─── ICL Prompt ──────────────────────────────────────────────────────────

    /// Build ICL (in-context learning) prompt for voice cloning.
    ///
    /// Supports two modes controlled by `non_streaming`:
    ///
    /// **Streaming** (`non_streaming=false`, official Python default):
    /// Text and codec embeddings are aligned element-wise. If one is shorter,
    /// it is padded. Remaining text tokens become trailing context fed during
    /// generation.
    ///
    /// **Non-streaming** (`non_streaming=true`, used by mlx-audio):
    /// Text and codec are kept as separate sequential blocks:
    /// `[text + codec_pad_embed || codec + tts_pad_embed]`. All text is consumed
    /// in the prefix, so trailing is just `tts_pad`. This gives the model complete
    /// context before generation starts.
    ///
    /// # Returns
    /// `(icl_embed, trailing_text_embed)`
    pub fn build_icl_prompt(
        &self,
        target_text_ids: &[u32],
        ref_text_ids: &[u32],
        ref_codec_embeds: &Tensor, // [1, T_ref, hidden]
        non_streaming: bool,
    ) -> Result<(Tensor, Tensor)> {
        use codec_tokens::*;
        use tts_tokens::*;

        // --- 1. Text embeddings: [ref_text, target_text, tts_eos] projected ---
        let mut all_text_ids: Vec<u32> =
            Vec::with_capacity(ref_text_ids.len() + target_text_ids.len() + 1);
        all_text_ids.extend_from_slice(ref_text_ids);
        all_text_ids.extend_from_slice(target_text_ids);
        all_text_ids.push(TTS_EOS);

        let text_embed = self.get_projected_text_embeddings(&all_text_ids)?; // [1, N_text, hidden]
        let n_text = text_embed.dim(1)?;

        // --- 2. Codec embeddings: prepend codec_bos, then ref_codec_embeds ---
        let bos_id = Tensor::new(&[CODEC_BOS], &self.device)?;
        let bos_embed = self.codec_embedding.forward(&bos_id)?.unsqueeze(0)?; // [1, 1, hidden]
        let codec_embed = Tensor::cat(&[&bos_embed, ref_codec_embeds], 1)?; // [1, T_ref+1, hidden]
        let n_codec = codec_embed.dim(1)?;

        let tts_pad_embed = self.get_tts_pad_embed()?; // [1, 1, hidden]

        if non_streaming {
            // --- 3a. Non-streaming: sequential [text+codec_pad, codec+tts_pad] ---
            // Each text position gets codec_pad overlay
            let codec_pad_id = Tensor::new(&[CODEC_PAD], &self.device)?;
            let codec_pad_embed = self.codec_embedding.forward(&codec_pad_id)?.unsqueeze(0)?;
            let codec_pad_broadcast =
                codec_pad_embed.broadcast_as((1, n_text, self.config.hidden_size))?;
            let text_with_codec_pad = text_embed.add(&codec_pad_broadcast)?;

            // Each codec position gets tts_pad overlay
            let tts_pad_broadcast =
                tts_pad_embed.broadcast_as((1, n_codec, self.config.hidden_size))?;
            let codec_with_tts_pad = codec_embed.add(&tts_pad_broadcast)?;

            let icl_embed = Tensor::cat(&[&text_with_codec_pad, &codec_with_tts_pad], 1)?;
            Ok((icl_embed, tts_pad_embed))
        } else {
            // --- 3b. Streaming: element-wise overlay ---
            if n_text > n_codec {
                let text_head = text_embed.i((.., ..n_codec, ..))?;
                let icl_embed = text_head.add(&codec_embed)?;
                let trailing = text_embed.i((.., n_codec.., ..))?;
                Ok((icl_embed, trailing))
            } else {
                let pad_count = n_codec - n_text;
                let padded_text = if pad_count > 0 {
                    let pad_broadcast =
                        tts_pad_embed.broadcast_as((1, pad_count, self.config.hidden_size))?;
                    Tensor::cat(&[&text_embed, &pad_broadcast], 1)?
                } else {
                    text_embed
                };
                let icl_embed = padded_text.add(&codec_embed)?;
                Ok((icl_embed, tts_pad_embed))
            }
        }
    }

    // ─── Generation Step ─────────────────────────────────────────────────────

    /// Generate step with pre-built input embedding.
    ///
    /// This allows the caller to build the full input embedding externally
    /// (e.g., semantic_embed + acoustic_embeds + text_embed for CustomVoice).
    pub fn generate_step_with_embed(
        &self,
        input_embed: &Tensor,
        kv_caches: &mut [AnyKVCache],
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        // Single token attending to all previous positions via KV cache —
        // no masking needed (causal mask for seq_len=1 is all zeros).
        let mut hidden = input_embed.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &self.rope, None, Some(&mut kv_caches[i]), offset)?;
        }

        // Final norm
        hidden = self.norm.forward(&hidden)?;

        // Get logits
        let logits = self.codec_head.forward(&hidden)?;

        Ok((hidden, logits))
    }

    // ─── Private Helpers ─────────────────────────────────────────────────────

    /// Build the role prefix embeddings: text_proj([im_start, assistant, newline]).
    ///
    /// Returns a `[1, 3, hidden_size]` tensor used at the start of every prefill variant.
    fn build_role_prefix(&self) -> Result<Tensor> {
        use special_tokens::*;
        let role_prefix_ids = Tensor::new(&[IM_START, ASSISTANT, NEWLINE], &self.device)?;
        let role_prefix_embed = self.text_embedding.forward(&role_prefix_ids)?;
        let role_prefix_embed = role_prefix_embed.unsqueeze(0)?;
        self.text_projection.forward(&role_prefix_embed)
    }

    /// Build tts_pad (projected, count copies) and tts_bos (projected, 1 copy).
    ///
    /// Returns a `[1, pad_count + 1, hidden_size]` tensor of
    /// `[tts_pad × pad_count, tts_bos × 1]`. Uses cached embeddings.
    fn build_tts_pad_bos(&self, pad_count: usize) -> Result<Tensor> {
        let tts_pad_expanded =
            self.cached_tts_pad
                .broadcast_as((1, pad_count, self.config.hidden_size))?;
        Ok(Tensor::cat(&[&tts_pad_expanded, &self.cached_tts_bos], 1)?)
    }

    /// Build first text token combined with codec_bos embedding.
    ///
    /// Returns `Some([1, 1, hidden_size])` if text_tokens is non-empty, `None` otherwise.
    fn build_first_text_combined(
        &self,
        text_tokens: &[u32],
        codec_bos_embed: &Tensor,
    ) -> Result<Option<Tensor>> {
        if text_tokens.is_empty() {
            return Ok(None);
        }
        let first_text_id = Tensor::new(&[text_tokens[0]], &self.device)?;
        let first_text_embed = self.text_embedding.forward(&first_text_id)?.unsqueeze(0)?;
        let first_text_proj = self.text_projection.forward(&first_text_embed)?;
        Ok(Some(first_text_proj.add(codec_bos_embed)?))
    }

    // ─── Raw Forward / Prefill ───────────────────────────────────────────────

    /// Raw forward pass: embed input_ids and run through all layers.
    ///
    /// Returns logits for the full sequence (no KV cache).
    /// This is a low-level method for reference validation; prefer the
    /// mode-specific prefill methods for actual generation.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embed = self.text_embedding.forward(input_ids)?;
        let projected = self.text_projection.forward(&embed)?;

        let seq_len = projected.dim(1)?;
        let mask = self.create_causal_mask(seq_len, 0)?;

        let mut hidden = projected;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &self.rope, Some(&mask), None, 0)?;
        }
        hidden = self.norm.forward(&hidden)?;
        Ok(self.codec_head.forward(&hidden)?)
    }

    /// Raw prefill: embed input_ids, run through layers, populate KV caches.
    ///
    /// Returns `(hidden_states, logits)` where logits are for the last position only.
    /// This is a low-level method for reference validation; prefer the
    /// mode-specific prefill methods for actual generation.
    pub fn prefill(
        &self,
        input_ids: &Tensor,
        kv_caches: &mut [AnyKVCache],
    ) -> Result<(Tensor, Tensor)> {
        let embed = self.text_embedding.forward(input_ids)?;
        let projected = self.text_projection.forward(&embed)?;
        self.run_prefill_layers(projected, kv_caches)
    }

    /// Run prefill through all layers: causal mask -> layers -> norm -> logits.
    ///
    /// Returns `(hidden_states, logits)` for the full sequence.
    fn run_prefill_layers(
        &self,
        mut hidden: Tensor,
        kv_caches: &mut [AnyKVCache],
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = hidden.dim(1)?;
        let mask = self.create_causal_mask(seq_len, 0)?;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &self.rope, Some(&mask), Some(&mut kv_caches[i]), 0)?;
        }

        hidden = self.norm.forward(&hidden)?;

        let last_hidden = hidden.i((.., seq_len - 1..seq_len, ..))?;
        let logits = self.codec_head.forward(&last_hidden)?;

        Ok((hidden, logits))
    }

    // ─── Embedding Helpers ───────────────────────────────────────────────────

    /// Get tts_pad text embedding (projected) — cached.
    ///
    /// This is added to codec embeddings during generation after trailing text is exhausted.
    pub fn get_tts_pad_embed(&self) -> Result<Tensor> {
        Ok(self.cached_tts_pad.clone())
    }

    /// Get tts_eos text embedding (projected) — cached.
    ///
    /// This marks the end of text input.
    pub fn get_tts_eos_embed(&self) -> Result<Tensor> {
        Ok(self.cached_tts_eos.clone())
    }

    /// Get projected text embeddings for a sequence of token IDs.
    ///
    /// Returns [1, seq_len, hidden_size] tensor of projected text embeddings.
    pub fn get_projected_text_embeddings(&self, token_ids: &[u32]) -> Result<Tensor> {
        if token_ids.is_empty() {
            // Return empty tensor with correct shape; quantized models output F32.
            return Ok(Tensor::zeros(
                (1, 0, self.config.hidden_size),
                DType::F32,
                &self.device,
            )?);
        }

        let ids_tensor = Tensor::new(token_ids, &self.device)?;
        let embeds = self.text_embedding.forward(&ids_tensor)?;
        let embeds = embeds.unsqueeze(0)?; // [1, seq_len, text_embed_dim]
        self.text_projection.forward(&embeds)
    }

    fn create_causal_mask(&self, seq_len: usize, offset: usize) -> Result<Tensor> {
        super::super::transformer::create_causal_mask(seq_len, offset, &self.device)
    }

    // ─── KV Cache Management ─────────────────────────────────────────────────

    /// Create new KV caches for generation.
    pub fn new_kv_caches(&self, max_seq: usize) -> Vec<AnyKVCache> {
        if (self.device.is_cuda() || self.device.is_metal()) && max_seq > 0 {
            // Quantized model outputs are always F32; use F32 for cache dtype.
            let dtype = DType::F32;
            (0..self.config.num_hidden_layers)
                .map(|_| {
                    PreAllocKVCache::new(
                        1, // batch
                        self.config.num_key_value_heads,
                        max_seq,
                        self.config.head_dim,
                        dtype,
                        &self.device,
                    )
                    .map(AnyKVCache::PreAlloc)
                    .unwrap_or_else(|_| AnyKVCache::Concat(KVCache::new()))
                })
                .collect()
        } else {
            (0..self.config.num_hidden_layers)
                .map(|_| AnyKVCache::Concat(KVCache::new()))
                .collect()
        }
    }

    // ─── Public Accessors ────────────────────────────────────────────────────

    /// Get codec embedding for a token (used by code predictor).
    pub fn get_codec_embedding(&self, token_id: u32) -> Result<Tensor> {
        let token_tensor = Tensor::new(&[token_id], &self.device)?;
        let embed = self.codec_embedding.forward(&token_tensor)?;
        Ok(embed.unsqueeze(0)?) // [1, 1, hidden_size]
    }

    /// Look up codec embedding from a GPU-resident token tensor.
    ///
    /// Avoids the CPU->GPU roundtrip of creating a new tensor from a u32.
    /// `token` should be a scalar or 1-element tensor of token indices.
    pub fn get_codec_embedding_from_tensor(&self, token: &Tensor) -> Result<Tensor> {
        let token = token.flatten_all()?;
        let embed = self.codec_embedding.forward(&token)?;
        Ok(embed.unsqueeze(0)?) // [1, 1, hidden_size]
    }

    /// Get config.
    pub fn config(&self) -> &TalkerConfig {
        &self.config
    }

    /// Get an iterator over transformer layers (for running ICL extension passes).
    pub fn layers_iter(&self) -> impl Iterator<Item = &QuantizedDecoderLayer> {
        self.layers.iter()
    }

    /// Get a reference to the rotary position embedding.
    pub fn rope(&self) -> &RoPEType {
        &self.rope
    }

    /// Apply final RMS norm.
    pub fn apply_norm(&self, hidden: &Tensor) -> Result<Tensor> {
        Ok(self.norm.forward(hidden)?)
    }

    /// Apply codec head (hidden -> codec logits).
    pub fn apply_codec_head(&self, hidden: &Tensor) -> Result<Tensor> {
        Ok(self.codec_head.forward(hidden)?)
    }

    /// Embed a batch of codec tokens.
    ///
    /// # Arguments
    /// * `token_ids` — 1-D tensor of codec token IDs, shape `[T]`
    ///
    /// # Returns
    /// Tensor of shape `[1, T, hidden_size]`
    pub fn get_codec_embedding_batch(&self, token_ids: &Tensor) -> Result<Tensor> {
        let embed = self.codec_embedding.forward(token_ids)?; // [T, hidden_size]
        Ok(embed.unsqueeze(0)?) // [1, T, hidden_size]
    }
}
