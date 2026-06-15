//! Quantized transformer building blocks for Qwen3-TTS
//!
//! Contains quantized versions (`QTensor`-backed) of `Attention`, `MLP`,
//! and `DecoderLayer` — used by quantized `TalkerModel` and `CodePredictor`.
//!
//! Forward pass logic is identical to the regular `transformer.rs` but uses
//! `candle_transformers::quantized_var_builder::VarBuilder` / `quantized_nn`
//! instead of `candle_nn::VarBuilder` / `candle_nn`.

use anyhow::Result;
use candle_core::{Module, Tensor, D};
use candle_transformers::quantized_nn::{self, Linear, RmsNorm};
use candle_transformers::quantized_var_builder::VarBuilder;

use super::super::config::Qwen3TTSConfig;
use super::super::transformer::{AnyKVCache, RoPEType};

/// Quantized multi-head attention with grouped-query attention and QK normalization.
///
/// Identical forward logic to `Attention` but backed by `QTensor` projections
/// and `quantized_nn::RmsNorm`.
pub struct QuantizedAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl QuantizedAttention {
    pub fn new(config: &Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim();

        let q_proj =
            quantized_nn::linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj =
            quantized_nn::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj =
            quantized_nn::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj =
            quantized_nn::linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        // QK normalization: RMSNorm applied per-head after projection
        let q_norm = RmsNorm::new(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        rope: &RoPEType,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut AnyKVCache>,
        offset: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden_states.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape to [batch, seq, heads, head_dim] for QK norm
        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;

        // Apply QK normalization (per-head RMSNorm)
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Apply rotary embeddings
        let (q, k) = rope.apply(&q, &k, offset)?;

        // Update KV cache
        let (k, v) = if let Some(cache) = kv_cache {
            cache.update(&k, &v)?
        } else {
            (k, v)
        };

        // Manual scaled dot-product attention (no flash-attn, no Metal SDPA)
        let k = self.repeat_kv(&k)?;
        let v = self.repeat_kv(&v)?;
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let attn_weights =
            (q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)? * self.scale)?;
        let attn_weights = if let Some(mask) = attention_mask {
            let mask = mask.to_dtype(attn_weights.dtype())?;
            attn_weights.broadcast_add(&mask)?
        } else {
            attn_weights
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        Ok(self.o_proj.forward(&attn_output)?)
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            return Ok(x.clone());
        }

        let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;
        let x = x
            .unsqueeze(2)?
            .expand((batch, num_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((batch, num_kv_heads * n_rep, seq_len, head_dim))?;
        Ok(x)
    }
}

/// Quantized MLP block with SwiGLU activation.
///
/// Identical forward logic to `MLP` but backed by `QTensor` projections.
pub struct QuantizedMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl QuantizedMLP {
    pub fn new(config: &Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        Ok(Self {
            gate_proj: quantized_nn::linear_no_bias(
                hidden_size,
                intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: quantized_nn::linear_no_bias(
                hidden_size,
                intermediate_size,
                vb.pp("up_proj"),
            )?,
            down_proj: quantized_nn::linear_no_bias(
                intermediate_size,
                hidden_size,
                vb.pp("down_proj"),
            )?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        Ok(self.down_proj.forward(&(gate * up)?)?)
    }
}

/// Quantized transformer decoder layer.
///
/// Identical forward logic to `DecoderLayer` but uses `QuantizedAttention`,
/// `QuantizedMLP`, and regular `quantized_nn::RmsNorm` (no `FusedRmsNorm`).
pub struct QuantizedDecoderLayer {
    self_attn: QuantizedAttention,
    mlp: QuantizedMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl QuantizedDecoderLayer {
    pub fn new(config: &Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: QuantizedAttention::new(config, vb.pp("self_attn"))?,
            mlp: QuantizedMLP::new(config, vb.pp("mlp"))?,
            input_layernorm: RmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        rope: &RoPEType,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut AnyKVCache>,
        offset: usize,
    ) -> Result<Tensor> {
        // Self-attention with residual
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states =
            self.self_attn
                .forward(&hidden_states, rope, attention_mask, kv_cache, offset)?;

        // Post-attention layernorm (sequential, no fused CUDA kernel)
        let hidden_states = (hidden_states + residual)?;
        let normed = self.post_attention_layernorm.forward(&hidden_states)?;

        // MLP with residual
        let mlp_out = self.mlp.forward(&normed)?;
        let hidden_states = (hidden_states + mlp_out)?;

        Ok(hidden_states)
    }
}
