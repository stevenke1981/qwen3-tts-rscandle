//! Quantized (GGUF / QMatMul) model variants for CPU inference.
//!
//! These modules mirror the regular `src/models/` architecture but use
//! `candle_transformers::quantized_nn` primitives backed by `QTensor`
//! (quantized matrix multiplication via GGUF weights).
//!
//! Forward pass logic is identical to the regular variants — only weight
//! loading and matmul execution differs.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use candle_transformers::quantized_var_builder::VarBuilder;
//!
//! let vb = VarBuilder::from_gguff("model.gguf", &device)?;
//! let talker = QuantizedTalkerModel::from_gguf(vb.pp("talker"), config_talker, &device)?;
//! let cp = QuantizedCodePredictor::new(config_cp, vb.pp("talker.code_predictor"))?;
//! ```

mod code_predictor;
mod talker;
mod transformer;

pub use code_predictor::QuantizedCodePredictor;
pub use talker::{QuantizedTalkerModel, QuantizedTextProjection};
pub use transformer::{QuantizedAttention, QuantizedDecoderLayer, QuantizedMLP};
