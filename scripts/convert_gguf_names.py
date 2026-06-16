"""
Convert GGUF tensor names from llama.cpp naming to Qwen3-TTS Rust project naming.

Usage:
    python scripts/convert_gguf_names.py models/qwen3_tts_talker.q5_k.gguf

What it does:
    Reads a GGUF file, remaps tensor names from llama.cpp format (blk.*, token_embd.*, etc.)
    to the format expected by Qwen3-TTS Rust QuantizedTalkerModel/QuantizedCodePredictor:
    - talker.model.* for transformer layers
    - talker.* for other components
"""

import os, sys, struct
import numpy as np

try:
    from gguf import GGUFWriter, GGUFReader, GGUFValueType, GGMLQuantizationType
except ImportError:
    print("gguf package not found. Install with: pip install gguf")
    sys.exit(1)


# ── Tensor name mapping: llama.cpp → Qwen3-TTS ──────────────────────────

# Root-level tensors (talker)
TALKER_MAP = {
    # In the llama.cpp GGUF, token_embd.weight is the codec embedding [2048, 3072]
    # (codec_vocab=3072), NOT the text embedding (text_vocab=151936).
    # Text embedding must be loaded separately from safetensors.
    'token_embd.weight':    'talker.model.codec_embedding.weight',
    'output_norm.weight':   'talker.model.norm.weight',
    'output.weight':        'talker.codec_head.weight',
}

# Per-layer mapping for talker (28 layers, 2048 hidden)
TALKER_LAYER_MAP = {
    'attn_norm.weight':     'model.layers.{}.input_layernorm.weight',
    'attn_q.weight':        'model.layers.{}.self_attn.q_proj.weight',
    'attn_k.weight':        'model.layers.{}.self_attn.k_proj.weight',
    'attn_v.weight':        'model.layers.{}.self_attn.v_proj.weight',
    'attn_output.weight':   'model.layers.{}.self_attn.o_proj.weight',
    'attn_q_norm.weight':   'model.layers.{}.self_attn.q_norm.weight',
    'attn_k_norm.weight':   'model.layers.{}.self_attn.k_norm.weight',
    'ffn_norm.weight':      'model.layers.{}.post_attention_layernorm.weight',
    'ffn_gate.weight':      'model.layers.{}.mlp.gate_proj.weight',
    'ffn_up.weight':        'model.layers.{}.mlp.up_proj.weight',
    'ffn_down.weight':      'model.layers.{}.mlp.down_proj.weight',
}

# Same per-layer mapping for predictor (5 layers, 1024 hidden)
# The predictor uses `talker.code_predictor.*` prefix
PREDICTOR_MAP = {
    'token_embd.weight':    'talker.code_predictor.codec_embeddings.0.weight',
    'output_norm.weight':   'talker.code_predictor.model.norm.weight',
    'output.weight':        'talker.code_predictor.lm_head.0.weight',
}


def remap_tensor(old_name: str) -> str | None:
    """Remap a single tensor name from llama.cpp → Qwen3-TTS format."""
    if old_name in TALKER_MAP:
        return TALKER_MAP[old_name]
    if old_name in PREDICTOR_MAP:
        return PREDICTOR_MAP[old_name]
    
    # Per-layer tensors: blk.N.suffix
    if old_name.startswith('blk.'):
        parts = old_name.split('.')
        if len(parts) >= 3:
            try:
                layer_idx = int(parts[1])
                suffix = '.'.join(parts[2:])
            except ValueError:
                return None
            
            # Try talker layer mapping (talker.model.layers.{}.suffix)
            if suffix in TALKER_LAYER_MAP:
                return 'talker.' + TALKER_LAYER_MAP[suffix].format(layer_idx)
            
            return None
    
    return None


def get_field_value(field):
    """Get the Python value from a GGUF ReaderField."""
    parts = field.parts
    if not parts:
        return None
    
    # Determine value type from the last element
    val = parts[-1]
    
    # Check type
    t = field.types[-1]
    
    if t == GGUFValueType.STRING:
        return bytes(val).decode('utf-8') if isinstance(val, np.ndarray) else val.decode('utf-8')
    elif t == GGUFValueType.ARRAY:
        # Array: return as list of decoded values
        inner_type = field.types[0] if len(field.types) > 1 else None
        items = []
        for item in parts:
            if isinstance(item, np.ndarray):
                if np.issubdtype(item.dtype, np.integer):
                    items.append(int(item))
                elif np.issubdtype(item.dtype, np.floating):
                    items.append(float(item))
                else:
                    items.append(str(item))
            elif isinstance(item, bytes):
                items.append(item.decode('utf-8', errors='replace'))
        return items
    elif t == GGUFValueType.FLOAT32:
        return float(val)
    elif t == GGUFValueType.FLOAT64:
        return float(val)
    elif t == GGUFValueType.UINT8:
        return int(val)
    elif t == GGUFValueType.INT8:
        return int(val) if val.shape else int(val)
    elif t == GGUFValueType.UINT16:
        return int(val)
    elif t == GGUFValueType.INT16:
        return int(val)
    elif t == GGUFValueType.UINT32:
        return int(val)
    elif t == GGUFValueType.INT32:
        return int(val)
    elif t == GGUFValueType.UINT64:
        return int(val)
    elif t == GGUFValueType.INT64:
        return int(val)
    elif t == GGUFValueType.BOOL:
        return bool(val)
    else:
        return str(val)


def write_metadata(writer, reader):
    """Copy metadata from reader to writer."""
    meta = {}
    for key, field in reader.fields.items():
        # Skip internal GGUF header fields (handled by writer automatically)
        if key.startswith('GGUF.'):
            continue
        val = None
        parts = field.parts
        if parts:
            t = field.types[-1]
            if t == GGUFValueType.STRING:
                raw = parts[-1]
                if isinstance(raw, np.ndarray):
                    val = bytes(raw).decode('utf-8', errors='replace')
                elif isinstance(raw, bytes):
                    val = raw.decode('utf-8', errors='replace')
                else:
                    val = str(raw)
            elif t == GGUFValueType.ARRAY:
                items = []
                # Check inner type
                for p in parts:
                    if isinstance(p, np.ndarray):
                        if np.issubdtype(p.dtype, np.integer):
                            items.append(int(p.item()) if p.ndim == 0 else int(p))
                        elif np.issubdtype(p.dtype, np.floating):
                            items.append(float(p.item()) if p.ndim == 0 else float(p))
                        else:
                            pass
                    elif isinstance(p, bytes):
                        items.append(p.decode('utf-8', errors='replace'))
                val = items if items else None
            elif t in (GGUFValueType.FLOAT32, GGUFValueType.FLOAT64):
                raw = parts[-1]
                val = float(raw.item()) if hasattr(raw, 'item') else float(raw)
            elif t in (GGUFValueType.UINT8, GGUFValueType.INT8,
                       GGUFValueType.UINT16, GGUFValueType.INT16,
                       GGUFValueType.UINT32, GGUFValueType.INT32,
                       GGUFValueType.UINT64, GGUFValueType.INT64):
                raw = parts[-1]
                val = int(raw.item()) if hasattr(raw, 'item') else int(raw)
            elif t == GGUFValueType.BOOL:
                raw = parts[-1]
                val = bool(raw.item()) if hasattr(raw, 'item') else bool(raw)
            else:
                val = str(parts[-1])
        meta[key] = val
    
    # Skip keys already handled by GGUFWriter constructor
    skip_keys = {'general.architecture', 'general.name'}
    
    # Now write metadata using the writer's methods
    for key, val in meta.items():
        if val is None:
            continue
        if key in skip_keys:
            continue
        
        # Get the field type
        field = reader.fields.get(key)
        if not field:
            continue
        t = field.types[-1]
        
        try:
            if t == GGUFValueType.STRING:
                writer.add_string(key, val)
            elif t == GGUFValueType.ARRAY and isinstance(val, list):
                # Determine element type for arrays
                if val and isinstance(val[0], int):
                    writer.add_array(key, val)
                elif val and isinstance(val[0], str):
                    writer.add_array(key, val)
                elif val and isinstance(val[0], float):
                    writer.add_array(key, val)
                else:
                    writer.add_array(key, [str(v) for v in val] if val else val)
            elif t in (GGUFValueType.FLOAT32, GGUFValueType.FLOAT64):
                writer.add_float32(key, val)
            elif t == GGUFValueType.UINT32:
                writer.add_uint32(key, val)
            elif t == GGUFValueType.UINT64:
                writer.add_uint64(key, val)
            elif t == GGUFValueType.BOOL:
                writer.add_bool(key, val)
            elif t in (GGUFValueType.INT32, GGUFValueType.INT64):
                writer.add_int64(key, val)
            else:
                writer.add_string(key, str(val))
        except Exception as e:
            print(f"    Warning: metadata '{key}': {e}")


def convert_gguf(input_path: str, output_path: str | None = None):
    """Convert GGUF tensor names from llama.cpp → Qwen3-TTS format."""
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_converted{ext}"
    
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()
    
    # Read original GGUF
    reader = GGUFReader(input_path)
    
    tensor_count = len(reader.tensors)
    
    # Get block count
    block_count_field = reader.fields.get('qwen3.block_count')
    block_count = int(block_count_field.parts[-1].item()) if block_count_field else 0
    
    print(f"Total tensors: {tensor_count}")
    print(f"Block count: {block_count}")
    
    is_talker = block_count >= 28
    print(f"Mode: {'Talker (28+ layers)' if is_talker else 'Predictor (5 layers)'}")
    print()
    
    # Prepare output writer
    writer = GGUFWriter(output_path, "qwen3")
    
    # Copy metadata
    print("Copying metadata...")
    write_metadata(writer, reader)
    
    # Update the name to indicate conversion
    writer.add_string('general.name', 'Qwen3-TTS-1.7B-Talker-Converted')
    
    # Rename and copy tensors
    renamed = 0
    errors = 0
    
    for tensor in reader.tensors:
        old_name = tensor.name
        data = tensor.data
        
        new_name = remap_tensor(old_name)
        if new_name is None:
            print(f"  WARNING: No mapping for '{old_name}', dropping")
            errors += 1
            continue
        
        # Get quantization type
        raw_dtype = tensor.tensor_type
        
        # For quantized types, data is raw uint8 bytes
        # For F32, data is float32
        if isinstance(data, memoryview):
            data_np = np.asarray(data)
        else:
            data_np = data
        
        # Pass raw_dtype for quantized types so GGUFWriter preserves the format
        # F32 tensors (raw_dtype=0) are handled automatically
        if raw_dtype == GGMLQuantizationType.F32:
            writer.add_tensor(new_name, data_np)
        else:
            writer.add_tensor(new_name, data_np, raw_dtype=raw_dtype)
        
        renamed += 1
        
        if renamed <= 10 or renamed == tensor_count or renamed % 100 == 0:
            print(f"  [{renamed}/{tensor_count}] {new_name}")
    
    print()
    print(f"Renamed: {renamed} tensors")
    print(f"Dropped: {errors} tensors (no mapping found)")
    
    # Write output
    print(f"\nWriting {output_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    if os.path.exists(output_path):
        output_size = os.path.getsize(output_path)
        print(f"Done! Output size: {output_size / 1024 / 1024:.2f} MB")
    else:
        print("ERROR: Output file was not created!")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.gguf> [output.gguf]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    convert_gguf(input_path, output_path)
