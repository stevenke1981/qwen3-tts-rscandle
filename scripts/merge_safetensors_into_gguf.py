"""
Merge TTS-specific tensors from Qwen safetensors into converted GGUF.

The llama.cpp GGUF conversion only has the base LLM architecture (Qwen3)
but is missing TTS-specific weights needed by QuantizedTalkerModel:
  - talker.text_projection.linear_fc1.*
  - talker.text_projection.linear_fc2.*
  - talker.model.codec_embedding.weight

Usage:
    python scripts/merge_safetensors_into_gguf.py models/converted.gguf models/merged.gguf

Requires model.safetensors from Qwen/Qwen3-TTS-12Hz-1.7B-Base cached locally.
"""

import os, sys, json
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from gguf import GGUFWriter, GGUFReader, GGMLQuantizationType

# Which keys to extract from safetensors
# Note: codec_embedding is already in GGUF from token_embd.weight mapping
# Text embedding needs to come from safetensors (F16 to save space)
NEEDED_KEYS = {
    "talker.model.text_embedding.weight",   # [151936, 2048] → store as F16
    "talker.text_projection.linear_fc1.weight",
    "talker.text_projection.linear_fc1.bias",
    "talker.text_projection.linear_fc2.weight",
    "talker.text_projection.linear_fc2.bias",
}

def load_needed_tensors(safetensors_path: str) -> dict:
    """Load only needed tensors from safetensors.
    
    - text_embedding: bf16 → F16 to save space (~300 MB vs 600 MB)
    - text_projection: bf16 → F32
    """
    tensors = {}
    with safe_open(safetensors_path, framework="pt") as f:
        for key in NEEDED_KEYS:
            data = f.get_tensor(key)  # torch.Tensor, dtype=torch.bfloat16
            if "text_embedding" in key:
                # Huge tensor (1.2 GB as F32, 600 MB as BF16, ~300 MB as F16)
                tensors[key] = data.half().cpu().numpy()  # bf16 → float16
            else:
                # Small projection weights: F32 is fine
                tensors[key] = data.float().cpu().numpy()
        print(f"Loaded {len(tensors)} tensors from safetensors")
        for k, v in tensors.items():
            mb = v.nbytes / 1024 / 1024
            print(f"  {k}: shape={list(v.shape)} dtype={v.dtype} ({mb:.1f} MB)")
    return tensors

def merge_into_gguf(input_gguf: str, output_gguf: str, extra_tensors: dict):
    """Read input GGUF, create new one with all original + extra tensors."""
    reader = GGUFReader(input_gguf)
    tensor_count = len(reader.tensors)
    print(f"\nInput GGUF: {tensor_count} tensors, {os.path.getsize(input_gguf)/1024/1024:.1f} MB")

    # Use the same architecture name as input
    arch_field = reader.fields.get("general.architecture")
    if arch_field:
        arch = str(arch_field.parts[-1])
    else:
        arch = "qwen3"
    print(f"Architecture: {arch}")

    writer = GGUFWriter(output_gguf, arch)

    # 1. Copy metadata (skip internal + arch/handled-by-constructor keys)
    meta = {}
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key in ("general.architecture", "general.name"):
            continue
        val = None
        parts = field.parts
        if not parts:
            continue
        t = field.types[-1]
        try:
            if t == 6:  # STRING
                raw = parts[-1]
                if isinstance(raw, np.ndarray):
                    val = bytes(raw).decode("utf-8", errors="replace")
                elif isinstance(raw, bytes):
                    val = raw.decode("utf-8", errors="replace")
                else:
                    val = str(raw)
            elif t == 7:  # ARRAY
                items = []
                for p in parts:
                    if isinstance(p, np.ndarray):
                        if np.issubdtype(p.dtype, np.integer):
                            items.append(int(p.item()) if p.ndim == 0 else int(p))
                        elif np.issubdtype(p.dtype, np.floating):
                            items.append(float(p.item()) if p.ndim == 0 else float(p))
                    elif isinstance(p, bytes):
                        items.append(p.decode("utf-8", errors="replace"))
                val = items or None
            elif t in (0, 11):  # FLOAT32, FLOAT64
                raw = parts[-1]
                val = float(raw.item()) if hasattr(raw, "item") else float(raw)
            elif t in (1, 2, 3, 4, 5, 12):  # INT types + BOOL(12)
                raw = parts[-1]
                val = int(raw.item()) if hasattr(raw, "item") else int(raw)
            else:
                val = str(parts[-1])
        except:
            pass
        meta[key] = val

    # Write metadata
    for key, val in meta.items():
        if val is None:
            continue
        field = reader.fields.get(key)
        if not field:
            continue
        t = field.types[-1]
        try:
            if t == 6:
                writer.add_string(key, val)
            elif t == 7 and isinstance(val, list):
                writer.add_array(key, val)
            elif t in (0, 11):
                writer.add_float32(key, val)
            elif t == 3:
                writer.add_uint32(key, val)
            elif t == 5:
                writer.add_uint64(key, val)
            elif t == 12:
                writer.add_bool(key, val)
            elif t in (1, 2, 4):
                writer.add_int64(key, val)
        except:
            pass

    # 2. Copy all original tensors (preserve quantization)
    copied = 0
    for tensor in reader.tensors:
        old_name = tensor.name
        data = tensor.data
        raw_dtype = tensor.tensor_type
        if isinstance(data, memoryview):
            data_np = np.asarray(data)
        else:
            data_np = data
        if raw_dtype == GGMLQuantizationType.F32:
            writer.add_tensor(old_name, data_np)
        else:
            writer.add_tensor(old_name, data_np, raw_dtype=raw_dtype)
        copied += 1

    print(f"Copied {copied} original tensors")

    # 3. Add extra tensors from safetensors
    added = 0
    for key, data in extra_tensors.items():
        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)
        
        # text_embedding stored as F16 to save ~300 MB; others as F32
        if "text_embedding" in key:
            writer.add_tensor(key, data, raw_dtype=GGMLQuantizationType.F16)
        else:
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            writer.add_tensor(key, data)
        added += 1
        print(f"  Added: {key} shape={list(data.shape)} dtype={data.dtype} ({data.nbytes/1024/1024:.1f} MB)")

    # 4. Write file
    print(f"\nWriting {output_gguf}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = os.path.getsize(output_gguf)
    print(f"Done! {out_size/1024/1024:.1f} MB ({copied + added} tensors)")

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.gguf> <output.gguf>")
        sys.exit(1)

    input_gguf = sys.argv[1]
    output_gguf = sys.argv[2]

    if not os.path.exists(input_gguf):
        print(f"ERROR: Input GGUF not found: {input_gguf}")
        sys.exit(1)

    # Get safetensors path from HF cache
    print("Locating model.safetensors from HF cache...")
    safetensors_path = hf_hub_download(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "model.safetensors"
    )
    print(f"Safetensors: {safetensors_path} ({os.path.getsize(safetensors_path)/1024/1024:.1f} MB)")

    # Load needed tensors
    extra_tensors = load_needed_tensors(safetensors_path)

    # Merge
    merge_into_gguf(input_gguf, output_gguf, extra_tensors)

if __name__ == "__main__":
    main()
