"""Check tensor data types in GGUF files."""
import gguf
import numpy as np

for path in ['models/qwen3_tts_predictor.q8_0.gguf', 'models/qwen3_tts_talker.q5_k.gguf']:
    print(f"=== {path} ===")
    reader = gguf.GGUFReader(path)
    
    dtypes_seen = set()
    for t in reader.tensors:
        dtypes_seen.add(t.tensor_type)
        if len(dtypes_seen) <= 3:
            print(f"  {t.name}: shape={t.shape}, dtype={t.tensor_type}, data_dtype={t.data.dtype}, n_bytes={t.n_bytes}")
    
    print(f"  All dtypes: {dtypes_seen}")
    
    # Map tensor_type to GGMLQuantizationType
    for t in reader.tensors[:3]:
        raw = t.tensor_type
        if hasattr(raw, 'name'):
            ggml_type = getattr(gguf.GGMLQuantizationType, raw.name, None)
            print(f"  raw={raw}, raw.name={raw.name}, ggml_type={ggml_type}")
        print(f"  type(raw)={type(raw)}")
    print()
