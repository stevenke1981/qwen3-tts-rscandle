"""Explore GGUFReader and GGUFWriter APIs."""
import gguf

r = gguf.GGUFReader('models/qwen3_tts_talker.q5_k.gguf')

print("GGUFReader fields:")
for name, field in r.fields.items():
    print(f"  {name}:")
    print(f"    types: {field.types}")
    val = field.parts
    if val and len(val) > 0:
        v = val[-1]
        if isinstance(v, bytes):
            try:
                print(f"    value: {v.decode('utf-8')[:200]}")
            except:
                print(f"    value: {v[:200]}")
        else:
            arr = v.tolist()
            print(f"    value: {str(arr)[:200]}")
    print()

# Check GGUFWriter constructor signature
import inspect
print("\nGGUFWriter constructor:")
sig = inspect.signature(gguf.GGUFWriter.__init__)
print(f"  {sig}")

# Check key methods
print("\nGGUFWriter public methods:")
for name in dir(gguf.GGUFWriter):
    if not name.startswith('_') and 'add' in name:
        print(f"  {name}")
