"""Examine GGUFWriter tensor writing for quantized types."""
import gguf
import inspect

# Check add_tensor signature
sig = inspect.signature(gguf.GGUFWriter.add_tensor)
print("add_tensor signature:", sig)
print()

# Check GGUFWriter internal methods
print("GGUFWriter methods for tensor handling:")
for name in sorted(dir(gguf.GGUFWriter)):
    if 'tensor' in name.lower() or 'raw' in name.lower():
        print(f"  {name}")

# Check if GGUFFile is used internally
for name in sorted(dir(gguf)):
    if 'File' in name or 'Writer' in name or 'Reader' in name:
        print(f"  {name}")

print()
# Check tensor type values
print("GGUFValueType:", list(gguf.GGUFValueType))

print()
# Check if there's a raw_dtype parameter
src = inspect.getsource(gguf.GGUFWriter.add_tensor_info)
print("add_tensor_info source:")
print(src[:500])
