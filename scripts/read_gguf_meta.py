"""Read GGUF metadata and tensor names."""
import struct, sys

path = sys.argv[1]
print(f"Reading GGUF: {path}")
print()

with open(path, 'rb') as f:
    magic = f.read(4)
    assert magic == b'GGUF', f'Not GGUF: {magic}'
    version = struct.unpack('<I', f.read(4))[0]
    tensor_count = struct.unpack('<Q', f.read(8))[0]
    meta_kv_count = struct.unpack('<Q', f.read(8))[0]
    
    print(f"GGUF v{version}")
    print(f"Tensors: {tensor_count}")
    print(f"Metadata entries: {meta_kv_count}")
    print()
    
    type_names = {
        0: 'uint8', 1: 'int8', 2: 'uint16', 3: 'int16',
        4: 'uint32', 5: 'int32', 6: 'float32', 7: 'bool',
        8: 'string', 9: 'array(uint8)', 10: 'array(int8)',
        11: 'array(uint16)', 12: 'array(int16)', 13: 'array(uint32)',
        14: 'array(int32)', 15: 'array(float32)', 16: 'array(bool)',
        17: 'array(string)', 18: 'array(array)'
    }
    
    for i in range(meta_kv_count):
        key_len = struct.unpack('<Q', f.read(8))[0]
        key = f.read(key_len).decode('utf-8')
        val_type = struct.unpack('<I', f.read(4))[0]
        
        val = '?'
        if val_type == 8:  # string
            s_len = struct.unpack('<Q', f.read(8))[0]
            val = f.read(s_len).decode('utf-8')
        elif val_type == 6:  # float32
            val = str(struct.unpack('<f', f.read(4))[0])
        elif val_type == 4:  # uint32
            val = str(struct.unpack('<I', f.read(4))[0])
        elif val_type == 5:  # int32
            val = str(struct.unpack('<i', f.read(4))[0])
        elif val_type == 7:  # bool
            val = str(bool(struct.unpack('<B', f.read(1))[0]))
        elif val_type == 0:  # uint8
            val = str(struct.unpack('<B', f.read(1))[0])
        elif val_type == 12:  # int64
            val = str(struct.unpack('<q', f.read(8))[0])
        elif val_type in [9, 10, 17]:  # array
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            items = []
            for j in range(arr_len):
                if arr_type == 8:  # string elements
                    sl = struct.unpack('<Q', f.read(8))[0]
                    items.append(f.read(sl).decode('utf-8'))
                elif arr_type == 6:  # float32 elements
                    items.append(struct.unpack('<f', f.read(4))[0])
                elif arr_type == 5:  # int32 elements
                    items.append(struct.unpack('<i', f.read(4))[0])
                elif arr_type == 3:  # int16 elements
                    items.append(struct.unpack('<h', f.read(2))[0])
                else:
                    f.read(4)
            val = str(items[:20])  # truncate
        else:
            f.read(4)
        
        print(f"  {key}: {val[:300]}")
