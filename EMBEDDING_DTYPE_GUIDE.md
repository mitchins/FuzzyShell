# FuzzyShell Embedding Storage Configuration

FuzzyShell now supports configurable embedding storage formats for optimal balance between quality and storage size.

## Configuration

Edit `src/fuzzyshell/fuzzyshell.py` and modify these constants:

```python
# Configuration constants for embedding storage
EMBEDDING_DTYPE = np.float16  # Options: np.int8, np.float16, np.float32
EMBEDDING_SCALE_FACTOR = 127  # Only used for INT8 quantization
```

## Storage Options

| Format | Quality | Storage per Embedding | Total for 4,402 commands | Notes |
|--------|---------|----------------------|---------------------------|-------|
| `np.int8` | Basic | 384 bytes | ~1.6 MB | Legacy quantized format |
| `np.float16` | **Recommended** | 768 bytes | ~3.2 MB | Best balance of quality/size |  
| `np.float32` | Highest | 1,536 bytes | ~6.4 MB | Full precision |

## Database Migration

When changing `EMBEDDING_DTYPE`, you need to regenerate your database:

```bash
# 1. Backup your current database
cp ~/.fuzzyshell/history.db ~/.fuzzyshell/history.db.backup

# 2. Remove the old database
rm ~/.fuzzyshell/history.db

# 3. Re-ingest with new format
fuzzyshell --ingest ~/.bash_history
```

## Compatibility Checking

The system automatically tracks the embedding format in database metadata:

- **First run**: Stores the current `EMBEDDING_DTYPE` in database metadata
- **Subsequent runs**: Warns if dtype has changed and suggests regenerating database
- **Database info**: `get_database_info()` returns the stored embedding format

## Implementation Details

### Storage
- **INT8**: Quantizes embeddings to [-127, 127] range with scale factor
- **FP16**: Stores embeddings as 16-bit floats (half precision)
- **FP32**: Stores embeddings as 32-bit floats (full precision)

### Retrieval
- All formats are converted to FP32 for computation to ensure compatibility
- Query embeddings from the model are handled transparently
- No changes needed to search API

### Metadata Tracking
```python
db_info = fs.get_database_info()
print(f"Embedding format: {db_info['embedding_dtype']}")
```

## Quality Impact

Based on testing with terminal commands:

- **INT8**: Quantization artifacts can reduce semantic similarity precision
- **FP16**: Excellent quality with minimal artifacts, 2x storage increase
- **FP32**: Perfect quality, 4x storage increase

**Recommendation**: Use `np.float16` for the best balance of search quality and storage efficiency.

## Advanced Usage

You can check if your database needs regeneration:

```python
from fuzzyshell.fuzzyshell import FuzzyShell, EMBEDDING_DTYPE

fs = FuzzyShell()
db_info = fs.get_database_info()
current_dtype = str(EMBEDDING_DTYPE).split('.')[-1]

if db_info['embedding_dtype'] != current_dtype:
    print(f"Database uses {db_info['embedding_dtype']}, but code is configured for {current_dtype}")
    print("Consider regenerating database for optimal quality")
```