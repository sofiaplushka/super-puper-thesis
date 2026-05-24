# Embeddings

Dataset: `data/processed/anekdots_tagged.csv`
Rows: **5509**
Model: `BAAI/bge-m3`
Device: `cuda:0`
Embedding shape: `[5509, 1024]`
PCA shape: `[5509, 128]`
Normalized: `True`
Runtime seconds: `124.072`
PCA explained variance: `0.6721252799034119`
Tokenizer max sequence length: `8192`
Texts truncated by tokenizer: `0`

## Text length statistics
```json
{
  "chars": {
    "min": 23,
    "median": 133.0,
    "mean": 193.24523506988564,
    "max": 5931
  },
  "words": {
    "min": 3,
    "median": 21.0,
    "mean": 30.793428934470867,
    "max": 892
  }
}
```

No character-level truncation is applied by the pipeline. If truncation is needed,
it is handled by the model tokenizer and counted above.

## Reproduce
```bash
python scripts/02_compute_embeddings.py --input data/processed/anekdots_tagged.csv --text-column text --id-column id --model BAAI/bge-m3 --batch-size 32 --mode local --output-dir data/embeddings --seed 42
```