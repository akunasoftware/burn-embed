# Burn-Embed

Simple text embedding models built with [Burn](https://github.com/tracel-ai/burn).

The embedding API accepts any Burn backend supported by the model.

With `burn-wgpu`, `Wgpu` already defaults to `f32` / `i32`,
so callers can usually just write `TextEmbedding::<Wgpu>`.

## Usage

```rust
use burn_embed::{EmbeddingModel, TextEmbedding, TextEmbeddingInitOptions};
use burn_wgpu::{Wgpu, WgpuDevice};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = WgpuDevice::default();
    let model =
        TextEmbedding::<Wgpu>::new(
            &device,
            TextEmbeddingInitOptions {
                model: EmbeddingModel::MiniLmL12,
                cache_dir: None,
            },
        )
        .await?;

    let single = model.embed("Hello world")?;
    assert!(!single.is_empty());

    let batch = model.embed_batch(&["Hello world", "Rust embeddings"], None)?;
    assert_eq!(batch.len(), 2);

    Ok(())
}
```

## Models

- `EmbeddingModel::MiniLmL12` default
- `EmbeddingModel::MiniLmL6`

## Testing

```bash
cargo test
```
