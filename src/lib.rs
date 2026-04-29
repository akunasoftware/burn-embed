//! Simple text embedding models built with Burn.
//!
//! # Example
//!
//! ```rust,no_run
//! use burn_embed::{EmbeddingModel, TextEmbedding, TextEmbeddingInitOptions};
//! use burn_wgpu::{Wgpu, WgpuDevice};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let device = WgpuDevice::default();
//!     let model =
//!         TextEmbedding::<Wgpu>::new(
//!             &device,
//!             TextEmbeddingInitOptions {
//!                 model: EmbeddingModel::MiniLmL12,
//!                 cache_dir: None,
//!             },
//!         )
//!         .await?;
//!
//!     let single = model.embed("Hello world")?;
//!     assert!(!single.is_empty());
//!
//!     let batch = model.embed_batch(&["Hello world", "Rust embeddings"], None)?;
//!     assert_eq!(batch.len(), 2);
//!
//!     Ok(())
//! }
//! ```

mod minilm;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use burn::tensor::{Tensor, backend::Backend};

use crate::minilm::{MiniLmEmbeddingModel, MiniLmVariant, load_pretrained_mini_lm};

/// Supported embedding model checkpoints.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EmbeddingModel {
    MiniLmL6,
    #[default]
    MiniLmL12,
}

impl From<EmbeddingModel> for MiniLmVariant {
    fn from(value: EmbeddingModel) -> Self {
        match value {
            EmbeddingModel::MiniLmL6 => MiniLmVariant::L6,
            EmbeddingModel::MiniLmL12 => MiniLmVariant::L12,
        }
    }
}

/// Initialization options for [`TextEmbedding`].
#[derive(Debug, Clone, Default)]
pub struct TextEmbeddingInitOptions {
    /// Which embedding checkpoint to load.
    pub model: EmbeddingModel,
    /// Optional Hugging Face cache directory override.
    pub cache_dir: Option<PathBuf>,
}

/// Minimal text embedding interface inspired by `fastembed-rs`.
#[derive(Debug)]
pub struct TextEmbedding<B: Backend> {
    model: MiniLmEmbeddingModel<B>,
    device: B::Device,
}

impl<B> TextEmbedding<B>
where
    B: Backend,
{
    /// Loads a MiniLM text embedding model onto the provided device.
    pub async fn new(device: &B::Device, options: TextEmbeddingInitOptions) -> Result<Self> {
        let model =
            load_pretrained_mini_lm(device, options.model.into(), options.cache_dir).await?;

        Ok(Self {
            model,
            device: device.clone(),
        })
    }

    /// Embeds a single document and returns one embedding vector.
    pub fn embed(&self, document: impl AsRef<str>) -> Result<Vec<f32>> {
        let document = document.as_ref();
        let documents = [document];
        let mut embeddings = self.embed_batch(documents.as_slice(), None)?;
        embeddings
            .pop()
            .context("expected one embedding for a single input document")
    }

    /// Embeds documents in batches and returns one vector per input string.
    pub fn embed_batch<S: AsRef<str>>(
        &self,
        documents: &[S],
        batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = batch_size_or_default(documents.len(), batch_size)?;

        let mut embeddings = Vec::with_capacity(documents.len());
        for batch in documents.chunks(batch_size) {
            let batch_documents = batch.iter().map(AsRef::as_ref).collect::<Vec<_>>();
            let batch_embeddings = self.model.encode(&batch_documents, &self.device)?;
            embeddings.extend(tensor_to_rows(batch_embeddings)?);
        }

        Ok(embeddings)
    }

    /// Returns the loaded embedding checkpoint.
    pub fn model(&self) -> EmbeddingModel {
        match self.model.variant {
            MiniLmVariant::L6 => EmbeddingModel::MiniLmL6,
            MiniLmVariant::L12 => EmbeddingModel::MiniLmL12,
        }
    }
}

fn batch_size_or_default(document_count: usize, batch_size: Option<usize>) -> Result<usize> {
    let batch_size = batch_size.unwrap_or(document_count);
    if batch_size == 0 {
        bail!("batch size must be greater than zero");
    }

    Ok(batch_size)
}

fn tensor_to_rows<B: Backend>(embeddings: Tensor<B, 2>) -> Result<Vec<Vec<f32>>> {
    let [row_count, column_count] = embeddings.dims();
    let data = embeddings.into_data().convert::<f32>();
    let values = data
        .as_slice::<f32>()
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to read embedding output tensor")?;

    Ok(values
        .chunks(column_count)
        .take(row_count)
        .map(|row| row.to_vec())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    use burn_wgpu::{Wgpu, WgpuDevice};

    #[test]
    fn tensor_rows_are_extracted() {
        let device = WgpuDevice::default();
        let embeddings =
            Tensor::<Wgpu<f32, i64>, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);

        let rows = tensor_to_rows(embeddings).expect("rows should extract");
        assert_eq!(rows, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn batch_size_must_be_non_zero() {
        let error = batch_size_or_default(1, Some(0)).expect_err("zero batch size should fail");
        assert!(
            error
                .to_string()
                .contains("batch size must be greater than zero")
        );
    }

    #[test]
    fn empty_batch_returns_no_embeddings() {
        let batch_size = batch_size_or_default(4, None).expect("default batch size should work");
        assert_eq!(batch_size, 4);
    }

    #[test]
    fn init_options_default_to_l12() {
        assert_eq!(
            TextEmbeddingInitOptions::default().model,
            EmbeddingModel::MiniLmL12
        );
    }

    #[tokio::test]
    #[ignore]
    async fn loads_and_embeds_text() {
        let device = WgpuDevice::default();
        let model = TextEmbedding::<Wgpu<f32, i64>>::new(
            &device,
            TextEmbeddingInitOptions {
                model: EmbeddingModel::MiniLmL6,
                cache_dir: None,
            },
        )
        .await
        .expect("model should load");

        let single = model
            .embed("Hello world")
            .expect("single embed should work");
        assert!(!single.is_empty());

        let batch = model
            .embed_batch(&["Hello world", "Rust embeddings"], None)
            .expect("batch embed should work");
        assert_eq!(batch.len(), 2);
        assert!(batch.iter().all(|embedding| !embedding.is_empty()));
    }

    #[tokio::test]
    #[ignore]
    async fn loads_and_embeds_text_with_non_i64_backend() {
        let device = WgpuDevice::default();
        let model = TextEmbedding::<Wgpu<f32, i32>>::new(
            &device,
            TextEmbeddingInitOptions {
                model: EmbeddingModel::MiniLmL6,
                cache_dir: None,
            },
        )
        .await
        .expect("model should load");

        let single = model
            .embed("Hello world")
            .expect("single embed should work");
        assert!(!single.is_empty());
    }
}
