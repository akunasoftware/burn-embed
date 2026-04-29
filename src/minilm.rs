use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn::module::Module;
use burn::nn::Initializer::KaimingUniform;
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig,
    transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
};
use burn::tensor::{Bool, Int, Tensor, backend::Backend};
use burn_store::{KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use hf_hub::api::tokio::ApiBuilder;
use serde::Deserialize;
use tokenizers::Tokenizer;

const MINILM_L6_REPO_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
const MINILM_L12_REPO_ID: &str = "sentence-transformers/all-MiniLM-L12-v2";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MiniLmVariant {
    L6,
    #[default]
    L12,
}

impl MiniLmVariant {
    pub fn repo_id(self) -> &'static str {
        match self {
            Self::L6 => MINILM_L6_REPO_ID,
            Self::L12 => MINILM_L12_REPO_ID,
        }
    }
}

#[derive(Debug)]
pub struct HfModelFiles {
    pub config_path: PathBuf,
    pub weights_path: PathBuf,
    pub tokenizer_path: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MiniLmConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub hidden_dropout_prob: f64,
    pub layer_norm_eps: f64,
}

#[derive(Debug)]
pub struct MiniLmOutput<B: Backend> {
    pub hidden_states: Tensor<B, 3>,
}

#[derive(Module, Debug)]
pub struct MiniLmEmbeddings<B: Backend> {
    word_embeddings: Embedding<B>,
    position_embeddings: Embedding<B>,
    token_type_embeddings: Embedding<B>,
    layer_norm: LayerNorm<B>,
    dropout: Dropout,
}

#[derive(Module, Debug)]
pub struct MiniLmModel<B: Backend> {
    pub embeddings: MiniLmEmbeddings<B>,
    pub encoder: TransformerEncoder<B>,
}

#[derive(Debug)]
pub struct MiniLmEmbeddingModel<B: Backend> {
    pub model: MiniLmModel<B>,
    pub tokenizer: Tokenizer,
    pub variant: MiniLmVariant,
}

impl MiniLmConfig {
    pub fn load_from_hf(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read MiniLM config at {}", path.display()))?;

        serde_json::from_str(&content)
            .with_context(|| format!("failed to parse MiniLM config at {}", path.display()))
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> MiniLmModel<B> {
        let embeddings = MiniLmEmbeddings::new(self, device);
        let encoder = TransformerEncoderConfig::new(
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_hidden_layers,
        )
        .with_dropout(self.hidden_dropout_prob)
        .with_norm_first(false)
        .with_quiet_softmax(false)
        .with_initializer(KaimingUniform {
            gain: 1.0 / 3.0f64.sqrt(),
            fan_out_only: false,
        })
        .init(device);

        MiniLmModel {
            embeddings,
            encoder,
        }
    }
}

impl<B: Backend> MiniLmEmbeddings<B> {
    fn new(config: &MiniLmConfig, device: &B::Device) -> Self {
        let word_embeddings =
            EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device);
        let position_embeddings =
            EmbeddingConfig::new(config.max_position_embeddings, config.hidden_size).init(device);
        let token_type_embeddings =
            EmbeddingConfig::new(config.type_vocab_size, config.hidden_size).init(device);
        let layer_norm = LayerNormConfig::new(config.hidden_size)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        let dropout = DropoutConfig::new(config.hidden_dropout_prob).init();

        Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
        }
    }

    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        token_type_ids: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = input_ids.device();
        let word_embeddings = self.word_embeddings.forward(input_ids);

        let position_ids = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .reshape([1, seq_len])
            .expand([batch_size, seq_len]);
        let position_embeddings = self.position_embeddings.forward(position_ids);

        let token_type_ids = token_type_ids
            .unwrap_or_else(|| Tensor::<B, 2, Int>::zeros([batch_size, seq_len], &device));
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids);

        let embeddings = word_embeddings + position_embeddings + token_type_embeddings;
        let embeddings = self.layer_norm.forward(embeddings);
        self.dropout.forward(embeddings)
    }
}

impl<B: Backend> MiniLmModel<B> {
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2>,
        token_type_ids: Option<Tensor<B, 2, Int>>,
    ) -> MiniLmOutput<B> {
        let embeddings = self.embeddings.forward(input_ids, token_type_ids);
        let device = attention_mask.device();
        let zeros = Tensor::<B, 2>::zeros(attention_mask.shape(), &device);
        let mask_pad: Tensor<B, 2, Bool> = attention_mask.equal(zeros);
        let encoder_input = TransformerEncoderInput::new(embeddings).mask_pad(mask_pad);
        let hidden_states = self.encoder.forward(encoder_input);

        MiniLmOutput { hidden_states }
    }
}

impl<B> MiniLmEmbeddingModel<B>
where
    B: Backend,
{
    pub fn encode(&self, sentences: &[&str], device: &B::Device) -> Result<Tensor<B, 2>> {
        let (input_ids, attention_mask) = tokenize_batch::<B>(&self.tokenizer, sentences, device)?;
        let output = self.model.forward(input_ids, attention_mask.clone(), None);

        Ok(normalize_l2(mean_pooling(
            output.hidden_states,
            attention_mask,
        )))
    }
}

pub async fn load_pretrained_mini_lm<B>(
    device: &B::Device,
    variant: MiniLmVariant,
    cache_dir: Option<PathBuf>,
) -> Result<MiniLmEmbeddingModel<B>>
where
    B: Backend,
{
    let files = download_hf_model(variant, cache_dir).await?;
    let config = MiniLmConfig::load_from_hf(&files.config_path)?;
    let mut model = config.init(device);
    load_pretrained_weights(&mut model, &files.weights_path)?;
    let tokenizer = Tokenizer::from_file(&files.tokenizer_path)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .with_context(|| {
            format!(
                "failed to load MiniLM tokenizer from {}",
                files.tokenizer_path.display()
            )
        })?;

    Ok(MiniLmEmbeddingModel {
        model,
        tokenizer,
        variant,
    })
}

pub async fn download_hf_model(
    variant: MiniLmVariant,
    cache_dir: Option<PathBuf>,
) -> Result<HfModelFiles> {
    let mut builder = ApiBuilder::new().with_progress(true);
    if let Some(cache_dir) = cache_dir {
        builder = builder.with_cache_dir(cache_dir);
    }

    let api = builder
        .build()
        .context("failed to initialize Hugging Face API for MiniLM")?;
    let repo = api.model(variant.repo_id().to_string());

    let config_path = repo
        .get("config.json")
        .await
        .with_context(|| format!("failed to fetch MiniLM config for {}", variant.repo_id()))?;
    let weights_path = repo
        .get("model.safetensors")
        .await
        .with_context(|| format!("failed to fetch MiniLM weights for {}", variant.repo_id()))?;
    let tokenizer_path = repo
        .get("tokenizer.json")
        .await
        .with_context(|| format!("failed to fetch MiniLM tokenizer for {}", variant.repo_id()))?;

    Ok(HfModelFiles {
        config_path,
        weights_path,
        tokenizer_path,
    })
}

pub fn load_pretrained_weights<B: Backend>(
    model: &mut MiniLmModel<B>,
    checkpoint_path: impl AsRef<Path>,
) -> Result<()> {
    let key_mappings = vec![
        ("^bert\\.(.+)", "$1"),
        ("encoder\\.layer\\.([0-9]+)", "encoder.layers.$1"),
        ("attention\\.self\\.query", "mha.query"),
        ("attention\\.self\\.key", "mha.key"),
        ("attention\\.self\\.value", "mha.value"),
        ("attention\\.output\\.dense", "mha.output"),
        ("attention\\.output\\.LayerNorm", "norm_1"),
        ("intermediate\\.dense", "pwff.linear_inner"),
        ("(layers\\.[0-9]+)\\.output\\.dense", "$1.pwff.linear_outer"),
        ("(layers\\.[0-9]+)\\.output\\.LayerNorm", "$1.norm_2"),
        ("embeddings\\.LayerNorm", "embeddings.layer_norm"),
    ];

    let remapper = KeyRemapper::from_patterns(key_mappings)
        .context("failed to create MiniLM weight remapper")?;
    let mut store = SafetensorsStore::from_file(checkpoint_path.as_ref())
        .with_from_adapter(PyTorchToBurnAdapter)
        .remap(remapper);

    model.load_from(&mut store).with_context(|| {
        format!(
            "failed to load MiniLM weights from {}",
            checkpoint_path.as_ref().display()
        )
    })?;

    Ok(())
}

pub fn tokenize_batch<B: Backend>(
    tokenizer: &Tokenizer,
    sentences: &[&str],
    device: &B::Device,
) -> Result<(Tensor<B, 2, Int>, Tensor<B, 2>)> {
    let encodings = tokenizer
        .encode_batch(sentences.to_vec(), true)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to tokenize input batch")?;

    let max_len = encodings
        .iter()
        .map(|encoding| encoding.get_ids().len())
        .max()
        .unwrap_or(1);

    let batch_size = sentences.len();
    let mut input_ids = vec![0i32; batch_size * max_len];
    let mut attention_mask = vec![0.0f32; batch_size * max_len];

    for (batch_index, encoding) in encodings.iter().enumerate() {
        for (token_index, token_id) in encoding.get_ids().iter().enumerate() {
            input_ids[batch_index * max_len + token_index] = *token_id as i32;
            attention_mask[batch_index * max_len + token_index] =
                encoding.get_attention_mask()[token_index] as f32;
        }
    }

    let input_ids =
        Tensor::<B, 1, Int>::from_ints(input_ids.as_slice(), device).reshape([batch_size, max_len]);
    let attention_mask = Tensor::<B, 1>::from_floats(attention_mask.as_slice(), device)
        .reshape([batch_size, max_len]);

    Ok((input_ids, attention_mask))
}

fn mean_pooling<B: Backend>(
    hidden_states: Tensor<B, 3>,
    attention_mask: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let [batch_size, seq_len, hidden_size] = hidden_states.dims();
    let mask_expanded = attention_mask
        .clone()
        .reshape([batch_size, seq_len, 1])
        .expand([batch_size, seq_len, hidden_size]);
    let sum_hidden = (hidden_states * mask_expanded)
        .sum_dim(1)
        .reshape([batch_size, hidden_size]);
    let token_counts = attention_mask
        .sum_dim(1)
        .reshape([batch_size, 1])
        .expand([batch_size, hidden_size])
        .clamp_min(1e-9);

    sum_hidden / token_counts
}

fn normalize_l2<B: Backend>(embeddings: Tensor<B, 2>) -> Tensor<B, 2> {
    use burn::tensor::linalg::{Norm, vector_normalize};

    vector_normalize(embeddings, Norm::L2, 1, 1e-12)
}
