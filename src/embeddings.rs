use anyhow::{Error as E, Result};
use atrium_api::{app::bsky::feed::post, types::string};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::modernbert;
use flume::Sender;
use hf_hub::{Repo, RepoType, api::sync::Api};
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};
use tokenizers::{PaddingParams, Tokenizer};

pub struct Embeddings {
    tx: Sender<(string::Did, Box<post::Record>)>,
    count: Arc<AtomicUsize>,
}

impl Embeddings {
    pub fn try_new<S>(query: S, threshold: f32) -> anyhow::Result<Self>
    where
        S: AsRef<str> + Send + Sync,
    {
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            "answerdotai/ModernBERT-base".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        let tokenizer_filename = repo.get("tokenizer.json")?;
        let config_filename = repo.get("config.json")?;
        let weights_filename = repo.get("model.safetensors")?;

        let config = std::fs::read_to_string(config_filename)?;
        let config: modernbert::Config = serde_json::from_str(&config)?;

        //XXX let device = candle_examples::device(args.cpu)?;
        let device = Arc::new(Device::Cpu); //XXX test metal/cuda
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights_filename],
                candle_core::DType::F32,
                &device,
            )?
        };

        // Max BlueSky post length
        let max_post_length: usize = 300.min(config.max_position_embeddings);
        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        tokenizer
            .with_padding(Some(PaddingParams {
                strategy: tokenizers::PaddingStrategy::Fixed(max_post_length),
                pad_id: config.pad_token_id,
                ..Default::default()
            }))
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: max_post_length,
                strategy: tokenizers::TruncationStrategy::LongestFirst,
                stride: 0,
                ..Default::default()
            }))
            .map_err(E::msg)?;
        let tokenizer = Arc::new(tokenizer);

        let model = Arc::new(modernbert::ModernBert::load(vb, &config)?);

        let thread_count = thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(2);

        let query = Arc::new(embed(&model, &tokenizer, query.as_ref(), &device)?);
        let post_count = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = flume::unbounded::<(string::Did, Box<post::Record>)>();

        for _ in 0..thread_count {
            let model = model.clone();
            let tokenizer = tokenizer.clone();
            let device = device.clone();
            let query = query.clone();
            let post_count = post_count.clone();
            let rx = rx.clone();
            thread::spawn(move || {
                while let Ok((did, post)) = rx.recv() {
                    match embed(&model, &tokenizer, &post.text, &device) {
                        Ok(embedding) => match cosine_similarity(&query, &embedding) {
                            Ok(similarity) => {
                                if similarity > threshold {
                                    println!("{}:\n{}", did.as_str(), post.text);
                                }
                            }
                            Err(e) => eprintln!("Error calculating cosine similarity: {}", e),
                        },
                        Err(e) => eprintln!("Error embedding: {}", e),
                    }
                    post_count.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
        Ok(Self {
            tx,
            count: post_count.clone(),
        })
    }

    pub fn count(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    pub fn add_post(&self, (did, post): (string::Did, Box<post::Record>)) -> anyhow::Result<()> {
        self.tx.send((did, post))?;
        Ok(())
    }
}

fn embed(
    model: &modernbert::ModernBert,
    tokenizer: &Arc<Tokenizer>,
    input: &str,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let tokens = tokenizer.encode(input, true).map_err(E::msg)?;
    let token_ids = tokens.get_ids().to_vec();
    let attention_mask = tokens.get_attention_mask().to_vec();

    // Convert to tensors
    let token_ids = Tensor::new(token_ids.as_slice(), device)?;
    let attention_mask = Tensor::new(attention_mask.as_slice(), device)?;

    // Add batch dimension
    let token_ids = token_ids.unsqueeze(0)?;
    let attention_mask = attention_mask.unsqueeze(0)?;

    let embeddings = model.forward(&token_ids, &attention_mask)?;

    // Mean pooling with attention mask
    // First, expand attention mask to match embedding dimensions
    let (_batch_size, seq_len, hidden_size) = embeddings.dims3()?;

    // Sum the embeddings (for each position where attention_mask is 1)
    // First convert attention_mask to same dtype as embeddings
    let attention_mask_f = attention_mask.to_dtype(embeddings.dtype())?;

    // Reshape attention mask for broadcasting
    let attention_mask_expanded =
        attention_mask_f
            .unsqueeze(2)?
            .broadcast_as((1, seq_len, hidden_size))?;

    // Apply mask (zeros out padding tokens)
    let masked_embeddings = embeddings.mul(&attention_mask_expanded)?;

    // Sum embeddings and mask values
    let summed = masked_embeddings.sum(1)?;
    let mask_sum = attention_mask_f.sum(1)?.unsqueeze(1)?;

    // Average = sum(masked_embeddings) / sum(mask)
    let mean_pooled = summed.broadcast_div(&mask_sum)?;

    // L2 normalize
    let norm = mean_pooled.sqr()?.sum_all()?.sqrt()?;
    let normalized = mean_pooled.broadcast_div(&norm)?;

    Ok(normalized.reshape((hidden_size,))?)
}

fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
    let sum_ab = (a * b)?.sum_all()?.to_scalar::<f32>()?;
    let sum_a = (a * a)?.sum_all()?.to_scalar::<f32>()?;
    let sum_b = (b * b)?.sum_all()?.to_scalar::<f32>()?;
    Ok(sum_ab / (sum_a * sum_b).sqrt())
}
