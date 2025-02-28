use anyhow::{Error as E, Result};
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
use tokenizers::Tokenizer;

use crate::jetstream::PostInfo;

pub struct Embeddings {
    tx: Sender<PostInfo>,
    count: Arc<AtomicUsize>,
}

impl Embeddings {
    pub fn try_new<S>(query: S, threshold: f32) -> anyhow::Result<Self>
    where
        S: AsRef<str> + Send + Sync,
    {
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            "answerdotai/ModernBERT-large".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        let tokenizer_filename = repo.get("tokenizer.json")?;
        let config_filename = repo.get("config.json")?;
        let weights_filename = repo.get("model.safetensors")?;

        let config = std::fs::read_to_string(config_filename)?;
        let config: modernbert::Config = serde_json::from_str(&config)?;

        // https://github.com/huggingface/candle/issues/2637
        // let device = Arc::new(Device::new_metal(0)?);
        let device = Arc::new(Device::Cpu); //XXX test metal/cuda - has thread issues
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights_filename],
                candle_core::DType::F32,
                &device,
            )?
        };

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)?;
        let tokenizer = Arc::new(tokenizer);

        let model = Arc::new(modernbert::ModernBert::load(vb, &config)?);

        let thread_count = thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(2);

        let query = Arc::new(embed(&model, &tokenizer, query.as_ref(), &device)?);
        let post_count = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = flume::unbounded::<PostInfo>();

        for _ in 0..thread_count {
            let model = model.clone();
            let tokenizer = tokenizer.clone();
            let device = device.clone();
            let query = query.clone();
            let post_count = post_count.clone();
            let rx = rx.clone();
            thread::spawn(move || {
                while let Ok(post) = rx.recv() {
                    match embed(&model, &tokenizer, &post.record.text, &device) {
                        Ok(embedding) => match cosine_similarity(&query, &embedding) {
                            Ok(similarity) => {
                                if similarity > threshold {
                                    println!(
                                        "\n{}\nhttps://bsky.app/profile/{}/post/{}",
                                        post.record.text,
                                        post.did.as_str(),
                                        post.rkey,
                                    );
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

    pub fn add_post(&self, post: PostInfo) -> anyhow::Result<()> {
        self.tx.send(post)?;
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

    // [CLS] token
    let cls_embedding = embeddings.get(0)?.get(0)?;
    normalize_l2(&cls_embedding)
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor, anyhow::Error> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
    let sum_ab = (a * b)?.sum_all()?.to_scalar::<f32>()?;
    let sum_a = (a * a)?.sum_all()?.to_scalar::<f32>()?;
    let sum_b = (b * b)?.sum_all()?.to_scalar::<f32>()?;
    Ok(sum_ab / (sum_a * sum_b).sqrt())
}
