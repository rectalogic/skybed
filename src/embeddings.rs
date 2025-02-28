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
                &device, //XXX test metal/cuda
            )?
        };

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        tokenizer
            .with_padding(Some(PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                pad_id: config.pad_token_id,
                ..Default::default()
            }))
            .with_truncation(None)
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

    // Use the [CLS] token embedding (first token)
    Ok(embeddings.get(0)?.get(0)?)
}

fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.reshape((a.elem_count(),))?;
    let b = b.reshape((b.elem_count(),))?;

    let dot_product = (&a * &b)?.sum_all()?;

    let a_norm = a.sqr()?.sum_all()?.sqrt()?;
    let b_norm = b.sqr()?.sum_all()?.sqrt()?;

    let norm_product = a_norm.mul(&b_norm)?;
    let similarity = dot_product.div(&norm_product)?.to_scalar()?;

    Ok(similarity)
}
