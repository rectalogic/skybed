use anyhow::{Error as E, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::modernbert;
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Tokenizer;

pub(crate) struct Embedder {
    model: modernbert::ModernBert,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    pub fn try_new() -> anyhow::Result<Self> {
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
        // let device = Device::new_metal(0)?;
        let device = Device::Cpu; //XXX test metal/cuda - has thread issues
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

        let model = modernbert::ModernBert::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn embed(&self, input: &str) -> anyhow::Result<Tensor> {
        let tokens = self.tokenizer.encode(input, true).map_err(E::msg)?;
        let token_ids = tokens.get_ids().to_vec();
        let attention_mask = tokens.get_attention_mask().to_vec();

        // Convert to tensors
        let token_ids = Tensor::new(token_ids.as_slice(), &self.device)?;
        let attention_mask = Tensor::new(attention_mask.as_slice(), &self.device)?;

        // Add batch dimension
        let token_ids = token_ids.unsqueeze(0)?;
        let attention_mask = attention_mask.unsqueeze(0)?;

        let embeddings = self.model.forward(&token_ids, &attention_mask)?;

        // Average the last hidden states of all tokens
        // This is a simple way to get a sentence embedding
        let embeddings = embeddings.squeeze(0)?; // Remove batch dimension

        // Simple mean across all token embeddings
        let mean_embedding = embeddings.mean(0)?;

        // Normalize
        Embedder::normalize_l2(&mean_embedding)
    }

    fn normalize_l2(v: &Tensor) -> Result<Tensor, anyhow::Error> {
        let norm = v.sqr()?.sum(0)?.sqrt()?;
        Ok(v.broadcast_div(&norm)?)
    }

    pub fn cosine_similarity(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
        Ok((a * b)?.sum_all()?.to_scalar::<f32>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_similarity() -> Result<()> {
        let embedder = Embedder::try_new()?;

        let text1 = "This is a test sentence about artificial intelligence.";
        let text2 = "AI and machine learning are fascinating topics.";
        let text3 = "The weather is nice today.";

        let embedding1 = embedder.embed(text1)?;
        let embedding2 = embedder.embed(text2)?;
        let embedding3 = embedder.embed(text3)?;

        let similarity12 = Embedder::cosine_similarity(&embedding1, &embedding2)?;
        let similarity13 = Embedder::cosine_similarity(&embedding1, &embedding3)?;

        println!("Similarity between related texts: {}", similarity12);
        println!("Similarity between unrelated texts: {}", similarity13);

        // Related texts should have higher similarity
        assert!(similarity12 > similarity13);

        Ok(())
    }

    #[test]
    fn test_identical_similarity() -> Result<()> {
        let embedder = Embedder::try_new()?;

        let text1 = "The weather is nice today.";
        let text2 = "The weather is nice today.";

        let embedding1 = embedder.embed(text1)?;
        let embedding2 = embedder.embed(text2)?;

        let similarity = Embedder::cosine_similarity(&embedding1, &embedding2)?;

        println!("Similarity between identical texts: {}", similarity);

        assert!(similarity == 1.0);

        Ok(())
    }

    #[test]
    fn test_nearly_identical_similarity() -> Result<()> {
        let embedder = Embedder::try_new()?;

        let text1 = "The weather is nice today.";
        let text2 = "The weather is really nice today.";

        let embedding1 = embedder.embed(text1)?;
        let embedding2 = embedder.embed(text2)?;

        let similarity = Embedder::cosine_similarity(&embedding1, &embedding2)?;

        println!("Similarity between nearly identical texts: {}", similarity);

        assert!(similarity >= 0.9);

        Ok(())
    }

    #[test]
    fn test_unrelated_similarity() -> Result<()> {
        let embedder = Embedder::try_new()?;

        let text1 = "US federal government layoffs";
        let text2 = "I like pizza";

        let embedding1 = embedder.embed(text1)?;
        let embedding2 = embedder.embed(text2)?;

        let similarity = Embedder::cosine_similarity(&embedding1, &embedding2)?;

        println!("Similarity between unrelated texts: {}", similarity);

        assert!(similarity < 0.5);

        Ok(())
    }
}
