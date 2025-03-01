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

        // Mean pooling
        let (_batch_size, _n_tokens, hidden_size) = embeddings.dims3()?;

        // Use attention mask to only average over real tokens
        let expanded_mask = attention_mask
            .unsqueeze(2)?
            .expand(&[embeddings.dim(0)?, embeddings.dim(1)?, embeddings.dim(2)?])?
            .to_dtype(embeddings.dtype())?;

        // Check if mask sum is too small
        let mask_sum = expanded_mask.sum_all()?;
        let mask_sum_value = mask_sum.to_scalar::<f32>()?;
        if mask_sum_value < 1e-10 {
            return Err(anyhow::anyhow!("Attention mask sum too close to zero"));
        }

        // Calculate mean pooled embedding
        let sum_masked = embeddings.mul(&expanded_mask)?.sum(1)?;
        // The mask sum should be summed along the sequence dimension for each feature
        let mask_sum_per_feature = expanded_mask.sum(1)?;
        // Now both tensors have shape [batch_size, hidden_size]
        let mean_pooled = sum_masked.broadcast_div(&mask_sum_per_feature)?;

        // Normalize
        let normalized = Self::normalize_l2(&mean_pooled)?;
        Ok(normalized.reshape((hidden_size,))?)
    }

    fn normalize_l2(v: &Tensor) -> Result<Tensor, anyhow::Error> {
        Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
    }

    pub fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
        let sum_ab = (a * b)?.sum_all()?.to_scalar::<f32>()?;
        let sum_a = (a * a)?.sum_all()?.to_scalar::<f32>()?;
        let sum_b = (b * b)?.sum_all()?.to_scalar::<f32>()?;
        Ok(sum_ab / (sum_a * sum_b).sqrt())
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
}
