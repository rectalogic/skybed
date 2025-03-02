use anyhow::Error as E;
use candle_core::{Device, Tensor};
use candle_nn::var_builder::VarBuilder;
use hf_hub::{Repo, RepoType, api::sync::Api};
use mpnet_rs::mpnet::{MPNetConfig, MPNetModel, MPNetPooler, PoolingConfig};
use tokenizers::Tokenizer;

pub(crate) struct Embedder {
    model: MPNetModel,
    tokenizer: Tokenizer,
    pooler: MPNetPooler,
    device: Device,
}

impl Embedder {
    pub fn try_new() -> anyhow::Result<Self> {
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            "sentence-transformers/all-mpnet-base-v2".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        let tokenizer_filename = repo.get("tokenizer.json")?;
        let config_filename = repo.get("config.json")?;
        let weights_filename = repo.get("model.safetensors")?;

        let config = MPNetConfig::load(&config_filename)?;

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

        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let pooler = MPNetPooler::load(vb.clone(), &PoolingConfig::default())?;
        let model = MPNetModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            pooler,
            device,
        })
    }

    pub fn embed(&self, input: &str) -> anyhow::Result<Tensor> {
        let tokens = self.tokenizer.encode(input, true).map_err(E::msg)?;
        let token_ids = tokens.get_ids().to_vec();

        let token_ids = Tensor::new(token_ids.as_slice(), &self.device)?;
        let token_ids = token_ids.unsqueeze(0)?;

        let embeddings = self.model.forward(&token_ids, false)?;
        Ok(self.pooler.forward(&embeddings)?)
    }

    pub fn cosine_similarity(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
        Ok((a * b)?.sum_all()?.to_scalar::<f32>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_similarity() -> anyhow::Result<()> {
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
    fn test_identical_similarity() -> anyhow::Result<()> {
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
    fn test_nearly_identical_similarity() -> anyhow::Result<()> {
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
    fn test_unrelated_similarity() -> anyhow::Result<()> {
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
