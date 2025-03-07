use anyhow::Error as E;
use candle_core::{Device, Tensor};
use candle_nn::var_builder::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Tokenizer;

pub(crate) struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    pub fn try_new<S>(model: S) -> anyhow::Result<Self>
    where
        S: AsRef<str> + Send + Sync,
    {
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            model.as_ref().to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        let tokenizer_filename = repo.get("tokenizer.json")?;
        let config_filename = repo.get("config.json")?;
        let weights_filename = repo.get("model.safetensors")?;

        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;

        // https://github.com/huggingface/candle/issues/2637
        // let device = Device::new_metal(0)?;
        let device = Device::Cpu; //XXX test metal/cuda - has thread issues
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)?;
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub fn embed(&self, input: &str) -> anyhow::Result<Tensor> {
        let tokens = self.tokenizer.encode(input, true).map_err(E::msg)?;

        let token_ids = tokens.get_ids().to_vec();
        let token_ids = Tensor::new(token_ids.as_slice(), &self.device)?.unsqueeze(0)?;

        let attention_mask = tokens.get_attention_mask().to_vec();
        let attention_mask = Tensor::new(attention_mask.as_slice(), &self.device)?.unsqueeze(0)?;

        let token_type_ids = token_ids.zeros_like()?;

        let embeddings = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;

        Embedder::normalize_l2(&embeddings)
    }

    fn normalize_l2(v: &Tensor) -> anyhow::Result<Tensor> {
        Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
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
        let embedder = Embedder::try_new("sentence-transformers/all-MiniLM-L6-v2")?;

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
        let embedder = Embedder::try_new("sentence-transformers/all-MiniLM-L6-v2")?;

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
        let embedder = Embedder::try_new("sentence-transformers/all-MiniLM-L6-v2")?;

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
        let embedder = Embedder::try_new("sentence-transformers/all-MiniLM-L6-v2")?;

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
