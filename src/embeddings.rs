use rbert::{Bert, EmbedderExt, Embedding};

pub struct Embeddings {
    bert: Bert,
    query: Embedding,
    count: usize,
}

impl Embeddings {
    pub async fn try_new<S>(query: S) -> anyhow::Result<Self>
    where
        S: ToString,
    {
        let bert = Bert::new().await?;
        let embedding = bert.embed(query).await?;
        Ok(Self {
            bert,
            query: embedding,
            count: 0,
        })
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub async fn add_message<S: ToString>(&mut self, message: S) -> Result<f32, anyhow::Error> {
        let embedding = self.bert.embed(message).await?;
        self.count += 1;
        Ok(embedding.cosine_similarity(&self.query))
    }
}
