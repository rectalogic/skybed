use fastembed::Embedding;
use ndarray::ArrayView1;
use std::{
    sync::mpsc,
    thread::{self, JoinHandle},
};

pub(super) struct EmbeddingText {
    embedding: Embedding,
    text: String,
}

impl EmbeddingText {
    pub fn new(embedding: Embedding, text: String) -> Self {
        Self { embedding, text }
    }
}

pub(super) struct Search {
    tx: mpsc::Sender<Vec<EmbeddingText>>,
    join: JoinHandle<Result<(), anyhow::Error>>,
    count: usize,
}

impl Search {
    pub fn new(query: Embedding, threshold: f32) -> Self {
        let (tx, rx) = mpsc::channel::<Vec<EmbeddingText>>();
        let join = thread::spawn(move || -> anyhow::Result<()> {
            loop {
                while let Ok(embeddings) = rx.recv() {
                    for embedding in embeddings.iter() {
                        let similarity = cosine_similarity(&query, &embedding.embedding);
                        if similarity >= threshold {
                            println!("{}", embedding.text);
                        }
                    }
                }
            }
        });
        Self { tx, join, count: 0 }
    }

    pub fn search(&mut self, embeddings: Vec<EmbeddingText>) -> Result<(), anyhow::Error> {
        self.count += embeddings.len();
        self.tx.send(embeddings)?;
        Ok(())
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn join(self) -> Result<(), anyhow::Error> {
        self.join
            .join()
            .map_err(|e| anyhow::anyhow!("Thread join error: {:?}", e))?
            .map_err(|e| anyhow::anyhow!("Thread error: {}", e))?;
        Ok(())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let a = ArrayView1::from(a);
    let b = ArrayView1::from(b);

    let dot_product = a.dot(&b);
    let norm_a = a.dot(&a).sqrt();
    let norm_b = b.dot(&b).sqrt();

    dot_product / (norm_a * norm_b)
}
