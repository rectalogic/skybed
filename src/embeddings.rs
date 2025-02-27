mod search;
use std::{
    iter::zip,
    sync::mpsc,
    thread::{self, JoinHandle},
};

use crate::LOG_COUNT;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use search::{EmbeddingText, Search};

pub struct Embeddings {
    tx: mpsc::Sender<String>,
    join: JoinHandle<Result<(), anyhow::Error>>,
}

impl Embeddings {
    pub fn try_new<S>(query: S, threshold: f32) -> anyhow::Result<Self>
    where
        S: AsRef<str> + Send + Sync,
    {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::ModernBertEmbedLarge)
                .with_show_download_progress(true),
        )?;
        let query_embedding = model.embed(vec![query], None)?.into_iter().next().unwrap();
        let (tx, rx) = mpsc::channel::<String>();
        let join = thread::spawn(move || -> anyhow::Result<()> {
            const MAX_CAPACITY: usize = 128;
            let mut count: usize = 0;
            let mut search = Search::new(query_embedding, threshold);
            let mut messages = Vec::with_capacity(MAX_CAPACITY);
            while let Ok(message) = rx.recv() {
                if messages.len() == messages.capacity() {
                    let embedding_result =
                        model.embed(messages.iter().collect::<Vec<_>>(), Some(MAX_CAPACITY / 4))?;
                    search.search(
                        zip(embedding_result, messages)
                            .map(|(embedding, message)| EmbeddingText::new(embedding, message))
                            .collect::<Vec<_>>(),
                    )?;
                    if search.count() - count >= LOG_COUNT {
                        eprintln!("{} embeddings", search.count());
                        count = search.count()
                    }
                    messages = Vec::with_capacity(MAX_CAPACITY);
                } else {
                    messages.push(message);
                }
            }
            search.join()
        });
        Ok(Self { tx, join })
    }

    pub fn add_message(&self, message: String) -> Result<(), anyhow::Error> {
        self.tx.send(message)?;
        Ok(())
    }

    pub fn join(self) -> Result<(), anyhow::Error> {
        self.join
            .join()
            .map_err(|e| anyhow::anyhow!("Thread join error: {:?}", e))?
            .map_err(|e| anyhow::anyhow!("Thread error: {}", e))?;
        Ok(())
    }
}
