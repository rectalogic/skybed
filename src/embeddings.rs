use atrium_api::{app::bsky::feed::post, types::string};
use flume::Sender;
use rbert::{Bert, EmbedderExt, Pooling};
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

pub struct Embeddings {
    tx: Sender<(string::Did, Box<post::Record>)>,
    count: Arc<AtomicUsize>,
}

impl Embeddings {
    pub async fn try_new<S>(query: S, threshold: f32) -> anyhow::Result<Self>
    where
        S: ToString,
    {
        let thread_count = thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(2);

        let bert = Arc::new(Bert::new().await?);

        // ModernBERT fails with "Error: Failed to load config: missing field `hidden_act` at line 44 column 1"
        // config.json uses `hidden_activation` instead of `hidden_act`
        #[cfg(any())]
        let bert = Arc::new(
            Bert::builder()
                .with_source(
                    BertSource::default()
                        .with_model(FileSource::huggingface(
                            "answerdotai/ModernBERT-base".to_string(),
                            "8949b909ec900327062f0ebf497f51aef5e6f0c8".to_string(),
                            "model.safetensors".to_string(),
                        ))
                        .with_tokenizer(FileSource::huggingface(
                            "answerdotai/ModernBERT-base".to_string(),
                            "8949b909ec900327062f0ebf497f51aef5e6f0c8".to_string(),
                            "tokenizer.json".to_string(),
                        ))
                        .with_config(FileSource::huggingface(
                            "answerdotai/ModernBERT-base".to_string(),
                            "8949b909ec900327062f0ebf497f51aef5e6f0c8".to_string(),
                            "config.json".to_string(),
                        )),
                )
                .build()
                .await?,
        );
        let query = Arc::new(bert.embed(query).await?);
        let post_count = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = flume::unbounded::<(string::Did, Box<post::Record>)>();

        for _ in 0..thread_count {
            let bert = bert.clone();
            let query = query.clone();
            let post_count = post_count.clone();
            let rx = rx.clone();
            thread::spawn(move || {
                while let Ok((did, post)) = rx.recv() {
                    if let Ok(embedding) = bert.embed_with_pooling(&post.text, Pooling::CLS) {
                        if embedding.cosine_similarity(&query) > threshold {
                            println!("{}:\n{}", did.as_str(), post.text);
                        }
                        post_count.fetch_add(1, Ordering::Relaxed);
                    }
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
