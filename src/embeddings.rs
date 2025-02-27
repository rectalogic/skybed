use atrium_api::app::bsky::feed::post;
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
    tx: Sender<Box<post::Record>>,
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
        let query = Arc::new(bert.embed(query).await?);
        let post_count = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = flume::unbounded::<Box<post::Record>>();

        for _ in 0..thread_count {
            let bert = bert.clone();
            let query = query.clone();
            let post_count = post_count.clone();
            let rx = rx.clone();
            thread::spawn(move || {
                while let Ok(post) = rx.recv() {
                    if let Ok(embedding) = bert.embed_with_pooling(&post.text, Pooling::CLS) {
                        if embedding.cosine_similarity(&query) > threshold {
                            println!("{:?}", post.text);
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

    pub fn add_post(&self, post: Box<post::Record>) -> anyhow::Result<()> {
        self.tx.send(post)?;
        Ok(())
    }
}
