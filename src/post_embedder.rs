use flume::Sender;
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

use crate::embedder::Embedder;
use crate::jetstream::PostData;

pub struct PostEmbedder {
    tx: Sender<PostData>,
    count: Arc<AtomicUsize>,
}

impl PostEmbedder {
    pub fn try_new<S>(query: S, model: S, threshold: f32) -> anyhow::Result<Self>
    where
        S: AsRef<str> + Send + Sync,
    {
        let embedder = Arc::new(Embedder::try_new(model)?);

        let thread_count = thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(2);

        let query = Arc::new(embedder.embed(query.as_ref())?);
        let post_count = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = flume::unbounded::<PostData>();

        for _ in 0..thread_count {
            let embedder = embedder.clone();
            let query = query.clone();
            let post_count = post_count.clone();
            let rx = rx.clone();
            thread::spawn(move || {
                while let Ok(post) = rx.recv() {
                    match embedder.embed(&post.text) {
                        Ok(embedding) => match Embedder::cosine_similarity(&query, &embedding) {
                            Ok(similarity) => {
                                if similarity > threshold {
                                    println!("\n{}\n{}", post.text, post.url(),);
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

    pub fn add_post(&self, post: PostData) -> anyhow::Result<()> {
        self.tx.send(post)?;
        Ok(())
    }
}
