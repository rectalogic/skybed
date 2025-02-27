use atrium_api::{app::bsky::feed::post, types::string};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use flume::Sender;
use ndarray::ArrayView1;
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
    pub fn try_new<S>(query: S, threshold: f32) -> anyhow::Result<Self>
    where
        S: AsRef<str> + Send + Sync,
    {
        let thread_count = thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(2);

        let model = Arc::new(TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::ModernBertEmbedLarge)
                .with_show_download_progress(true),
        )?);
        let query = Arc::new(model.embed(vec![query], None)?.into_iter().next().unwrap());
        let post_count = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = flume::unbounded::<(string::Did, Box<post::Record>)>();

        for _ in 0..thread_count {
            let model = model.clone();
            let query = query.clone();
            let post_count = post_count.clone();
            let rx = rx.clone();
            thread::spawn(move || {
                while let Ok((did, post)) = rx.recv() {
                    if let Ok(embedding_vec) = model.embed(vec![&post.text], None) {
                        if let Some(embedding) = embedding_vec.first() {
                            if cosine_similarity(&query, embedding) > threshold {
                                println!("{}:\n{}", did.as_str(), post.text);
                            }
                            post_count.fetch_add(1, Ordering::Relaxed);
                        }
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

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let a = ArrayView1::from(a);
    let b = ArrayView1::from(b);

    let dot_product = a.dot(&b);
    let norm_a = a.dot(&a).sqrt();
    let norm_b = b.dot(&b).sqrt();

    dot_product / (norm_a * norm_b)
}
