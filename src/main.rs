use atrium_api::types::string::{self};
use clap::Parser;
use jetstream_oxide::{JetstreamCompression, JetstreamConfig};
use skybed::{Embeddings, Jetstream, LOG_COUNT};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The DIDs to listen for events on, if not provided we will listen for all DIDs.
    #[arg(short, long)]
    did: Option<Vec<string::Did>>,
    /// The NSID for the collection to listen for (e.g. `app.bsky.feed.post`).
    #[arg(short, long, default_value = "app.bsky.feed.post")]
    nsid: string::Nsid,
    /// The IETF language tag to use for the embeddings.
    #[arg(short, long, default_value = "en")]
    language: String,
    /// The threshold for the similarity score.
    #[arg(short, long, default_value = "0.6")]
    threshold: f32,
    /// The query to search for.
    #[arg()]
    query: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let language = string::Language::new(args.language).map_err(|e| anyhow::anyhow!("{}", e))?;
    let embeddings = Embeddings::try_new(args.query, args.threshold)?;

    let dids = args.did.unwrap_or_default();
    println!("Listening for '{:?}' events on DIDs: {:?}", args.nsid, dids);
    let config = JetstreamConfig {
        wanted_collections: vec![args.nsid.clone()],
        wanted_dids: dids,
        compression: JetstreamCompression::Zstd,
        ..JetstreamConfig::default()
    };
    let mut jetstream = Jetstream::connect(config, language).await?;
    while let Ok((did, record)) = jetstream.recv().await {
        if jetstream.count() % LOG_COUNT == 0 {
            eprintln!("{} posts", jetstream.count());
        }
        embeddings.add_post((did, record))?;
        let count = embeddings.count();
        if count % LOG_COUNT == 0 {
            eprintln!("{} embeddings", count);
        }
    }
    Ok(())
}
