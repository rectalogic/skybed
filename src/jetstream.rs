use atrium_api::{app::bsky::feed::post, record::KnownRecord::AppBskyFeedPost, types::string};

use jetstream_oxide::{
    JetstreamConfig, JetstreamConnector, JetstreamReceiver,
    events::{JetstreamEvent::Commit, commit::CommitEvent},
};

pub struct Jetstream {
    receiver: JetstreamReceiver,
    language: string::Language,
    count: usize,
}

pub struct PostInfo {
    pub did: string::Did,
    pub rkey: String,
    pub record: Box<post::Record>,
}

impl Jetstream {
    pub async fn connect(
        config: JetstreamConfig,
        language: string::Language,
    ) -> anyhow::Result<Self> {
        let connector = JetstreamConnector::new(config)?;
        Ok(Self {
            receiver: connector.connect().await?,
            language,
            count: 0,
        })
    }

    pub async fn recv(&mut self) -> anyhow::Result<PostInfo> {
        loop {
            let event = self.receiver.recv_async().await?;
            if let Commit(CommitEvent::Create { info, commit }) = event {
                if let AppBskyFeedPost(record) = commit.record {
                    if record.text.trim().len() < 10 {
                        continue;
                    }
                    match &record.langs {
                        Some(languages) => {
                            if languages.contains(&self.language) {
                                self.count += 1;
                                return Ok(PostInfo {
                                    did: info.did,
                                    rkey: commit.info.rkey,
                                    record,
                                });
                            } else {
                                continue;
                            }
                        }
                        None => continue,
                    }
                }
            }
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }
}
