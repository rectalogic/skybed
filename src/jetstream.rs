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

    pub async fn recv(&mut self) -> anyhow::Result<(string::Did, Box<post::Record>)> {
        loop {
            let event = self.receiver.recv_async().await?;
            if let Commit(CommitEvent::Create { info, commit }) = event {
                if let AppBskyFeedPost(record) = commit.record {
                    if record.text.is_empty() {
                        continue;
                    }
                    match &record.langs {
                        Some(languages) => {
                            if languages.contains(&self.language) {
                                self.count += 1;
                                return Ok((info.did, record));
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
