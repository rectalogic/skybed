use std::mem::take;

use atrium_api::{
    app::bsky::feed::post,
    record::KnownRecord::AppBskyFeedPost,
    types::{Union, string},
};

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

    pub async fn recv(&mut self) -> anyhow::Result<PostData> {
        loop {
            let event = self.receiver.recv_async().await?;
            if let Commit(CommitEvent::Create { info, commit }) = event {
                if let AppBskyFeedPost(mut record) = commit.record {
                    if record.text.trim().len() < 10 {
                        continue;
                    }
                    match &record.langs {
                        Some(languages) => {
                            if languages.contains(&self.language) {
                                self.count += 1;
                                let text = if let Some(Union::Refs(
                                    post::RecordEmbedRefs::AppBskyEmbedExternalMain(ref main),
                                )) = record.embed
                                {
                                    format!(
                                        "{}\n{}\n{}",
                                        record.text,
                                        &main.external.data.title,
                                        &main.external.data.description
                                    )
                                } else {
                                    take(&mut record.text)
                                };
                                return Ok(PostData {
                                    did: info.did,
                                    rkey: commit.info.rkey,
                                    text,
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

pub struct PostData {
    did: string::Did,
    rkey: String,
    pub text: String,
}

impl PostData {
    pub fn url(&self) -> String {
        format!(
            "https://bsky.app/profile/{}/post/{}",
            self.did.as_str(),
            self.rkey
        )
    }
}
