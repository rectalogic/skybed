Realtime BlueSky post [jetstream](https://docs.bsky.app/blog/jetstream) feed,
filtered on a query using embeddings similarity via a local LLM.

Uses [jetstream-oxide](https://crates.io/crates/jetstream-oxide) to consume jetstream.
Posts are farmed out to [candle-transformers](https://crates.io/crates/candle-transformers) using [flume](https://crates.io/crates/flume).

```
$ cargo run --release --  "US federal government layoffs"

Five years of always saying yes to 84-hour weeks for months on end. One terse two-minute 45-second phone call layoff, and not even a thank you.
https://bsky.app/profile/did:plc:yudnkybncmkbejdqwc54p64i/post/3ljijctirws2f

Trump and Musk want a shutdown: It will allow them to get Democrats to shut down federal agencies.

Is that right?
https://bsky.app/profile/did:plc:3qox3txkvcpvqcdc3mhvrlyx/post/3ljijcskbes2q

Meanwhile, there have been mass illegal firings of federal workers, cuts to cancer research, and the elimination of the agency meant to protect citizens from corporate price gouging. If this is “winning,” it sure doesn’t look like it.
(1/4)
https://bsky.app/profile/did:plc:m46ucbkis2fu6qfsm7xc74vn/post/3ljijd4bfgk2n
...
```

```sh-session
$  cargo run --release -- --help
Usage: skybedding [OPTIONS] <QUERY>

Arguments:
 <QUERY>  The query to search for

Options:
 -m, --model <MODEL>          The model to use for embedding [default: sentence-transformers/all-MiniLM-L6-v2]
 -d, --did <DID>              The DIDs to listen for events on, if not provided we will listen for all DIDs
 -n, --nsid <NSID>            The NSID for the collection to listen for (e.g. `app.bsky.feed.post`) [default: app.bsky.feed.post]
 -l, --language <LANGUAGE>    The IETF language tag to use for the embeddings [default: en]
 -t, --threshold <THRESHOLD>  The threshold for the similarity score [default: 0.3]
 -h, --help                   Print help
 -V, --version                Print version
 ```
