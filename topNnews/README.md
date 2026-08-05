# topNnews

Top **N** news stories across your topics from the last **T** days.

Pick topics (AI, Politics, India, Tech, …), set how many stories you want and how far
back to look, or search for any topic of your own. Zero dependencies — just Node 18+.

```bash
npm start          # http://localhost:3000
PORT=8080 npm start
```

## Inputs

| Input | Default | Range | Notes |
| --- | --- | --- | --- |
| `n` — how many stories | **10** | 1–100 | |
| `days` — look-back window | **7** | 1–90 | Enforced against each story's publish date |
| topics | AI, Politics, India, Tech | 14 built in | Multi-select |
| search | – | any text | Free-text search, replaces topic selection |
| edition | Auto | US, IN, GB, AU, CA, SG | "Auto" uses each topic's natural edition (India → IN) |

The current view is mirrored into the URL, so any result set can be bookmarked or shared.

## How "top" is decided

Headlines come from Google News RSS. Fetching more than you asked for (100 candidates per
topic) leaves room to actually rank them, rather than just returning whatever came first:

1. **Cluster** near-identical headlines so one story from six outlets takes one slot,
   using Jaccard similarity over de-stopworded title tokens.
2. **Score** each cluster:

   | Signal | Weight | Meaning |
   | --- | --- | --- |
   | Relevance | 0.40 | How high Google News itself ranked it |
   | Recency | 0.30 | Exponential decay, half-life at the midpoint of your window |
   | Coverage | 0.20 | How many distinct outlets carried it — the strongest "this is big" signal |
   | Prominence | 0.10 | Weighting for established newsrooms |

   Stories whose only source is first-party (`.gov` releases, investor-relations posts,
   university newsrooms, PR wires) are demoted — on-topic, but not *news*.
3. **Spread the slots** so a busy topic (an election night) can't take all ten;
   each topic's best story still places early. Repeated outlets are discounted too.

Because `days` sets the recency half-life, widening the window genuinely re-weights
toward significance instead of only enlarging the pool.

## API

```bash
curl 'localhost:3000/api/news?topics=ai,india&n=5&days=3'
curl 'localhost:3000/api/news?q=ISRO+Gaganyaan&n=5&days=30'
curl 'localhost:3000/api/topics'
```

`GET /api/news` — `topics` (csv), `q`, `n`, `days`, `edition`. Either `topics` or `q` is
required; `q` wins when both are given. Out-of-range numbers clamp, junk falls back to
defaults.

```jsonc
{
  "articles": [{
    "title": "...", "link": "...", "source": "Reuters",
    "publishedAt": "2026-08-05T04:35:02.000Z",
    "score": 0.846,           // 0–1, see above
    "coverageCount": 12,      // distinct outlets on this story
    "alsoCoveredBy": ["BBC", "CNN"],
    "topic": "ai", "topicLabel": "AI"
  }],
  "meta": { "n": 10, "days": 7, "feeds": ["AI"], "candidates": 100, "failed": [] }
}
```

A topic whose feed fails is reported in `meta.failed` while the others still return —
one bad feed never takes down the response.

## Layout

```
server.js         HTTP server, JSON API, static files
src/news.js       fetch → cluster → score → diversify
src/rss.js        minimal RSS parser, source-name tidying
src/topics.js     topic catalogue and editions
public/           UI (vanilla JS, light + dark)
```

Feed responses are cached in memory for 5 minutes, capped at 5 concurrent fetches.
Throttled feeds (Google answers `200` with an empty channel) are retried rather than
cached as an empty topic.

## Notes

Google News RSS needs no API key, but it is an undocumented endpoint that can rate-limit
or change shape. Topic queries are anchored on subject-specific phrases — a bare word like
`technology` matches any article mentioning it and buries the real headlines.
