import { parseFeed } from './rss.js';
import { TOPICS_BY_ID, EDITIONS, DEFAULT_EDITION } from './topics.js';

const FEED_HOST = 'https://news.google.com/rss/search';
const USER_AGENT = 'Mozilla/5.0 (compatible; topNnews/1.0; +https://localhost)';
const CACHE_TTL_MS = 5 * 60 * 1000;
const FETCH_TIMEOUT_MS = 8_000;
const MAX_CONCURRENT_FEEDS = 5;

export const LIMITS = {
  n: { min: 1, max: 100, default: 10 },
  days: { min: 1, max: 90, default: 7 },
};

const cache = new Map();

function cached(key, produce) {
  const hit = cache.get(key);
  if (hit && Date.now() - hit.storedAt < CACHE_TTL_MS) return hit.value;

  const value = produce().catch((error) => {
    cache.delete(key); // never cache a failure
    throw error;
  });
  cache.set(key, { storedAt: Date.now(), value });
  return value;
}

function feedUrl(query, days, editionId) {
  const edition = EDITIONS[editionId] ?? EDITIONS[DEFAULT_EDITION];
  const url = new URL(FEED_HOST);
  // `when:Nd` asks Google News for the window; we re-filter by pubDate below
  // because the operator is a hint rather than a hard cut-off.
  url.searchParams.set('q', `${query} when:${days}d`);
  url.searchParams.set('hl', edition.hl);
  url.searchParams.set('gl', edition.gl);
  url.searchParams.set('ceid', edition.ceid);
  return url.toString();
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function fetchFeedOnce(url) {
  const response = await fetch(url, {
    headers: { 'User-Agent': USER_AGENT, Accept: 'application/rss+xml, application/xml' },
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
  });
  if (!response.ok) throw new Error(`Google News responded ${response.status}`);

  const items = parseFeed(await response.text());
  // Google occasionally answers 200 with an empty channel when it throttles.
  // Treat that as a failure so it is reported and retried instead of being
  // cached as a legitimately empty topic for the next five minutes.
  if (items.length === 0) throw new Error('Feed returned no items (likely throttled)');
  return items;
}

async function fetchFeed(url) {
  try {
    return await fetchFeedOnce(url);
  } catch (error) {
    // A timeout already cost the full budget; retrying would double the wait a
    // user sits through. Only retry the fast failures (throttled empty feed,
    // HTTP error), where a second attempt usually succeeds straight away.
    if (error?.name === 'TimeoutError' || error?.name === 'AbortError') throw error;
    await sleep(400);
    return fetchFeedOnce(url);
  }
}

/** Run tasks with a ceiling on in-flight requests, preserving input order. */
async function pooled(tasks, limit) {
  const results = new Array(tasks.length);
  let next = 0;

  const worker = async () => {
    while (next < tasks.length) {
      const index = next++;
      try {
        results[index] = { status: 'fulfilled', value: await tasks[index]() };
      } catch (reason) {
        results[index] = { status: 'rejected', reason };
      }
    }
  };

  await Promise.all(Array.from({ length: Math.min(limit, tasks.length) }, worker));
  return results;
}

/* ---------------------------------------------------------------- ranking */

const STOP_WORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'from', 'has', 'have', 'how',
  'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that', 'the', 'their', 'this', 'to', 'was', 'were',
  'will', 'with', 'after', 'over', 'new', 'says', 'said', 'amid', 'into', 'more', 'than', 'you',
]);

// Outlets whose coverage is a reasonable signal that a story is significant.
// Everything not listed keeps the neutral baseline of 0.5.
const SOURCE_WEIGHTS = new Map(Object.entries({
  'reuters': 1, 'associated press': 1, 'ap news': 1, 'bbc': 1, 'bbc news': 1,
  'the new york times': 0.95, 'the wall street journal': 0.95, 'financial times': 0.95,
  'the washington post': 0.9, 'bloomberg': 0.9, 'the guardian': 0.9, 'cnbc': 0.85,
  'cnn': 0.85, 'nbc news': 0.85, 'abc news': 0.85, 'npr': 0.85, 'axios': 0.8, 'politico': 0.8,
  'the economist': 0.9, 'the atlantic': 0.8, 'wired': 0.8, 'the verge': 0.8, 'techcrunch': 0.8,
  'ars technica': 0.8, 'nature': 0.95, 'science': 0.9, 'the hindu': 0.9, 'the times of india': 0.85,
  'hindustan times': 0.85, 'indian express': 0.85, 'the indian express': 0.85, 'ndtv': 0.8,
  'livemint': 0.8, 'mint': 0.8, 'business standard': 0.8, 'the economic times': 0.85,
  'al jazeera': 0.85, 'deutsche welle': 0.8, 'france 24': 0.8, 'sky news': 0.8,
}));

// Keyword feeds surface a lot of first-party material — agency press releases,
// investor-relations posts, university athletics departments. It is on-topic but
// it is not "top news", so it gets demoted rather than dropped.
const PRIMARY_SOURCE_PATTERNS = [
  /\(\.(gov|edu)\)/i,
  /\b(pr ?newswire|business ?wire|globenewswire|ein presswire|accesswire|prweb|newswire|newswise)\b/i,
  /\b(investor relations|newsroom|news network|media cent(er|re)|press release)\b/i,
  /\b(university|college|athletics|institute of technology|school of)\b/i,
  /\b(department of|ministry of|world health organization|centers for disease control)\b/i,
  // Companies publishing their own results: "Exxon Mobil Corporation",
  // "Arrowhead Pharmaceuticals, Inc." — newsrooms are not named this way.
  /\b(inc|corp|corporation|ltd|limited|plc|llc|holdings|pharmaceuticals|therapeutics|technologies)\.?$/i,
];

function isPrimarySource(source) {
  return PRIMARY_SOURCE_PATTERNS.some((pattern) => pattern.test(source));
}

function sourceWeight(source) {
  const normalized = source.trim().toLowerCase();
  if (SOURCE_WEIGHTS.has(normalized)) return SOURCE_WEIGHTS.get(normalized);
  return isPrimarySource(normalized) ? 0.15 : 0.5;
}

function titleTokens(title) {
  return new Set(
    title
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter((word) => word.length > 2 && !STOP_WORDS.has(word)),
  );
}

function jaccard(a, b) {
  let shared = 0;
  for (const token of a) if (b.has(token)) shared += 1;
  const union = a.size + b.size - shared;
  return union === 0 ? 0 : shared / union;
}

// Group near-identical headlines so the same story from six outlets occupies one
// slot — and so that breadth of coverage can act as an importance signal.
function clusterStories(articles) {
  const SIMILARITY_THRESHOLD = 0.45;
  const byToken = new Map();
  const clusters = [];

  for (const article of articles) {
    const tokens = titleTokens(article.title);

    const candidates = new Set();
    for (const token of tokens) {
      for (const index of byToken.get(token) ?? []) candidates.add(index);
    }

    let target = -1;
    let best = SIMILARITY_THRESHOLD;
    for (const index of candidates) {
      const score = jaccard(tokens, clusters[index].tokens);
      if (score >= best) {
        best = score;
        target = index;
      }
    }

    if (target === -1) {
      target = clusters.length;
      clusters.push({ tokens, members: [] });
    }
    clusters[target].members.push(article);

    for (const token of tokens) {
      if (!byToken.has(token)) byToken.set(token, []);
      byToken.get(token).push(target);
    }
  }

  return clusters;
}

function scoreCluster(cluster, { now, windowMs }) {
  // Lead article: the most prominent outlet, tie-broken by feed position.
  const members = [...cluster.members].sort(
    (a, b) => sourceWeight(b.source) - sourceWeight(a.source) || a.feedRank - b.feedRank,
  );
  const lead = members[0];

  // How high Google News itself ranked the best member (1 = top of the feed).
  const relevance = Math.max(...members.map((m) => m.feedRelevance));

  const newest = members.reduce(
    (max, m) => Math.max(max, m.publishedAt?.getTime() ?? 0),
    0,
  );
  const age = newest ? now - newest : windowMs;
  // Half-life at the midpoint of the requested window, so `days` genuinely
  // changes what "recent" means rather than only widening the pool.
  const recency = Math.exp((-Math.LN2 * age) / (windowMs / 2));

  const outlets = new Set(members.map((m) => m.source));
  const coverage = Math.min(1, Math.log2(1 + outlets.size) / 3);
  const prominence = sourceWeight(lead.source);

  let score = 0.4 * relevance + 0.3 * recency + 0.2 * coverage + 0.1 * prominence;

  // The lead is the most prominent outlet in the cluster, so if it is still a
  // first-party source then no newsroom picked the story up.
  if (isPrimarySource(lead.source)) score *= 0.6;

  return {
    ...lead,
    score: Math.round(score * 1000) / 1000,
    coverageCount: outlets.size,
    alsoCoveredBy: [...outlets].filter((name) => name !== lead.source).slice(0, 6),
    topics: [...new Set(members.map((m) => m.topic).filter(Boolean))],
  };
}

// A single busy topic (an election night, say) can otherwise take every slot.
// Pick greedily by score, but discount stories whose topic or outlet is already
// represented — the top story of each topic still wins its place early.
function selectDiverse(scored, count, { topicPenalty }) {
  const picked = [];
  const remaining = [...scored];
  const topicSeen = new Map();
  const sourceSeen = new Map();

  while (picked.length < count && remaining.length > 0) {
    let bestIndex = 0;
    let bestValue = -Infinity;

    for (let i = 0; i < remaining.length; i += 1) {
      const article = remaining[i];
      const topicHits = topicSeen.get(article.topic ?? '') ?? 0;
      const sourceHits = sourceSeen.get(article.source) ?? 0;
      const value = article.score * topicPenalty ** topicHits * 0.93 ** sourceHits;
      if (value > bestValue) {
        bestValue = value;
        bestIndex = i;
      }
    }

    const [chosen] = remaining.splice(bestIndex, 1);
    topicSeen.set(chosen.topic ?? '', (topicSeen.get(chosen.topic ?? '') ?? 0) + 1);
    sourceSeen.set(chosen.source, (sourceSeen.get(chosen.source) ?? 0) + 1);
    picked.push(chosen);
  }

  return picked;
}

/* ------------------------------------------------------------------- api */

export function clamp(value, { min, max, default: fallback }) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, parsed));
}

/**
 * @param {object} options
 * @param {string[]} options.topics   Topic ids from the catalogue.
 * @param {string}   options.query    Free-text search; used instead of topics when set.
 * @param {number}   options.n        How many stories to return.
 * @param {number}   options.days     Look-back window in days.
 * @param {string}   options.edition  Edition id, or 'auto' to follow each topic.
 */
export async function getTopNews({ topics = [], query = '', n, days, edition = 'auto' }) {
  const count = clamp(n, LIMITS.n);
  const window = clamp(days, LIMITS.days);

  const feeds = query.trim()
    ? [{ topic: null, label: query.trim(), query: query.trim(), edition }]
    : topics
        .map((id) => TOPICS_BY_ID.get(id))
        .filter(Boolean)
        .map((topic) => ({
          topic: topic.id,
          label: topic.label,
          query: topic.query,
          edition: edition === 'auto' ? topic.edition ?? DEFAULT_EDITION : edition,
        }));

  // Keep the response shape identical to the success case so clients never have
  // to special-case "nothing to ask for".
  if (feeds.length === 0) {
    return {
      articles: [],
      meta: {
        n: count,
        days: window,
        edition,
        feeds: [],
        candidates: 0,
        failed: [],
        generatedAt: new Date().toISOString(),
      },
    };
  }

  const resolvedEdition = (id) => (id === 'auto' || !EDITIONS[id] ? DEFAULT_EDITION : id);

  const results = await pooled(
    feeds.map((feed) => () => {
      const url = feedUrl(feed.query, window, resolvedEdition(feed.edition));
      return cached(url, () => fetchFeed(url)).then((items) => ({ feed, items }));
    }),
    MAX_CONCURRENT_FEEDS,
  );

  const now = Date.now();
  const windowMs = window * 24 * 60 * 60 * 1000;
  const cutoff = now - windowMs;
  const failed = [];
  const seenLinks = new Set();
  const articles = [];

  results.forEach((result, index) => {
    if (result.status === 'rejected') {
      failed.push({ topic: feeds[index].label, error: String(result.reason?.message ?? result.reason) });
      return;
    }
    const { feed, items } = result.value;
    items.forEach((item, position) => {
      // The `when:` operator is approximate — enforce the window ourselves.
      if (item.publishedAt && item.publishedAt.getTime() < cutoff) return;
      if (seenLinks.has(item.link)) return;
      seenLinks.add(item.link);

      articles.push({
        ...item,
        topic: feed.topic,
        topicLabel: feed.label,
        feedRank: position,
        feedRelevance: 1 - position / Math.max(items.length, 1),
      });
    });
  });

  const scored = clusterStories(articles)
    .map((cluster) => scoreCluster(cluster, { now, windowMs }))
    .sort((a, b) => b.score - a.score || (b.publishedAt ?? 0) - (a.publishedAt ?? 0));

  // With one feed there is no topic mix to protect, so only the outlet spread
  // matters; across several topics, share the slots out.
  const ranked = selectDiverse(scored, count, { topicPenalty: feeds.length > 1 ? 0.85 : 1 });

  return {
    articles: ranked.map((article) => ({
      title: article.title,
      link: article.link,
      source: article.source,
      sourceUrl: article.sourceUrl,
      publishedAt: article.publishedAt?.toISOString() ?? null,
      score: article.score,
      coverageCount: article.coverageCount,
      alsoCoveredBy: article.alsoCoveredBy,
      topic: article.topic,
      topicLabel: article.topicLabel,
      topics: article.topics,
    })),
    meta: {
      n: count,
      days: window,
      edition,
      feeds: feeds.map((feed) => feed.label),
      candidates: articles.length,
      failed,
      generatedAt: new Date(now).toISOString(),
    },
  };
}
