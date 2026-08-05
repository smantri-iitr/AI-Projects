// The built-in topic catalogue. `query` is Google News search syntax, so OR
// groups and "quoted phrases" both work.

// Queries are deliberately anchored on subject-specific terms: broad single
// words like "technology" or "sports" match any article that merely mentions
// them (university names, press releases) and drown out the real headlines.
export const TOPICS = [
  { id: 'ai', label: 'AI', emoji: '🤖', query: '"artificial intelligence" OR "machine learning" OR OpenAI OR Anthropic' },
  { id: 'politics', label: 'Politics', emoji: '🏛️', query: 'politics OR election OR congress OR parliament OR "prime minister" OR president' },
  { id: 'india', label: 'India', emoji: '🇮🇳', query: 'India', edition: 'IN' },
  { id: 'tech', label: 'Tech', emoji: '💻', query: '"tech industry" OR "big tech" OR smartphone OR semiconductor OR software OR chipmaker' },
  { id: 'business', label: 'Business', emoji: '📈', query: '"stock market" OR economy OR earnings OR inflation OR "central bank" OR merger' },
  { id: 'startups', label: 'Startups', emoji: '🚀', query: 'startup OR "venture capital" OR "funding round" OR "Series A" OR IPO' },
  { id: 'science', label: 'Science', emoji: '🔬', query: '"new study" OR researchers OR scientists OR "peer-reviewed"' },
  { id: 'space', label: 'Space', emoji: '🛰️', query: 'NASA OR SpaceX OR ISRO OR "space mission" OR satellite OR astronaut' },
  { id: 'health', label: 'Health', emoji: '🩺', query: 'health OR medicine OR FDA OR vaccine OR "public health" OR hospital' },
  { id: 'climate', label: 'Climate', emoji: '🌍', query: '"climate change" OR "global warming" OR emissions OR "renewable energy"' },
  { id: 'crypto', label: 'Crypto', emoji: '₿', query: 'cryptocurrency OR bitcoin OR ethereum OR blockchain OR stablecoin' },
  { id: 'sports', label: 'Sports', emoji: '🏅', query: '"world cup" OR olympics OR NBA OR NFL OR cricket OR "premier league" OR tennis' },
  { id: 'entertainment', label: 'Entertainment', emoji: '🎬', query: '"box office" OR Hollywood OR Netflix OR "new album" OR streaming OR celebrity' },
  { id: 'world', label: 'World', emoji: '🌐', query: 'geopolitics OR diplomacy OR "foreign policy" OR ceasefire OR "United Nations"' },
];

export const TOPICS_BY_ID = new Map(TOPICS.map((topic) => [topic.id, topic]));

// Google News edition codes: hl = interface language, gl = country, ceid = both.
export const EDITIONS = {
  US: { hl: 'en-US', gl: 'US', ceid: 'US:en', label: 'United States' },
  IN: { hl: 'en-IN', gl: 'IN', ceid: 'IN:en', label: 'India' },
  GB: { hl: 'en-GB', gl: 'GB', ceid: 'GB:en', label: 'United Kingdom' },
  AU: { hl: 'en-AU', gl: 'AU', ceid: 'AU:en', label: 'Australia' },
  CA: { hl: 'en-CA', gl: 'CA', ceid: 'CA:en', label: 'Canada' },
  SG: { hl: 'en-SG', gl: 'SG', ceid: 'SG:en', label: 'Singapore' },
};

export const DEFAULT_EDITION = 'US';
