// Minimal RSS 2.0 parser. Google News feeds are simple and well-formed, so a
// regex reader is enough and keeps the app dependency-free.

const NAMED_ENTITIES = {
  amp: '&',
  lt: '<',
  gt: '>',
  quot: '"',
  apos: "'",
  nbsp: ' ',
};

export function decodeEntities(text) {
  return text.replace(/&(#x?[0-9a-fA-F]+|[a-zA-Z]+);/g, (match, entity) => {
    if (entity[0] === '#') {
      const hex = entity[1] === 'x' || entity[1] === 'X';
      const code = parseInt(hex ? entity.slice(2) : entity.slice(1), hex ? 16 : 10);
      return Number.isFinite(code) && code > 0 ? String.fromCodePoint(code) : match;
    }
    const named = NAMED_ENTITIES[entity.toLowerCase()];
    return named === undefined ? match : named;
  });
}

function stripCdata(value) {
  const cdata = value.match(/^\s*<!\[CDATA\[([\s\S]*?)\]\]>\s*$/);
  return cdata ? cdata[1] : value;
}

function readTag(xml, name) {
  const match = xml.match(new RegExp(`<${name}(?:\\s[^>]*)?>([\\s\\S]*?)</${name}>`, 'i'));
  return match ? decodeEntities(stripCdata(match[1])).trim() : '';
}

function readAttr(xml, name, attr) {
  const match = xml.match(new RegExp(`<${name}\\s[^>]*${attr}="([^"]*)"`, 'i'));
  return match ? decodeEntities(match[1]) : '';
}

// Google News puts the outlet in the headline as "Real headline - The Outlet".
// Trim it so titles stay readable next to the source badge.
function splitSourceSuffix(title, source) {
  if (!source) return title;
  const suffix = ` - ${source}`;
  return title.endsWith(suffix) ? title.slice(0, -suffix.length).trim() : title;
}

const KNOWN_ACRONYMS = new Set([
  'espn', 'nasa', 'fda', 'cdc', 'bbc', 'cnn', 'abc', 'nbc', 'cbs', 'pbs', 'afp', 'ap',
  'axios', 'ndtv', 'wsj', 'nyt', 'ft', 'nih', 'who', 'imf', 'nato', 'sec', 'irs',
]);

// Some feed entries name the outlet by bare domain ("reuters.com"). Turn those
// into something presentable: drop the public suffix, keep the last label.
export function prettySource(name) {
  if (!/^[a-z0-9.-]+\.[a-z]{2,}$/i.test(name) || name.includes(' ')) return name;
  const bare = name
    .toLowerCase()
    .replace(/^www\./, '')
    .replace(/\.(co|com|org|net|gov|edu|ac)\.[a-z]{2}$/, '')
    .replace(/\.[a-z]{2,}$/, '');
  const label = bare.split('.').pop() || bare;

  // Call letters and initialisms read wrong in title case ("Npr", "Wsj").
  if (KNOWN_ACRONYMS.has(label) || (label.length <= 5 && !/[aeiou]/.test(label))) {
    return label.toUpperCase();
  }
  return label.charAt(0).toUpperCase() + label.slice(1);
}

export function parseFeed(xml) {
  const items = [];
  for (const chunk of xml.split('<item>').slice(1)) {
    const block = chunk.split('</item>')[0];
    const link = readTag(block, 'link');
    if (!link) continue;

    const source = readTag(block, 'source');
    const pubDate = readTag(block, 'pubDate');
    const publishedAt = pubDate ? new Date(pubDate) : null;

    items.push({
      title: splitSourceSuffix(readTag(block, 'title'), source),
      link,
      source: source ? prettySource(source) : 'Unknown',
      sourceUrl: readAttr(block, 'source', 'url'),
      publishedAt: publishedAt && !Number.isNaN(publishedAt.getTime()) ? publishedAt : null,
    });
  }
  return items;
}
