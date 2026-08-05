const els = {
  form: document.getElementById('controls'),
  query: document.getElementById('query'),
  count: document.getElementById('count'),
  days: document.getElementById('days'),
  edition: document.getElementById('edition'),
  submit: document.getElementById('submit'),
  topics: document.getElementById('topics'),
  results: document.getElementById('results'),
  status: document.getElementById('status'),
  headlineN: document.getElementById('headline-n'),
  headlineDays: document.getElementById('headline-days'),
};

const selected = new Set();
let config = null;

/* --------------------------------------------------------------- helpers */

function timeAgo(iso) {
  if (!iso) return 'recently';
  const minutes = Math.round((Date.now() - new Date(iso)) / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  return days === 1 ? 'yesterday' : `${days}d ago`;
}

function el(tag, props = {}, children = []) {
  const node = Object.assign(document.createElement(tag), props);
  for (const child of children) node.append(child);
  return node;
}

function setStatus(message, isError = false) {
  els.status.textContent = message;
  els.status.classList.toggle('error', isError);
}

/** Mirror the current state into the URL so a view can be bookmarked/shared. */
function syncUrl(params) {
  history.replaceState(null, '', params.toString() ? `?${params}` : location.pathname);
}

/* ------------------------------------------------------------- rendering */

function renderTopics(topics) {
  els.topics.replaceChildren(
    ...topics.map((topic) => {
      const chip = el('button', {
        type: 'button',
        className: 'chip',
        textContent: `${topic.emoji} ${topic.label}`,
        title: `Toggle ${topic.label}`,
      });
      chip.setAttribute('aria-pressed', String(selected.has(topic.id)));
      chip.dataset.topic = topic.id;
      chip.addEventListener('click', () => {
        selected.has(topic.id) ? selected.delete(topic.id) : selected.add(topic.id);
        chip.setAttribute('aria-pressed', String(selected.has(topic.id)));
        load();
      });
      return chip;
    }),
  );
}

function renderSkeletons(n) {
  els.results.replaceChildren(
    ...Array.from({ length: Math.min(n, 6) }, () => el('li', { className: 'skeleton' })),
  );
}

function renderArticles(articles) {
  if (articles.length === 0) {
    els.results.replaceChildren(
      el('li', {
        className: 'empty',
        textContent: 'No stories matched. Try a wider window, a different topic, or another search.',
      }),
    );
    return;
  }

  els.results.replaceChildren(
    ...articles.map((article, index) => {
      const meta = [
        el('span', { className: 'source', textContent: article.source }),
        el('span', { className: 'dot', textContent: '·' }),
        el('span', { textContent: timeAgo(article.publishedAt) }),
      ];

      if (article.topicLabel) {
        meta.push(el('span', { className: 'badge', textContent: article.topicLabel }));
      }
      if (article.coverageCount > 1) {
        meta.push(
          el('span', {
            className: 'badge coverage',
            textContent: `${article.coverageCount} outlets`,
            title: 'Number of outlets covering this story — a signal of how big it is',
          }),
        );
      }

      const body = [
        el('a', {
          className: 'card-title',
          href: article.link,
          target: '_blank',
          rel: 'noopener noreferrer',
          textContent: article.title,
        }),
        el('div', { className: 'card-meta' }, meta),
      ];

      if (article.alsoCoveredBy?.length) {
        body.push(
          el('div', {
            className: 'also',
            textContent: `Also covered by ${article.alsoCoveredBy.join(', ')}`,
          }),
        );
      }

      return el('li', { className: 'card' }, [
        el('div', { className: 'rank', textContent: String(index + 1) }),
        el('div', { className: 'card-body' }, body),
      ]);
    }),
  );
}

/* ------------------------------------------------------------- data flow */

let requestToken = 0;

async function load() {
  const query = els.query.value.trim();
  const n = els.count.value;
  const days = els.days.value;
  const edition = els.edition.value;
  const topics = [...selected];

  els.headlineN.textContent = n;
  els.headlineDays.textContent = days;
  els.form.classList.toggle('searching', Boolean(query));

  const params = new URLSearchParams({ n, days });
  if (edition !== 'auto') params.set('edition', edition);
  if (query) params.set('q', query);
  else if (topics.length) params.set('topics', topics.join(','));

  syncUrl(params);

  if (!query && topics.length === 0) {
    els.results.replaceChildren();
    setStatus('Pick at least one topic, or search for something specific.');
    return;
  }

  const token = ++requestToken;
  els.submit.disabled = true;
  setStatus(query ? `Searching “${query}”…` : `Loading ${topics.length} topic${topics.length > 1 ? 's' : ''}…`);
  renderSkeletons(Number(n) || 10);

  try {
    const response = await fetch(`/api/news?${params}`);
    const data = await response.json();
    if (token !== requestToken) return; // a newer request already took over
    if (!response.ok) throw new Error(data.error ?? `Request failed (${response.status})`);

    renderArticles(data.articles);

    const scope = query ? `“${query}”` : data.meta.feeds.join(', ');
    let summary = `Top ${data.articles.length} of ${data.meta.candidates} stories · ${scope} · last ${data.meta.days} day${data.meta.days > 1 ? 's' : ''}`;
    if (data.meta.failed.length) {
      summary += ` · ${data.meta.failed.length} feed(s) unavailable`;
    }
    setStatus(summary);
  } catch (error) {
    if (token !== requestToken) return;
    els.results.replaceChildren();
    setStatus(error.message || 'Something went wrong fetching news.', true);
  } finally {
    if (token === requestToken) els.submit.disabled = false;
  }
}

const debounce = (fn, ms) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
};

/* ------------------------------------------------------------------ init */

async function init() {
  const response = await fetch('/api/topics');
  config = await response.json();

  els.edition.append(
    ...config.editions.map((edition) => el('option', { value: edition.id, textContent: edition.label })),
  );

  const url = new URLSearchParams(location.search);
  els.count.value = url.get('n') ?? config.defaults.n;
  els.days.value = url.get('days') ?? config.defaults.days;
  els.edition.value = url.get('edition') ?? 'auto';
  els.query.value = url.get('q') ?? '';

  const requested = (url.get('topics') ?? '').split(',').filter(Boolean);
  const valid = new Set(config.topics.map((topic) => topic.id));
  // Default view: the topics from the brief, so the page is useful on arrival.
  const initial = requested.filter((id) => valid.has(id));
  for (const id of initial.length ? initial : ['ai', 'politics', 'india', 'tech']) selected.add(id);

  renderTopics(config.topics);

  els.form.addEventListener('submit', (event) => {
    event.preventDefault();
    load();
  });

  els.query.addEventListener('input', debounce(load, 450));
  for (const input of [els.count, els.days]) input.addEventListener('change', load);
  els.edition.addEventListener('change', load);

  for (const button of document.querySelectorAll('[data-select]')) {
    button.addEventListener('click', () => {
      selected.clear();
      if (button.dataset.select === 'all') {
        for (const topic of config.topics) selected.add(topic.id);
      }
      for (const chip of els.topics.children) {
        chip.setAttribute('aria-pressed', String(selected.has(chip.dataset.topic)));
      }
      load();
    });
  }

  load();
}

init().catch((error) => setStatus(`Could not start: ${error.message}`, true));
