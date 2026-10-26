(function () {
  'use strict';
  const english = __FEEDWATCH_TRANSLATIONS__;
  if (window.FeedWatchI18n) {
    window.FeedWatchI18n.addTranslations(english);
    window.FeedWatchI18n.apply();
    return;
  }
  const storageKey = 'feedwatch-language';
  let language = 'en';
  try {
    const saved = localStorage.getItem(storageKey);
    if (saved === 'en' || saved === 'pt') language = saved;
  } catch {}
  const listeners = new Set();
  let templates = null;
  function expandNumericTemplate(source) {
    if (!templates) templates = Object.entries(english).filter(([key]) => /\{\w+\}/.test(key)).map(([key, target]) => {
      const fields = [];
      const expression = key.split(/(\{\w+\})/).map(piece => {
        if (/^\{\w+\}$/.test(piece)) { fields.push(piece.slice(1, -1)); return '([-+]?\\d[\\d.,]*%?)'; }
        return piece.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      }).join('');
      return { pattern: new RegExp('^' + expression + '$'), fields, target };
    });
    for (const entry of templates) {
      const match = source.match(entry.pattern);
      if (!match) continue;
      const values = {};
      entry.fields.forEach((field, index) => {
        let number = match[index + 1];
        if (number.includes(',')) number = number.replaceAll('.', '').replace(',', '.');
        else if (/^\d{1,3}(\.\d{3})+$/.test(number)) number = number.replaceAll('.', ',');
        values[field] = number;
      });
      return entry.target.replace(/\{(\w+)\}/g, (token, key) => values[key] ?? token);
    }
    return source;
  }
  function t(source, values = {}) {
    if (source == null) return '';
    const text = String(source);
    const translated = language === 'pt' ? text : (english[text] ?? expandNumericTemplate(text));
    return translated.replace(/\{(\w+)\}/g, (match, key) =>
      Object.hasOwn(values, key) ? String(values[key]) : match);
  }
  function number(value, minimumFractionDigits = 0, maximumFractionDigits = 1) {
    return Number(value).toLocaleString(language === 'pt' ? 'pt-BR' : 'en-US',
      { minimumFractionDigits, maximumFractionDigits });
  }
  function nodes(root, selector) {
    const result = [...root.querySelectorAll(selector)];
    if (root.matches?.(selector)) result.unshift(root);
    return result;
  }
  function graphPath(source) {
    return language === 'en' ? source.replace(/(?<!\.en)\.svg(?=\s|,|$)/g, '.en.svg') : source;
  }
  function apply(root = document) {
    for (const node of nodes(root, '[data-fw-i18n]')) {
      const value = t(node.getAttribute('data-fw-i18n'));
      if (node.textContent !== value) node.textContent = value;
    }
    for (const node of nodes(root, '[data-fw-i18n-attrs]')) {
      const attributes = JSON.parse(node.getAttribute('data-fw-i18n-attrs'));
      for (const [key, source] of Object.entries(attributes)) {
        const value = t(source);
        if (node.getAttribute(key) !== value) node.setAttribute(key, value);
      }
    }
    for (const node of nodes(root, '[data-fw-graph-pt]')) {
      const value = graphPath(node.getAttribute('data-fw-graph-pt'));
      if (node.getAttribute('src') !== value) node.setAttribute('src', value);
    }
    for (const node of nodes(root, '[data-fw-graph-srcset-pt]')) {
      const value = graphPath(node.getAttribute('data-fw-graph-srcset-pt'));
      if (node.getAttribute('srcset') !== value) node.setAttribute('srcset', value);
    }
    for (const select of nodes(root, '[data-language-select]')) select.value = language;
    document.documentElement.lang = language === 'pt' ? 'pt-BR' : 'en';
  }
  function setLanguage(next) {
    if (next !== 'en' && next !== 'pt') return;
    language = next;
    try { localStorage.setItem(storageKey, language); } catch {}
    apply();
    for (const listener of [...listeners]) listener(language);
    document.dispatchEvent(new CustomEvent('feedwatch:languagechange', { detail: { language } }));
  }
  window.FeedWatchI18n = {
    get language() { return language; },
    get locale() { return language === 'pt' ? 'pt-BR' : 'en-US'; },
    t, number, apply, setLanguage,
    subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
    addTranslations(values) { Object.assign(english, values); templates = null; }
  };
  document.addEventListener('change', event => {
    if (event.target.matches?.('[data-language-select]')) setLanguage(event.target.value);
  });
  window.addEventListener('storage', event => {
    if (event.key === storageKey && (event.newValue === 'en' || event.newValue === 'pt')) {
      language = event.newValue;
      apply();
      for (const listener of [...listeners]) listener(language);
    }
  });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => apply(), { once: true });
  else apply();
}());
