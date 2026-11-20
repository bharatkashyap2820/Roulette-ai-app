/* Roulette AI plugin — browser-only TensorFlow.js demo.
 * A fair RNG produces independent outcomes. This model does NOT forecast it reliably.
 * No casino connection, scraping, exploitation, or network upload of results.
 */
(function () {
  'use strict';
  const POCKETS = ['0', '00', ...Array.from({ length: 36 }, (_, i) => String(i + 1))];
  const WINDOW = 8;
  const MIN_RESULTS = 50; // educational demo threshold; too little for statistical inference
  const MODEL_KEY = 'roulette-ai-v2-tfjs';
  const META_KEY = 'rai_ai_model_meta';
  const TFJS_CDN = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
  let model = null;
  let libraryPromise = null;
  let meta = null;

  function isValidHistory(history) {
    return Array.isArray(history) && history.every(n => POCKETS.includes(String(n)));
  }
  function fingerprint(history) {
    const content = history.join(',');
    let hash = 2166136261;
    for (let i = 0; i < content.length; i++) {
      hash ^= content.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    return (hash >>> 0).toString(16);
  }
  function toFeatures(history) {
    const out = Array(WINDOW * POCKETS.length).fill(0);
    const last = history.slice(-WINDOW);
    for (let j = 0; j < last.length; j++) {
      const pocketIndex = POCKETS.indexOf(String(last[j]));
      if (pocketIndex >= 0) out[(WINDOW - last.length + j) * POCKETS.length + pocketIndex] = 1;
    }
    return out;
  }
  function toData(history) {
    const x = [], y = [];
    for (let i = WINDOW; i < history.length; i++) {
      x.push(toFeatures(history.slice(i - WINDOW, i)));
      const row = Array(POCKETS.length).fill(0);
      row[POCKETS.indexOf(String(history[i]))] = 1;
      y.push(row);
    }
    return { x, y };
  }
  function rankBaseline(history) {
    const recent = history.slice(-200);
    const counts = Object.fromEntries(POCKETS.map(p => [p, 0]));
    recent.forEach((p, i) => { counts[String(p)] += 0.5 + 0.5 * (i + 1) / recent.length; });
    return [...POCKETS].sort((a, b) => counts[b] - counts[a] || POCKETS.indexOf(a) - POCKETS.indexOf(b)).slice(0, 5);
  }
  function getMeta() {
    try { const m = JSON.parse(localStorage.getItem(META_KEY) || 'null'); return m && typeof m === 'object' ? m : null; }
    catch { return null; }
  }
  function loadTensorFlow() {
    if (window.tf) return Promise.resolve(window.tf);
    if (libraryPromise) return libraryPromise;
    libraryPromise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = TFJS_CDN;
      script.async = true;
      script.onload = () => window.tf ? resolve(window.tf) : reject(new Error('TensorFlow.js did not initialize.'));
      script.onerror = () => reject(new Error('Cannot load TensorFlow.js. Check your internet connection.'));
      document.head.appendChild(script);
    }).catch(err => { libraryPromise = null; throw err; });
    return libraryPromise;
  }
  function getModel() { return model; }
  async function train(history, onProgress = () => {}) {
    if (!isValidHistory(history)) throw new Error('Invalid history.');
    if (history.length < MIN_RESULTS) throw new Error(`Save at least ${MIN_RESULTS} results to try the neural-network demo.`);
    const tf = await loadTensorFlow();
    await tf.ready();
    const data = toData(history);
    const split = Math.floor(data.x.length * 0.8);
    if (split < 25 || data.x.length - split < 8) throw new Error('Not enough samples for a chronological holdout.');
    const tensors = [];
    let candidate = null;
    try {
      const tx = tf.tensor2d(data.x.slice(0, split)); tensors.push(tx);
      const ty = tf.tensor2d(data.y.slice(0, split)); tensors.push(ty);
      const vx = tf.tensor2d(data.x.slice(split)); tensors.push(vx);
      const vy = tf.tensor2d(data.y.slice(split)); tensors.push(vy);
      candidate = tf.sequential();
      candidate.add(tf.layers.dense({ inputShape: [WINDOW * POCKETS.length], units: 48,
        activation: 'relu', kernelRegularizer: tf.regularizers.l2({ l2: 0.005 }) }));
      candidate.add(tf.layers.dropout({ rate: 0.25 }));
      candidate.add(tf.layers.dense({ units: 38, activation: 'softmax' }));
      candidate.compile({ optimizer: tf.train.adam(0.002), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
      onProgress('Training neural network on earlier spins…');
      await candidate.fit(tx, ty, {
        epochs: 16, batchSize: 16, shuffle: false, validationData: [vx, vy],
        callbacks: { onEpochEnd: (epoch) => { onProgress(`Training: epoch ${epoch + 1}/16`); } }
      });
      const scores = tf.tidy(() => candidate.predict(vx).arraySync());
      let aiHits = 0, baselineHits = 0;
      scores.forEach((probabilities, offset) => {
        const order = probabilities.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p).slice(0, 5);
        const outcomeIndex = POCKETS.indexOf(String(history[WINDOW + split + offset]));
        if (order.some(entry => entry.i === outcomeIndex)) aiHits++;
        if (rankBaseline(history.slice(0, WINDOW + split + offset)).includes(String(history[WINDOW + split + offset]))) baselineHits++;
      });
      // Explicit training-bound history signature prevents restored models being applied to replaced data.
      const newMeta = {
        trainedOn: history.length,
        historyHash: fingerprint(history),
        valCount: scores.length,
        aiHits, baselineHits,
        trainedAt: new Date().toISOString(),
        note: 'Forward-in-time holdout, exploratory only; small samples can mislead.'
      };
      try { await tf.io.removeModel(`indexeddb://${MODEL_KEY}`).catch(() => {}); await candidate.save(`indexeddb://${MODEL_KEY}`); }
      catch { onProgress('Model trained; could not save to IndexedDB (train again after reload).'); }
      if (model) model.dispose();
      model = candidate;
      candidate = null;
      meta = newMeta;
      try { localStorage.setItem(META_KEY, JSON.stringify(newMeta)); } catch { /* private mode */ }
      return newMeta;
    } finally {
      tensors.forEach(t => t.dispose());
      if (candidate) candidate.dispose();
    }
  }
  async function restore(history) {
    const saved = getMeta();
    if (!saved || !Number.isInteger(saved.trainedOn) || history.length < saved.trainedOn ||
        fingerprint(history.slice(0, saved.trainedOn)) !== saved.historyHash) return null;
    try {
      const tf = await loadTensorFlow();
      await tf.ready();
      const loaded = await tf.loadLayersModel(`indexeddb://${MODEL_KEY}`);
      if (model) model.dispose();
      model = loaded;
      meta = saved;
      return saved;
    } catch { return null; }
  }
  function predict(history) {
    if (!model || !isValidHistory(history) || history.length < WINDOW) return null;
    const tf = window.tf;
    const probabilities = tf.tidy(() => {
      const input = tf.tensor2d([toFeatures(history)]);
      const output = model.predict(input);
      return Array.from(output.dataSync());
    });
    return probabilities.map((p, i) => ({ p, pocket: POCKETS[i] }))
      .sort((a, b) => b.p - a.p).slice(0, 5).map(x => x.pocket);
  }
  async function clear() {
    if (model) model.dispose();
    model = null;
    meta = null;
    try { localStorage.removeItem(META_KEY); } catch { /* private mode */ }
    if (window.tf) {
      try { await window.tf.io.removeModel(`indexeddb://${MODEL_KEY}`); } catch { /* absent or unsupported */ }
    }
  }
  window.RouletteAIPlugin = { MIN_RESULTS, train, restore, predict, clear, getModel, getMeta: () => meta, rankBaseline };
})();
