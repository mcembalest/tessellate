(function(){
  async function loadModel(url) {
    const res = await fetch(url, { cache: 'no-cache' });
    if (!res.ok) throw new Error('Failed to load model JSON');
    return await res.json();
  }

  function relu(v) { return v > 0 ? v : 0; }

  function matvec(W, x) {
    // W: [out, in], x: [in] -> y: [out]
    const out = new Array(W.length);
    for (let i = 0; i < W.length; i++) {
      const row = W[i];
      let s = 0;
      for (let j = 0; j < row.length; j++) s += row[j] * x[j];
      out[i] = s;
    }
    return out;
  }

  function addBias(v, b) {
    const y = new Array(v.length);
    for (let i = 0; i < v.length; i++) y[i] = v[i] + b[i];
    return y;
  }

  function layerNorm(x, gamma, beta, eps) {
    const n = x.length;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    let varsum = 0;
    for (let i = 0; i < n; i++) { const d = x[i] - mean; varsum += d * d; }
    const varr = varsum / n;
    const denom = 1.0 / Math.sqrt(varr + eps);
    const y = new Array(n);
    for (let i = 0; i < n; i++) {
      const xn = (x[i] - mean) * denom;
      y[i] = xn * gamma[i] + beta[i];
    }
    return y;
  }

  function argmaxMasked(q, valid) {
    let best = -Infinity, bestIdx = -1;
    const validSet = new Set(valid);
    for (let i = 0; i < q.length; i++) {
      if (!validSet.has(i)) continue;
      const v = q[i];
      if (v > best) { best = v; bestIdx = i; }
    }
    return bestIdx;
  }

  function forward(model, state) {
    const L0 = model.layers.linear0;
    const L1 = model.layers.linear1;
    const LN = model.layers.layernorm;
    const OUT = model.layers.output;
    let h = matvec(L0.weight, state); h = addBias(h, L0.bias); h = h.map(relu);
    h = matvec(L1.weight, h); h = addBias(h, L1.bias); h = h.map(relu);
    h = layerNorm(h, LN.gamma, LN.beta, LN.eps);
    let q = matvec(OUT.weight, h); q = addBias(q, OUT.bias);
    return q;
  }

  const PQN = {
    _model: null,
    async ensureLoaded(url) {
      if (!this._model) this._model = await loadModel(url);
      return this._model;
    },
    async selectAction(url, state, validActions) {
      const m = await this.ensureLoaded(url);
      const q = forward(m, state);
      return argmaxMasked(q, validActions);
    }
  };

  window.PQN = PQN;
})();
