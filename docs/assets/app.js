/* naive-inference documentation — shared behavior and interactive widgets.
   Vanilla JS, no build step. Each widget guards on element presence, so the
   same file serves every page. */
"use strict";

/* ---------- theme ---------- */
(function () {
  const saved = localStorage.getItem("ni-theme");
  if (saved) document.documentElement.dataset.theme = saved;
  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("theme-toggle");
    if (!btn) return;
    const label = () => (document.documentElement.dataset.theme === "light" ? "🌙 dark" : "☀️ light");
    btn.textContent = label();
    btn.addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
      document.documentElement.dataset.theme = next;
      localStorage.setItem("ni-theme", next);
      btn.textContent = label();
    });
  });
})();

/* ---------- nav active state + TOC + scrollspy ---------- */
document.addEventListener("DOMContentLoaded", () => {
  const page = location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".topnav nav a").forEach((a) => {
    if (a.getAttribute("href") === page) a.classList.add("active");
  });

  const toc = document.getElementById("toc");
  if (toc) {
    const heads = document.querySelectorAll("main h2[id], main h3[id]");
    heads.forEach((h) => {
      const a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.textContent;
      if (h.tagName === "H3") a.classList.add("h3");
      toc.appendChild(a);
    });
    const links = toc.querySelectorAll("a");
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            links.forEach((l) => l.classList.toggle("current", l.getAttribute("href") === "#" + e.target.id));
          }
        });
      },
      { rootMargin: "-80px 0px -70% 0px" }
    );
    heads.forEach((h) => obs.observe(h));
  }

  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  }
  if (window.hljs) hljs.highlightAll();

  initRope();
  initAttention();
  initDelta();
  initBpe();
  initBf16();
  initSampling();
  initMemory();
  initLayers();
});

/* ---------- helpers ---------- */
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const fmt = (x, d = 3) => (Math.abs(x) < 1e-12 ? "0" : x.toFixed(d));
function el(id) { return document.getElementById(id); }

/* ---------- 1. RoPE visualizer ---------- */
function initRope() {
  const cv = el("rope-canvas");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const posS = el("rope-pos"), freqS = el("rope-freq"), out = el("rope-readout");
  const THETA = 1e7, DROT = 64;
  function draw() {
    const pos = +posS.value, i = +freqS.value;
    const invf = Math.pow(THETA, (-2 * i) / DROT);
    const angle = pos * invf;
    const W = cv.width, H = cv.height, cx = W / 2, cy = H / 2, R = 120;
    const css = getComputedStyle(document.documentElement);
    const dim = css.getPropertyValue("--fg-dim").trim() || "#888";
    const acc = css.getPropertyValue("--accent").trim() || "#5eead4";
    const acc2 = css.getPropertyValue("--accent2").trim() || "#818cf8";
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = dim; ctx.lineWidth = 1; ctx.globalAlpha = 0.5;
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx - R - 15, cy); ctx.lineTo(cx + R + 15, cy);
    ctx.moveTo(cx, cy - R - 15); ctx.lineTo(cx, cy + R + 15); ctx.stroke();
    ctx.globalAlpha = 1;
    // angle arc
    ctx.strokeStyle = acc2; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(cx, cy, 34, 0, -angle, angle > 0); ctx.stroke();
    // base vector (1, 0)
    ctx.strokeStyle = dim; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + R, cy); ctx.stroke();
    // rotated vector
    const x = Math.cos(angle), y = Math.sin(angle);
    ctx.strokeStyle = acc; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + R * x, cy - R * y); ctx.stroke();
    ctx.fillStyle = acc;
    ctx.beginPath(); ctx.arc(cx + R * x, cy - R * y, 5, 0, 2 * Math.PI); ctx.fill();
    out.textContent =
      `pos = ${pos}   freq index i = ${i}\n` +
      `inv_freq[i] = 1e7^(-2·${i}/64) = ${invf.toExponential(4)}\n` +
      `angle = pos · inv_freq = ${fmt(angle, 5)} rad\n` +
      `cos = ${fmt(x, 5)}   sin = ${fmt(y, 5)}\n` +
      `rotated pair: (x₁·cos − x₂·sin,  x₂·cos + x₁·sin)`;
    el("rope-pos-val").textContent = pos;
    el("rope-freq-val").textContent = i;
  }
  posS.addEventListener("input", draw);
  freqS.addEventListener("input", draw);
  draw();
}

/* ---------- 2. causal attention heatmap ---------- */
function initAttention() {
  const host = el("attn-grid");
  if (!host) return;
  const N = 6, D = 4;
  const rng = mulberry32(7);
  const q = [], k = [];
  for (let t = 0; t < N; t++) {
    q.push(Array.from({ length: D }, () => rng() * 2 - 1));
    k.push(Array.from({ length: D }, () => rng() * 2 - 1));
  }
  const scores = [];
  for (let i = 0; i < N; i++) {
    scores.push([]);
    for (let j = 0; j < N; j++) {
      if (j > i) { scores[i].push(null); continue; }
      let dot = 0;
      for (let d = 0; d < D; d++) dot += q[i][d] * k[j][d];
      scores[i].push(dot / Math.sqrt(D));
    }
  }
  const probs = scores.map((row) => {
    const vals = row.filter((v) => v !== null);
    const m = Math.max(...vals);
    const ex = vals.map((v) => Math.exp(v - m));
    const s = ex.reduce((a, b) => a + b, 0);
    let idx = 0;
    return row.map((v) => (v === null ? null : ex[idx++] / s));
  });
  const toggle = el("attn-toggle");
  function render() {
    const showProbs = toggle.checked;
    host.innerHTML = "";
    host.style.gridTemplateColumns = `40px repeat(${N}, 46px)`;
    host.appendChild(Object.assign(document.createElement("div"), { className: "attn-label", textContent: "q\\k" }));
    for (let j = 0; j < N; j++)
      host.appendChild(Object.assign(document.createElement("div"), { className: "attn-label", textContent: "k" + j }));
    for (let i = 0; i < N; i++) {
      host.appendChild(Object.assign(document.createElement("div"), { className: "attn-label", textContent: "q" + i }));
      for (let j = 0; j < N; j++) {
        const cell = document.createElement("div");
        cell.className = "attn-cell";
        const v = showProbs ? probs[i][j] : scores[i][j];
        if (v === null) {
          cell.style.background = "var(--bg3)";
          cell.style.color = "var(--fg-dim)";
          cell.textContent = "−∞";
          cell.title = `masked: token ${i} cannot see future token ${j}`;
        } else {
          const t = showProbs ? v : Math.min(1, Math.max(0, (v + 1.5) / 3));
          const r = Math.round(30 + 190 * t), g = Math.round(40 + 60 * (1 - t)), b = Math.round(120 + 100 * (1 - t));
          cell.style.background = `rgb(${r},${g},${b})`;
          cell.textContent = fmt(v, 2);
          cell.title = showProbs
            ? `softmax weight: token ${i} attends to token ${j} with ${fmt(v * 100, 1)}%`
            : `raw score q${i}·k${j}/√d = ${fmt(v, 3)}`;
        }
        host.appendChild(cell);
      }
    }
  }
  toggle.addEventListener("change", render);
  render();
}

/* ---------- 3. gated delta rule stepper ---------- */
function initDelta() {
  const host = el("delta-widget");
  if (!host) return;
  const D = 4;
  const TOKENS = [
    { k: [1, 0, 0, 0], v: [2, 0, 1, 0], beta: 0.9, g: -0.05 },
    { k: [0, 1, 0, 0], v: [0, 3, 0, 1], beta: 0.8, g: -0.1 },
    { k: [1, 0, 0, 0], v: [5, 0, 0, 0], beta: 0.9, g: -0.02 }, // same key as t0: correction!
  ];
  let S, tok, phase, kv, delta, outv, logEntries;
  const grid = el("delta-state"), log = el("delta-log"), readout = el("delta-readout");
  const vecHost = el("delta-vectors");

  function reset() {
    S = Array.from({ length: D }, () => Array(D).fill(0));
    tok = 0; phase = 0; kv = delta = outv = null; logEntries = [];
    render();
  }
  function vecRow(label, vec, hot) {
    const row = document.createElement("div");
    row.className = "vec-row";
    row.innerHTML = `<span class="vec-label">${label}</span>`;
    (vec || Array(D).fill(null)).forEach((v) => {
      const c = document.createElement("div");
      c.className = "vec-cell";
      c.textContent = v === null ? "·" : fmt(v, 2);
      if (hot) c.style.borderColor = "var(--accent)";
      row.appendChild(c);
    });
    return row;
  }
  function render() {
    grid.innerHTML = "";
    const maxAbs = Math.max(0.01, ...S.flat().map(Math.abs));
    for (let i = 0; i < D; i++)
      for (let j = 0; j < D; j++) {
        const c = document.createElement("div");
        c.className = "state-cell";
        const t = Math.abs(S[i][j]) / maxAbs;
        c.style.background = `color-mix(in srgb, var(--accent2) ${Math.round(t * 55)}%, var(--bg3))`;
        c.textContent = fmt(S[i][j], 2);
        c.title = `S[${i}][${j}] — key dim ${i}, value dim ${j}`;
        grid.appendChild(c);
      }
    const t = TOKENS[Math.min(tok, TOKENS.length - 1)];
    vecHost.innerHTML = "";
    vecHost.appendChild(vecRow("k (key)", t.k));
    vecHost.appendChild(vecRow("v (value)", t.v));
    vecHost.appendChild(vecRow("kᵀS (read)", kv, phase === 1));
    vecHost.appendChild(vecRow("δ (delta)", delta, phase === 2));
    vecHost.appendChild(vecRow("qᵀS (out)", outv, phase === 4));
    readout.textContent =
      `token ${Math.min(tok + 1, TOKENS.length)}/${TOKENS.length}   β = ${t.beta}   g = ${t.g}   decay e^g = ${fmt(Math.exp(t.g), 4)}\n` +
      `phase: ${["decay S ← e^g·S", "read kv = kᵀS", "delta δ = β·(v − kv)", "write S ← S + k·δᵀ", "output o = qᵀS (q = k here)"][phase]}`;
    log.innerHTML = "";
    logEntries.forEach((e, i) => {
      const d = document.createElement("div");
      d.className = "entry" + (i === logEntries.length - 1 ? " active" : "");
      d.innerHTML = e;
      log.appendChild(d);
    });
    log.scrollTop = log.scrollHeight;
    el("delta-step").disabled = tok >= TOKENS.length;
  }
  function step() {
    if (tok >= TOKENS.length) return;
    const t = TOKENS[tok];
    if (phase === 0) {
      const dec = Math.exp(t.g);
      for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) S[i][j] *= dec;
      logEntries.push(`<b>t${tok} decay:</b> every S entry × e^${t.g} = ${fmt(dec, 4)} — old memories fade.`);
      phase = 1;
    } else if (phase === 1) {
      kv = Array(D).fill(0);
      for (let j = 0; j < D; j++) for (let i = 0; i < D; i++) kv[j] += t.k[i] * S[i][j];
      logEntries.push(`<b>t${tok} read:</b> kv = kᵀS = [${kv.map((x) => fmt(x, 2)).join(", ")}] — what the state already predicts for this key.`);
      phase = 2;
    } else if (phase === 2) {
      delta = t.v.map((v, j) => t.beta * (v - kv[j]));
      logEntries.push(`<b>t${tok} delta:</b> δ = β·(v − kv) = [${delta.map((x) => fmt(x, 2)).join(", ")}] — only the <i>correction</i> is written.`);
      phase = 3;
    } else if (phase === 3) {
      for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) S[i][j] += t.k[i] * delta[j];
      logEntries.push(`<b>t${tok} write:</b> S ← S + k·δᵀ (outer product). Token 3 reuses key [1,0,0,0]: watch it <i>overwrite</i> the old value instead of adding to it.`);
      phase = 4;
    } else {
      outv = Array(D).fill(0);
      for (let j = 0; j < D; j++) for (let i = 0; i < D; i++) outv[j] += t.k[i] * S[i][j];
      logEntries.push(`<b>t${tok} output:</b> o = qᵀS = [${outv.map((x) => fmt(x, 2)).join(", ")}] — reads the <i>updated</i> state (q = k in this demo).`);
      tok++; phase = 0; kv = delta = outv = null;
      if (tok >= TOKENS.length) logEntries.push(`<b>done.</b> Final S is the whole "memory" of this layer — constant size, no matter how many tokens came before.`);
    }
    render();
  }
  el("delta-step").addEventListener("click", step);
  el("delta-reset").addEventListener("click", reset);
  reset();
}

/* ---------- 4. BPE merge demo ---------- */
function initBpe() {
  const host = el("bpe-widget");
  if (!host) return;
  const MERGES = [["t", "h"], ["th", "e"], ["Ġ", "c"], ["Ġc", "a"], ["Ġca", "t"], ["Ġ", "s"], ["Ġs", "a"], ["Ġsa", "t"]];
  const rank = new Map(MERGES.map((m, i) => [m[0] + " " + m[1], i]));
  const input = el("bpe-input"), steps = el("bpe-steps");
  function byteEncode(s) {
    // GPT-2 byte table for the printable subset used here; space -> Ġ.
    return [...s].map((ch) => (ch === " " ? "Ġ" : ch)).join("");
  }
  function run() {
    const text = input.value || "the cat sat";
    steps.innerHTML = "";
    // crude pretokenization for the demo: split before spaces, keep the space.
    const pretoks = text.match(/ ?[^ ]+/g) || [];
    pretoks.forEach((pt) => {
      let syms = [...byteEncode(pt)];
      const header = document.createElement("div");
      header.className = "bpe-step";
      header.innerHTML = `<span class="bpe-rank">pretoken ${JSON.stringify(pt)} →</span>` + syms.map((s) => `<span class="bpe-sym">${s}</span>`).join("");
      steps.appendChild(header);
      for (;;) {
        let best = null, bestR = Infinity;
        for (let i = 0; i + 1 < syms.length; i++) {
          const r = rank.get(syms[i] + " " + syms[i + 1]);
          if (r !== undefined && r < bestR) { bestR = r; best = i; }
        }
        if (best === null) break;
        const merged = syms[best] + syms[best + 1];
        syms = [...syms.slice(0, best), merged, ...syms.slice(best + 2)];
        const row = document.createElement("div");
        row.className = "bpe-step";
        row.innerHTML =
          `<span class="bpe-rank"></span>` +
          syms.map((s, i) => `<span class="bpe-sym${i === best ? " merged" : ""}">${s}</span>`).join("");
        row.querySelector(".bpe-rank").textContent = `rank ${bestR} → merge "${merged}"`;
        steps.appendChild(row);
      }
      const fin = document.createElement("div");
      fin.className = "bpe-step";
      fin.innerHTML = `<span class="bpe-rank">final pieces:</span>` + syms.map((s) => `<span class="bpe-sym merged">${s}</span>`).join("");
      steps.appendChild(fin);
    });
  }
  el("bpe-run").addEventListener("click", run);
  run();
}

/* ---------- 5. bf16 bit explorer ---------- */
function initBf16() {
  const host = el("bf16-widget");
  if (!host) return;
  const input = el("bf16-input"), f32Host = el("bf16-f32"), bf16Host = el("bf16-bf16"), out = el("bf16-readout");
  const buf = new DataView(new ArrayBuffer(4));
  function bitsHTML(bits, n, droppedFrom) {
    let h = "";
    for (let i = n - 1; i >= 0; i--) {
      const b = (bits >>> i) & 1;
      const cls = i === n - 1 ? "sign" : i >= n - 9 ? "exp" : "man";
      const dropped = droppedFrom !== undefined && i < droppedFrom ? " dropped" : "";
      h += `<div class="bit ${cls}${dropped}">${b}</div>`;
    }
    return h;
  }
  function run() {
    const v = parseFloat(input.value);
    if (Number.isNaN(v)) { out.textContent = "enter a number"; return; }
    buf.setFloat32(0, v);
    const bits = buf.getUint32(0);
    // round-to-nearest-even to bf16 (same formula as arrays.fp32_to_bf16)
    const round = 0x7fff + ((bits >>> 16) & 1);
    const bf = (bits + round) >>> 16;
    buf.setUint32(0, bf << 16);
    const back = buf.getFloat32(0);
    f32Host.innerHTML = bitsHTML(bits, 32, 16);
    bf16Host.innerHTML = bitsHTML(bf, 16);
    out.textContent =
      `fp32: ${v}  (0x${bits.toString(16).padStart(8, "0")})\n` +
      `bf16: 0x${bf.toString(16).padStart(4, "0")} → back to fp32: ${back}\n` +
      `absolute error: ${Math.abs(back - v).toExponential(3)}\n` +
      `bf16 keeps the top 16 bits: 1 sign + 8 exponent + 7 mantissa (fp32 has 23).\n` +
      `Conversion is a shift — the engine fuses it into every matmul.`;
  }
  input.addEventListener("input", run);
  run();
}

/* ---------- 6. sampling playground ---------- */
function initSampling() {
  const host = el("sampling-widget");
  if (!host) return;
  const N = 5;
  const sliders = [], bars = el("sampling-bars"), out = el("sampling-readout");
  const defaults = [2, 1, 0.5, -1, -2];
  const ctrl = el("sampling-logits");
  for (let i = 0; i < N; i++) {
    const wrap = document.createElement("div");
    wrap.className = "control";
    wrap.innerHTML = `<label>logit ${i}: <span id="sv${i}">${defaults[i]}</span></label>`;
    const s = document.createElement("input");
    s.type = "range"; s.min = -5; s.max = 5; s.step = 0.1; s.value = defaults[i];
    s.addEventListener("input", () => { el("sv" + i).textContent = s.value; run(); });
    wrap.appendChild(s);
    ctrl.appendChild(wrap);
    sliders.push(s);
  }
  const temp = el("sampling-temp"), topk = el("sampling-topk"), topp = el("sampling-topp");
  [temp, topk, topp].forEach((c) => c.addEventListener("input", run));
  let empirical = null;
  function probs() {
    const logits = sliders.map((s) => +s.value);
    const T = +temp.value;
    if (T <= 0.01) return logits.map((_, i) => (i === logits.indexOf(Math.max(...logits)) ? 1 : 0));
    let scaled = logits.map((v) => v / T);
    let order = scaled.map((_, i) => i).sort((a, b) => scaled[b] - scaled[a]);
    let keep = order.length;
    const K = +topk.value;
    if (K > 0 && K < keep) keep = K;
    const P = +topp.value;
    const soft = (n) => {
      const m = scaled[order[0]];
      const ex = order.slice(0, n).map((i) => Math.exp(scaled[i] - m));
      const s = ex.reduce((a, b) => a + b, 0);
      return ex.map((e) => e / s);
    };
    if (P > 0 && P < 1) {
      const ps = soft(keep);
      let cum = 0, n = 0;
      while (n < keep) { cum += ps[n]; n++; if (cum >= P) break; }
      keep = n;
    }
    const ps = soft(keep);
    const p = Array(N).fill(0);
    order.slice(0, keep).forEach((idx, i) => (p[idx] = ps[i]));
    return p;
  }
  function run() {
    const p = probs();
    bars.innerHTML = "";
    p.forEach((pi, i) => {
      const wrap = document.createElement("div");
      wrap.className = "prob-bar-wrap";
      const bar = document.createElement("div");
      bar.className = "prob-bar";
      bar.style.height = Math.max(2, pi * 140) + "px";
      bar.style.position = "relative";
      if (empirical) {
        const e = document.createElement("div");
        e.className = "prob-bar empirical";
        e.style.height = Math.max(1, empirical[i] * 140) + "px";
        bar.appendChild(e);
      }
      const lab = document.createElement("div");
      lab.className = "prob-bar-label";
      lab.textContent = `t${i}\n${(pi * 100).toFixed(1)}%`;
      lab.style.whiteSpace = "pre";
      wrap.appendChild(bar);
      wrap.appendChild(lab);
      bars.appendChild(wrap);
    });
    el("sampling-temp-val").textContent = (+temp.value).toFixed(2);
    el("sampling-topp-val").textContent = (+topp.value).toFixed(2);
    out.textContent =
      `temperature ${temp.value} · top-k ${topk.value === "0" ? "off" : topk.value} · top-p ${topp.value}\n` +
      `purple = model probabilities${empirical ? " · teal overlay = 2000 empirical draws" : ""}\n` +
      `temperature → 0 approaches greedy (argmax); high temperature flattens the distribution.`;
  }
  el("sampling-draw").addEventListener("click", () => {
    const p = probs();
    const counts = Array(N).fill(0);
    const cum = [];
    let acc = 0;
    p.forEach((pi) => { acc += pi; cum.push(acc); });
    for (let n = 0; n < 2000; n++) {
      const r = Math.random();
      let i = 0;
      while (i < N - 1 && r >= cum[i]) i++;
      counts[i]++;
    }
    empirical = counts.map((c) => c / 2000);
    run();
  });
  el("sampling-clear").addEventListener("click", () => { empirical = null; run(); });
  run();
}

/* ---------- 7. memory calculator ---------- */
function initMemory() {
  const cv = el("mem-canvas");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const modelSel = el("mem-model"), ctxS = el("mem-ctx"), dtypeSel = el("mem-dtype"), out = el("mem-readout");
  const MODELS = {
    "4B": { layers: 32, full: 8, linear: 24, kvHeads: 4, headDim: 256, vHeads: 32, weightsGB: 9.3 },
    "27B": { layers: 64, full: 16, linear: 48, kvHeads: 4, headDim: 256, vHeads: 48, weightsGB: 55.6 },
  };
  function compute() {
    const m = MODELS[modelSel.value];
    const bytes = dtypeSel.value === "fp32" ? 4 : 2;
    const ctxLen = +ctxS.value * 1024;
    const kvPerToken = 2 * m.kvHeads * m.headDim * bytes * m.full; // K+V, all full layers
    const kvGB = (kvPerToken * ctxLen) / 1e9;
    const ssmGB = (m.vHeads * 128 * 128 * 4 * m.linear) / 1e9; // fp32 always
    const convGB = (m.linear * (2 * 16 * 128 + m.vHeads * 128) * 4 * 4) / 1e9;
    return { m, ctxLen, kvPerToken, kvGB, ssmGB, convGB };
  }
  function draw() {
    const { m, ctxLen, kvPerToken, kvGB, ssmGB, convGB } = compute();
    const W = cv.width, H = cv.height, padL = 52, padB = 30, padT = 14, padR = 14;
    const css = getComputedStyle(document.documentElement);
    const dim = css.getPropertyValue("--fg-dim").trim() || "#888";
    const acc = css.getPropertyValue("--accent").trim() || "#5eead4";
    const acc2 = css.getPropertyValue("--accent2").trim() || "#818cf8";
    const maxCtx = 262144;
    const maxY = Math.max(1, (kvPerToken * maxCtx) / 1e9) * 1.05;
    const X = (c) => padL + (Math.log2(c) - Math.log2(1024)) / (Math.log2(maxCtx) - Math.log2(1024)) * (W - padL - padR);
    const Y = (gb) => H - padB - (gb / maxY) * (H - padB - padT);
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = dim; ctx.fillStyle = dim; ctx.font = "11px Consolas, monospace"; ctx.globalAlpha = 0.6;
    [1024, 8192, 32768, 131072, 262144].forEach((c) => {
      ctx.beginPath(); ctx.moveTo(X(c), padT); ctx.lineTo(X(c), H - padB); ctx.stroke();
      ctx.fillText(c >= 1024 ? c / 1024 + "k" : c, X(c) - 10, H - 10);
    });
    for (let g = 0; g <= maxY; g += Math.ceil(maxY / 4)) {
      ctx.beginPath(); ctx.moveTo(padL, Y(g)); ctx.lineTo(W - padR, Y(g)); ctx.stroke();
      ctx.fillText(g + " GB", 6, Y(g) + 4);
    }
    ctx.globalAlpha = 1;
    // KV line (linear in context)
    ctx.strokeStyle = acc2; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(X(1024), Y((kvPerToken * 1024) / 1e9)); ctx.lineTo(X(maxCtx), Y((kvPerToken * maxCtx) / 1e9)); ctx.stroke();
    // SSM constant line
    ctx.strokeStyle = acc; ctx.lineWidth = 2.5; ctx.setLineDash([7, 5]);
    ctx.beginPath(); ctx.moveTo(padL, Y(ssmGB + convGB)); ctx.lineTo(W - padR, Y(ssmGB + convGB)); ctx.stroke();
    ctx.setLineDash([]);
    // current point
    ctx.fillStyle = "#fbbf24";
    ctx.beginPath(); ctx.arc(X(ctxLen), Y(kvGB), 5, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = acc2; ctx.fillText("KV cache (full attention)", padL + 8, padT + 12);
    ctx.fillStyle = acc; ctx.fillText("linear-attn state (constant)", padL + 8, padT + 28);
    out.textContent =
      `${modelSel.value} · context ${ctxS.value}k · cache dtype ${dtypeSel.value}\n` +
      `KV cache: ${(kvPerToken / 1024).toFixed(1)} KB/token → ${kvGB.toFixed(2)} GB at ${ctxS.value}k\n` +
      `SSM state: ${ssmGB.toFixed(2)} GB + conv ${(convGB * 1024).toFixed(1)} MB — CONSTANT at any context\n` +
      `weights (bf16): ~${m.weightsGB} GB\n` +
      `At 262k context the KV cache would be ${((kvPerToken * 262144) / 1e9).toFixed(1)} GB — the hybrid design keeps the rest flat.`;
    el("mem-ctx-val").textContent = ctxS.value + "k";
  }
  [modelSel, ctxS, dtypeSel].forEach((c) => c.addEventListener("input", draw));
  draw();
}

/* ---------- 8. layer pattern explorer ---------- */
function initLayers() {
  const host = el("layer-grid");
  if (!host) return;
  const sel = el("layer-model"), info = el("layer-info");
  const CFG = {
    "4B": { n: 32, hidden: 2560, inter: 9216, qHeads: 16, vHeads: 32 },
    "27B": { n: 64, hidden: 5120, inter: 17408, qHeads: 24, vHeads: 48 },
  };
  function build() {
    const c = CFG[sel.value];
    host.innerHTML = "";
    host.style.gridTemplateColumns = `repeat(${c.n === 32 ? 16 : 16}, 1fr)`;
    for (let i = 0; i < c.n; i++) {
      const full = i % 4 === 3;
      const cell = document.createElement("div");
      cell.className = "layer-cell " + (full ? "full" : "linear");
      cell.textContent = i;
      cell.title = `layer ${i}: ${full ? "full attention" : "linear attention"}`;
      cell.addEventListener("click", () => {
        host.querySelectorAll(".layer-cell").forEach((x) => x.classList.remove("selected"));
        cell.classList.add("selected");
        info.innerHTML = full
          ? `<b>Layer ${i} — full attention</b> <span class="badge full">GQA + KV cache</span><br>
             Q heads: ${c.qHeads} · KV heads: 4 · head_dim 256 (RoPE on first 64)<br>
             q_proj is double-width (query + sigmoid gate); q_norm/k_norm per head.<br>
             State: KV cache growing with context — 2·4·256·4B = 8 KB/token (fp32).<br>
             Code: <code>forward_full_layer</code> in naive_infer/forward.py`
          : `<b>Layer ${i} — linear attention (Gated DeltaNet)</b> <span class="badge linear">recurrent state</span><br>
             K heads: 16×128 · V heads: ${c.vHeads}×128 · conv kernel 4 (depthwise, SiLU).<br>
             beta = sigmoid(b); g = −exp(A_log)·softplus(a + dt_bias); fp32 delta rule.<br>
             State: S [${c.vHeads},128,128] fp32 ≈ ${(c.vHeads * 128 * 128 * 4 / 1e6).toFixed(1)} MB — constant forever.<br>
             Code: <code>forward_linear_layer</code> in naive_infer/forward.py`;
      });
      host.appendChild(cell);
    }
    info.innerHTML = `Click a layer. Pattern: 3× linear attention + 1× full attention, repeated ${c.n / 4} times. Both share the same SwiGLU MLP (hidden ${c.hidden} → ${c.inter}).`;
  }
  sel.addEventListener("change", build);
  build();
}