/**
 * On-device inference: the Python pipeline (backend/masking_ita.py,
 * backend/clahe_calibration.py, the ablation notebook) ported to the browser.
 *
 *   photo -> K-means skin mask + ITA -> bracket beta -> L*-CLAHE -> YOLO26 (ONNX)
 *
 * OpenCV.js (same OpenCV 5.0 as Python) performs colour conversion, resizing
 * and CLAHE, so those steps match the training data exactly. K-means and the
 * silhouette score are re-implemented here; they follow scikit-learn's
 * algorithm (k-means++, Lloyd, n_init restarts) but use a different random
 * stream, so ITA can differ slightly from Python on ambiguous images.
 *
 * Models and parameters come from models/manifest.json
 * (backend/export_app_models.py).
 */
const OnDevice = (() => {
  // Phone photos are reduced before processing; ITA is a mean skin colour
  // and the detector works at 640 px.
  const MAX_SIDE = 1280;
  const ITA_MAX_SIDE = 256;

  let manifestPromise = null;
  let cvPromise = null;
  const sessions = {};

  function loadManifest() {
    if (!manifestPromise) {
      manifestPromise = fetch('models/manifest.json').then(r => {
        if (!r.ok) throw new Error('models/manifest.json not found');
        return r.json();
      });
    }
    return manifestPromise;
  }

  // The runtimes are large (OpenCV.js ~13 MB), so load them on first use.
  const scripts = {};
  function loadScript(src) {
    if (!scripts[src]) {
      scripts[src] = new Promise((resolve, reject) => {
        const el = document.createElement('script');
        el.charset = 'utf-8';  // opencv.js embeds its WebAssembly as UTF-8 text
        el.src = src;
        el.onload = resolve;
        el.onerror = () => reject(new Error(`Failed to load ${src}`));
        document.head.appendChild(el);
      });
    }
    return scripts[src];
  }

  function loadCv() {
    if (!cvPromise) {
      cvPromise = (async () => {
        if (!window.cv) await loadScript('vendor/opencv.js');
        let module = window.cv;
        if (!module) throw new Error('OpenCV.js failed to load');
        if (module instanceof Promise) module = await module;
        else if (!module.Mat) await new Promise(resolve => { module.onRuntimeInitialized = resolve; });
        return module;
      })();
    }
    return cvPromise;
  }

  async function getSession(file) {
    if (!window.ort) await loadScript('vendor/ort.wasm.min.js');
    if (!sessions[file]) {
      // Must be a full URL: the loader is imported as an ES module.
      ort.env.wasm.wasmPaths = new URL('vendor/', document.baseURI).href;
      ort.env.wasm.numThreads = 1;  // WebViews are not cross-origin isolated.
      sessions[file] = ort.InferenceSession.create(`models/${file}`, { executionProviders: ['wasm'] });
    }
    return sessions[file];
  }

  // ---------------------------------------------------------------- helpers

  // Python's round(): half to even (used by the Ultralytics letterbox).
  function pyRound(x) {
    const r = Math.round(x);
    return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
  }

  function mulberry32(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function sampleIndices(n, count, seed) {
    if (n <= count) return Array.from({ length: n }, (_, i) => i);
    const rand = mulberry32(seed);
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < count; i++) {  // partial Fisher-Yates
      const j = i + Math.floor(rand() * (n - i));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return idx.slice(0, count);
  }

  function resizeMax(cv, mat, maxSide) {
    const scale = Math.min(1, maxSide / Math.max(mat.rows, mat.cols));
    if (scale >= 1) return mat.clone();
    const out = new cv.Mat();
    cv.resize(mat, out, new cv.Size(Math.max(1, pyRound(mat.cols * scale)), Math.max(1, pyRound(mat.rows * scale))), 0, 0, cv.INTER_AREA);
    return out;
  }

  // RGB (CV_8UC3) -> corrected CIELAB as Float32Array [L*, a*, b*] per pixel.
  function correctedLab(cv, rgb) {
    const lab = new cv.Mat();
    cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);
    const src = lab.data;
    const out = new Float32Array(src.length);
    for (let i = 0; i < src.length; i += 3) {
      out[i] = src[i] * 100 / 255;
      out[i + 1] = src[i + 1] - 128;
      out[i + 2] = src[i + 2] - 128;
    }
    lab.delete();
    return out;
  }

  function uniqueCount(points, indices, limit) {
    const seen = new Set();
    for (const i of indices) {
      seen.add(`${points[3 * i]},${points[3 * i + 1]},${points[3 * i + 2]}`);
      if (seen.size >= limit) break;
    }
    return seen.size;
  }

  // ---------------------------------------------------------------- K-means

  // Lloyd's K-means with k-means++ seeding and n_init restarts (lowest inertia wins),
  // mirroring sklearn.cluster.KMeans(algorithm="lloyd", max_iter=300, tol=1e-4).
  function kmeans(points, n, k, nInit, seed) {
    const rand = mulberry32(seed);
    let variance = 0;
    for (let d = 0; d < 3; d++) {
      let mean = 0;
      for (let i = 0; i < n; i++) mean += points[3 * i + d];
      mean /= n;
      for (let i = 0; i < n; i++) variance += (points[3 * i + d] - mean) ** 2;
    }
    const tol = 1e-4 * variance / (3 * n);
    let best = null;
    const labels = new Int32Array(n);
    const dist = new Float64Array(n);

    for (let run = 0; run < nInit; run++) {
      // k-means++ (greedy, 2 + log(k) local trials, as scikit-learn)
      const centers = new Float64Array(3 * k);
      const first = Math.floor(rand() * n);
      for (let d = 0; d < 3; d++) centers[d] = points[3 * first + d];
      for (let i = 0; i < n; i++) {
        dist[i] = (points[3 * i] - centers[0]) ** 2 + (points[3 * i + 1] - centers[1]) ** 2 + (points[3 * i + 2] - centers[2]) ** 2;
      }
      const trials = 2 + Math.floor(Math.log(k));
      for (let c = 1; c < k; c++) {
        let total = 0;
        for (let i = 0; i < n; i++) total += dist[i];
        let bestCandidate = -1, bestPot = Infinity;
        for (let t = 0; t < trials; t++) {
          let r = rand() * total, candidate = 0;
          while (candidate < n - 1 && r > dist[candidate]) { r -= dist[candidate]; candidate++; }
          let pot = 0;
          for (let i = 0; i < n; i++) {
            const dd = (points[3 * i] - points[3 * candidate]) ** 2 + (points[3 * i + 1] - points[3 * candidate + 1]) ** 2 + (points[3 * i + 2] - points[3 * candidate + 2]) ** 2;
            pot += Math.min(dist[i], dd);
          }
          if (pot < bestPot) { bestPot = pot; bestCandidate = candidate; }
        }
        for (let d = 0; d < 3; d++) centers[3 * c + d] = points[3 * bestCandidate + d];
        for (let i = 0; i < n; i++) {
          const dd = (points[3 * i] - centers[3 * c]) ** 2 + (points[3 * i + 1] - centers[3 * c + 1]) ** 2 + (points[3 * i + 2] - centers[3 * c + 2]) ** 2;
          if (dd < dist[i]) dist[i] = dd;
        }
      }

      // Lloyd iterations
      let inertia = 0;
      for (let iter = 0; iter < 300; iter++) {
        inertia = 0;
        for (let i = 0; i < n; i++) {
          let bi = 0, bd = Infinity;
          for (let c = 0; c < k; c++) {
            const dd = (points[3 * i] - centers[3 * c]) ** 2 + (points[3 * i + 1] - centers[3 * c + 1]) ** 2 + (points[3 * i + 2] - centers[3 * c + 2]) ** 2;
            if (dd < bd) { bd = dd; bi = c; }
          }
          labels[i] = bi;
          inertia += bd;
        }
        const sums = new Float64Array(3 * k), counts = new Float64Array(k);
        for (let i = 0; i < n; i++) {
          const c = labels[i];
          counts[c]++;
          for (let d = 0; d < 3; d++) sums[3 * c + d] += points[3 * i + d];
        }
        let shift = 0;
        for (let c = 0; c < k; c++) {
          if (!counts[c]) continue;  // keep an empty cluster's centre
          for (let d = 0; d < 3; d++) {
            const v = sums[3 * c + d] / counts[c];
            shift += (v - centers[3 * c + d]) ** 2;
            centers[3 * c + d] = v;
          }
        }
        if (shift <= tol) break;
      }
      if (!best || inertia < best.inertia) best = { inertia, labels: labels.slice(), centers: centers.slice() };
    }
    return best;
  }

  function silhouette(points, indices, labels, k) {
    const m = indices.length;
    let total = 0;
    const sums = new Float64Array(k), counts = new Float64Array(k);
    for (let a = 0; a < m; a++) counts[labels[a]]++;
    for (let a = 0; a < m; a++) {
      sums.fill(0);
      const pa = 3 * indices[a];
      for (let b = 0; b < m; b++) {
        if (a === b) continue;
        const pb = 3 * indices[b];
        sums[labels[b]] += Math.sqrt((points[pa] - points[pb]) ** 2 + (points[pa + 1] - points[pb + 1]) ** 2 + (points[pa + 2] - points[pb + 2]) ** 2);
      }
      const own = labels[a];
      if (counts[own] <= 1) continue;  // sklearn: silhouette 0 for singletons
      const intra = sums[own] / (counts[own] - 1);
      let nearest = Infinity;
      for (let c = 0; c < k; c++) if (c !== own && counts[c]) nearest = Math.min(nearest, sums[c] / counts[c]);
      total += (nearest - intra) / Math.max(intra, nearest);
    }
    return total / m;
  }

  // ---------------------------------------------------------------- masking + ITA

  function rankK(points, n, cfg) {
    const sample = sampleIndices(n, cfg.max_k_selection_pixels, cfg.random_state);
    const feats = new Float32Array(3 * sample.length);
    sample.forEach((p, i) => { feats[3 * i] = points[3 * p]; feats[3 * i + 1] = points[3 * p + 1]; feats[3 * i + 2] = points[3 * p + 2]; });
    const all = Array.from({ length: sample.length }, (_, i) => i);
    const candidates = [];
    for (const k of cfg.k_values) {
      if (sample.length <= k || uniqueCount(feats, all, k) < k) continue;
      const model = kmeans(feats, sample.length, k, cfg.n_init, cfg.random_state + k);
      const scoreIdx = sampleIndices(sample.length, cfg.max_silhouette_samples, cfg.random_state + k);
      const scoreLabels = scoreIdx.map(i => model.labels[i]);
      if (new Set(scoreLabels).size < 2) continue;
      candidates.push({ k, score: silhouette(feats, scoreIdx, scoreLabels, k) });
    }
    candidates.sort((a, b) => b.score - a.score || a.k - b.k);
    if (candidates.length) return candidates;
    if (sample.length >= 3 && uniqueCount(feats, all, 3) >= 3) return [{ k: 3, score: null }];
    throw new Error('KMEANS_NO_VALID_K');
  }

  function clusterInfo(points, n, labels, k, width, height, cfg) {
    const out = [];
    for (let c = 0; c < k; c++) {
      let count = 0, sl = 0, sa = 0, sb = 0, ok = 0, shadow = 0, highlight = 0;
      for (let i = 0; i < n; i++) {
        if (labels[i] !== c) continue;
        const L = points[3 * i], A = points[3 * i + 1], B = points[3 * i + 2];
        count++; sl += L; sa += A; sb += B;
        if (L >= cfg.l_min && L <= cfg.l_max && A >= cfg.a_min && A <= cfg.a_max && B >= cfg.b_min && B <= cfg.b_max) ok++;
        if (L < cfg.shadow_l_threshold) shadow++;
        if (L > cfg.highlight_l_threshold) highlight++;
      }
      if (!count) continue;
      const mean = [sl / count, sa / count, sb / count];
      const area = 100 * count / n;
      const plausible = (
        mean[0] >= cfg.l_min && mean[0] <= cfg.l_max &&
        mean[1] >= cfg.a_min && mean[1] <= cfg.a_max &&
        mean[2] >= cfg.b_min && mean[2] <= cfg.b_max &&
        area >= cfg.mask_area_min_percent && area <= cfg.mask_area_max_percent &&
        count >= cfg.minimum_valid_pixel_count &&
        ok / count >= cfg.minimum_pixel_plausibility_fraction &&
        shadow / count <= cfg.maximum_shadow_fraction &&
        highlight / count <= cfg.maximum_highlight_fraction
      );
      out.push({ label: c, count, mean, plausible });
    }
    return out;
  }

  function attempt(cv, rgb, cfg) {
    const points = correctedLab(cv, rgb);
    const n = rgb.rows * rgb.cols;
    const candidates = rankK(points, n, cfg);
    for (const { k } of candidates) {
      if (uniqueCount(points, sampleIndices(n, 10000, cfg.random_state + 200 + k), k) < k) continue;
      const model = kmeans(points, n, k, cfg.n_init, cfg.random_state);
      const plausible = clusterInfo(points, n, model.labels, k, rgb.cols, rgb.rows, cfg).filter(c => c.plausible);
      if (!plausible.length) continue;
      const chosen = plausible.reduce((a, b) => (b.count > a.count || (b.count === a.count && b.label < a.label) ? b : a));
      const [meanL, , meanB] = chosen.mean;
      if (Math.abs(meanB) < cfg.minimum_abs_b) continue;
      return { ita: Math.atan((meanL - 50) / meanB) * 180 / Math.PI, k, meanL, meanB };
    }
    return null;
  }

  function bracketOf(ita) {
    if (ita < 28) return 'Darkest';
    if (ita <= 41) return 'Medium';
    return 'Lightest';
  }

  // Primary attempt, then the centred 60% x 60% crop, else MASK_FAILED (no ITA).
  function computeIta(cv, rgb, cfg) {
    const small = resizeMax(cv, rgb, ITA_MAX_SIDE);
    try {
      let result = attempt(cv, small, cfg);
      let status = 'MASK_SUCCESS';
      if (!result) {
        const w = Math.max(1, pyRound(small.cols * cfg.center_crop_width_ratio));
        const h = Math.max(1, pyRound(small.rows * cfg.center_crop_height_ratio));
        const crop = small.roi(new cv.Rect(Math.floor((small.cols - w) / 2), Math.floor((small.rows - h) / 2), w, h)).clone();
        result = attempt(cv, crop, cfg);
        crop.delete();
        status = 'MASK_FALLBACK';
      }
      if (!result) return { ita: null, bracket: null, status: 'MASK_FAILED' };
      return { ita: result.ita, bracket: bracketOf(result.ita), status, meanL: result.meanL, meanB: result.meanB, k: result.k };
    } finally {
      small.delete();
    }
  }

  // ---------------------------------------------------------------- preprocessing

  function lClahe(cv, rgb, beta, grid) {
    const lab = new cv.Mat();
    cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);
    const channels = new cv.MatVector();
    cv.split(lab, channels);
    const clahe = new cv.CLAHE(beta, new cv.Size(grid[0], grid[1]));
    const l = channels.get(0), enhancedL = new cv.Mat();
    clahe.apply(l, enhancedL);
    channels.set(0, enhancedL);
    cv.merge(channels, lab);
    const out = new cv.Mat();
    cv.cvtColor(lab, out, cv.COLOR_Lab2RGB);
    [lab, channels, l, enhancedL].forEach(m => m.delete());
    clahe.delete();
    return out;
  }

  function rgbClahe(cv, rgb, beta, grid) {
    const channels = new cv.MatVector();
    cv.split(rgb, channels);
    const clahe = new cv.CLAHE(beta, new cv.Size(grid[0], grid[1]));
    for (let i = 0; i < 3; i++) {
      const c = channels.get(i), e = new cv.Mat();
      clahe.apply(c, e);
      channels.set(i, e);
      c.delete(); e.delete();
    }
    const out = new cv.Mat();
    cv.merge(channels, out);
    channels.delete();
    clahe.delete();
    return out;
  }

  function betaForD(ita, calib) {
    if (ita === null) {
      const beta = calib.beta_fallback ?? calib.beta_global;
      return { beta, source: calib.beta_fallback !== undefined ? 'medium_bracket_fallback' : 'beta_global_fallback' };
    }
    const bracket = bracketOf(ita);
    const beta = bracket === 'Darkest' ? calib.beta_high : bracket === 'Medium' ? calib.beta_mid : calib.beta_low;
    return { beta, source: 'ita_bracket' };
  }

  function preprocess(cv, rgb, kind, manifest, itaResult) {
    const grid = manifest.tile_grid_size;
    const calib = manifest.calibration;
    switch (kind) {
      case 'raw': return { image: rgb.clone(), beta: null };
      case 'rgb_clahe_fixed': return { image: rgbClahe(cv, rgb, manifest.fixed_beta, grid), beta: manifest.fixed_beta };
      case 'l_clahe_fixed': return { image: lClahe(cv, rgb, manifest.fixed_beta, grid), beta: manifest.fixed_beta };
      case 'l_clahe_global': return { image: lClahe(cv, rgb, calib.beta_global, grid), beta: calib.beta_global };
      case 'l_clahe_ita': {
        const { beta } = betaForD(itaResult.ita, calib);
        return { image: lClahe(cv, rgb, beta, grid), beta };
      }
      default: throw new Error(`Unknown preprocessing: ${kind}`);
    }
  }

  // ---------------------------------------------------------------- YOLO

  // Ultralytics LetterBox (auto=False, center=True, pad 114, INTER_LINEAR).
  function letterbox(cv, rgb, size) {
    const r = Math.min(size / rgb.rows, size / rgb.cols);
    const newW = pyRound(rgb.cols * r), newH = pyRound(rgb.rows * r);
    const dw = (size - newW) / 2, dh = (size - newH) / 2;
    const resized = new cv.Mat();
    if (newW !== rgb.cols || newH !== rgb.rows) cv.resize(rgb, resized, new cv.Size(newW, newH), 0, 0, cv.INTER_LINEAR);
    else rgb.copyTo(resized);
    const top = pyRound(dh - 0.1), bottom = pyRound(dh + 0.1);
    const left = pyRound(dw - 0.1), right = pyRound(dw + 0.1);
    const padded = new cv.Mat();
    cv.copyMakeBorder(resized, padded, top, bottom, left, right, cv.BORDER_CONSTANT, new cv.Scalar(114, 114, 114, 255));
    resized.delete();
    const data = padded.data, plane = size * size;
    const tensor = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      tensor[i] = data[3 * i] / 255;
      tensor[plane + i] = data[3 * i + 1] / 255;
      tensor[2 * plane + i] = data[3 * i + 2] / 255;
    }
    padded.delete();
    return { tensor, r, left, top };
  }

  const camSpecs = {};
  function loadCamSpec(file) {
    if (!camSpecs[file]) {
      camSpecs[file] = fetch(`models/${file}`).then(r => {
        if (!r.ok) throw new Error(`models/${file} not found`);
        return r.json();
      });
    }
    return camSpecs[file];
  }

  // Returns { detections, cam }. cam (Float32Array, rgb.rows x rgb.cols, 0..1) is
  // computed when camSpec is given.
  async function detect(cv, rgb, modelFile, manifest, camSpec = null) {
    const session = await getSession(modelFile);
    const size = manifest.imgsz;
    const { tensor, r, left, top } = letterbox(cv, rgb, size);
    const outputs = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', tensor, [1, 3, size, size]) });
    const out = outputs[session.outputNames[0]].data;  // [1, 300, 6]: x1 y1 x2 y2 score class
    const detections = [];
    for (let i = 0; i < out.length; i += 6) {
      const score = out[i + 4];
      if (score < manifest.confidence) continue;
      const clamp = (v, max) => Math.min(Math.max(v, 0), max);
      const x1 = clamp((out[i] - left) / r, rgb.cols), y1 = clamp((out[i + 1] - top) / r, rgb.rows);
      const x2 = clamp((out[i + 2] - left) / r, rgb.cols), y2 = clamp((out[i + 3] - top) / r, rgb.rows);
      detections.push({
        label: manifest.classes[Math.round(out[i + 5])],
        confidence: Math.round(score * 10000) / 10000,
        box: [x1 / rgb.cols, y1 / rgb.rows, x2 / rgb.cols, y2 / rgb.rows].map(v => Math.round(v * 10000) / 10000),
      });
    }
    detections.sort((a, b) => b.confidence - a.confidence);
    const cam = camSpec && outputs[camSpec.layers[0].feature]
      ? gradCam(cv, outputs, camSpec, manifest, { r, left, top, rows: rgb.rows, cols: rgb.cols })
      : null;
    return { detections, cam };
  }

  // ---------------------------------------------------------------- Grad-CAM

  // Grad-CAM at the last feature layer of the one-to-one class head (64 channels
  // per scale). That layer feeds a linear 1x1 convolution, so d(logit_c)/d(feature_k)
  // is the weight W[c][k]: the gradient is exact, no backpropagation needed.
  // Targets: detections with sigmoid(logit) >= confidence (up to max_targets), else
  // the single strongest; per scale, alpha_k = mean gradient, cam = ReLU(sum alpha_k A_k),
  // normalized, upsampled to 640, averaged over scales, then mapped to the photo.
  function gradCam(cv, outputs, camSpec, manifest, geo) {
    const settings = manifest.gradcam || { confidence: 0.25, max_targets: 10 };
    const size = manifest.imgsz;
    const layers = camSpec.layers.map(layer => {
      const f = outputs[layer.feature], l = outputs[layer.logits];
      return { spec: layer, feat: f.data, logits: l.data, C: f.dims[1], nc: l.dims[1], H: f.dims[2], W: f.dims[3] };
    });
    const candidates = [];
    let best = null;
    layers.forEach((L, j) => {
      const HW = L.H * L.W;
      for (let p = 0; p < HW; p++) {
        let c = 0, v = L.logits[p];
        for (let k = 1; k < L.nc; k++) {
          const x = L.logits[k * HW + p];
          if (x > v) { v = x; c = k; }
        }
        const prob = 1 / (1 + Math.exp(-v));
        const cand = { j, p, c, prob };
        if (prob >= settings.confidence) candidates.push(cand);
        if (!best || prob > best.prob) best = cand;
      }
    });
    candidates.sort((a, b) => b.prob - a.prob);
    const targets = candidates.length ? candidates.slice(0, settings.max_targets) : [best];

    const total = new cv.Mat(size, size, cv.CV_32F, new cv.Scalar(0));
    layers.forEach((L, j) => {
      const HW = L.H * L.W;
      const alpha = new Float32Array(L.C);
      targets.filter(t => t.j === j).forEach(t => {
        const w = L.spec.weight[t.c];
        for (let k = 0; k < L.C; k++) alpha[k] += w[k] / HW;
      });
      const cam = new Float32Array(HW);
      let max = 0;
      for (let p = 0; p < HW; p++) {
        let v = 0;
        for (let k = 0; k < L.C; k++) v += alpha[k] * L.feat[k * HW + p];
        cam[p] = v > 0 ? v : 0;
        if (cam[p] > max) max = cam[p];
      }
      if (max > 0) for (let p = 0; p < HW; p++) cam[p] /= max;
      const small = cv.matFromArray(L.H, L.W, cv.CV_32F, cam);
      const up = new cv.Mat();
      cv.resize(small, up, new cv.Size(size, size), 0, 0, cv.INTER_LINEAR);
      cv.add(total, up, total);
      small.delete(); up.delete();
    });
    const newW = pyRound(geo.cols * geo.r), newH = pyRound(geo.rows * geo.r);
    const crop = total.roi(new cv.Rect(geo.left, geo.top, newW, newH));
    const full = new cv.Mat();
    cv.resize(crop, full, new cv.Size(geo.cols, geo.rows), 0, 0, cv.INTER_LINEAR);
    crop.delete(); total.delete();
    const data = Float32Array.from(full.data32F);
    full.delete();
    let max = 0;
    for (let i = 0; i < data.length; i++) if (data[i] > max) max = data[i];
    if (max > 0) for (let i = 0; i < data.length; i++) data[i] /= max;
    // All 8 class scores (sigmoid) of the strongest lesion candidate, for the disease split.
    const main = targets[0], ML = layers[main.j], mHW = ML.H * ML.W;
    const classScores = manifest.classes.map((label, k) => {
      const z = ML.logits[k * mHW + main.p];
      return { label, logit: Math.round(z * 1000) / 1000, score: Math.round(1000 / (1 + Math.exp(-z))) / 1000 };
    });
    return { data, rows: geo.rows, cols: geo.cols, classScores,
             targets: targets.map(t => ({ label: manifest.classes[t.c], score: Math.round(t.prob * 1000) / 1000 })) };
  }

  // Measured from the photo inside the strongest lesion box, compared with the skin ring
  // around it (box enlarged by 50%). Descriptive only: the detector does not use these.
  function measureLesion(cv, rgb, detections, cam) {
    const detection = detections[0], count = detections.length;
    const W = rgb.cols, H = rgb.rows;
    const [x1, y1, x2, y2] = [detection.box[0] * W, detection.box[1] * H, detection.box[2] * W, detection.box[3] * H].map(Math.round);
    const bw = Math.max(1, x2 - x1), bh = Math.max(1, y2 - y1);
    const ox1 = Math.max(0, Math.round(x1 - bw / 4)), oy1 = Math.max(0, Math.round(y1 - bh / 4));
    const ox2 = Math.min(W, Math.round(x2 + bw / 4)), oy2 = Math.min(H, Math.round(y2 + bh / 4));
    const lab = new cv.Mat(), gray = new cv.Mat(), lap = new cv.Mat();
    cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);
    cv.cvtColor(rgb, gray, cv.COLOR_RGB2GRAY);
    cv.Laplacian(gray, lap, cv.CV_32F);
    const L = lab.data, P = lap.data32F;
    const acc = () => ({ n: 0, l: 0, a: 0, b: 0, t: 0, t2: 0 });
    const inside = acc(), ring = acc();
    let camIn = 0, camAll = 0;
    for (let y = oy1; y < oy2; y++) {
      for (let x = ox1; x < ox2; x++) {
        const i = y * W + x, tgt = (x >= x1 && x < x2 && y >= y1 && y < y2) ? inside : ring;
        tgt.n++; tgt.l += L[3 * i] * 100 / 255; tgt.a += L[3 * i + 1] - 128; tgt.b += L[3 * i + 2] - 128;
        tgt.t += P[i]; tgt.t2 += P[i] * P[i];
      }
    }
    if (cam) {
      // Share of the Grad-CAM heatmap that falls inside any detected lesion box.
      const inBox = new Uint8Array(W * H);
      detections.forEach(d => {
        const [bx1, by1, bx2, by2] = [d.box[0] * W, d.box[1] * H, d.box[2] * W, d.box[3] * H].map(Math.round);
        for (let y = Math.max(0, by1); y < Math.min(H, by2); y++) inBox.fill(1, y * W + Math.max(0, bx1), y * W + Math.min(W, bx2));
      });
      for (let i = 0; i < cam.data.length; i++) {
        camAll += cam.data[i];
        if (inBox[i]) camIn += cam.data[i];
      }
    }
    lab.delete(); gray.delete(); lap.delete();
    const mean = r => r.n ? { L: r.l / r.n, a: r.a / r.n, b: r.b / r.n, rough: Math.sqrt(Math.max(0, r.t2 / r.n - (r.t / r.n) ** 2)) } : null;
    const les = mean(inside), skin = mean(ring);
    const r1 = v => Math.round(v * 10) / 10;
    return {
      lesions: count,
      box_px: [bw, bh],
      area_percent: r1(100 * bw * bh / (W * H)),
      lesion_lab: les && [r1(les.L), r1(les.a), r1(les.b)],
      skin_lab: skin && [r1(skin.L), r1(skin.a), r1(skin.b)],
      delta_e: les && skin ? r1(Math.hypot(les.L - skin.L, les.a - skin.a, les.b - skin.b)) : null,
      redder: les && skin ? r1(les.a - skin.a) : null,
      yellower: les && skin ? r1(les.b - skin.b) : null,
      darker: les && skin ? r1(skin.L - les.L) : null,
      roughness_ratio: les && skin && skin.rough > 0 ? Math.round(100 * les.rough / skin.rough) / 100 : null,
      gradcam_in_box_percent: camAll > 0 ? r1(100 * camIn / camAll) : null,
    };
  }

  // Heatmap alone (transparent where the model does not look), for the Workspace overlay.
  function camHeatmapUrl(cv, cam, maxSide = 640) {
    const f32 = cv.matFromArray(cam.rows, cam.cols, cv.CV_32F, cam.data);
    const u8 = new cv.Mat();
    f32.convertTo(u8, cv.CV_8U, 255);
    const heat = new cv.Mat();
    cv.applyColorMap(u8, heat, cv.COLORMAP_JET);  // BGR
    const n = cam.rows * cam.cols, rgba = new Uint8ClampedArray(4 * n), hp = heat.data;
    for (let i = 0; i < n; i++) {
      rgba[4 * i] = hp[3 * i + 2];
      rgba[4 * i + 1] = hp[3 * i + 1];
      rgba[4 * i + 2] = hp[3 * i];
      rgba[4 * i + 3] = Math.round(255 * 0.65 * Math.min(Math.max(cam.data[i], 0), 1));
    }
    f32.delete(); u8.delete(); heat.delete();
    const full = document.createElement('canvas');
    full.width = cam.cols; full.height = cam.rows;
    full.getContext('2d').putImageData(new ImageData(rgba, cam.cols, cam.rows), 0, 0);
    const scale = Math.min(1, maxSide / Math.max(cam.rows, cam.cols));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(cam.cols * scale); canvas.height = Math.round(cam.rows * scale);
    canvas.getContext('2d').drawImage(full, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/png');
  }

  // Heatmap over the photo: JET colours blended by heatmap strength (as in the paper figures).
  function camToDataUrl(cv, rgb, cam, detections, maxSide = 640) {
    const f32 = cv.matFromArray(cam.rows, cam.cols, cv.CV_32F, cam.data);
    const u8 = new cv.Mat();
    f32.convertTo(u8, cv.CV_8U, 255);
    f32.delete();
    const heat = new cv.Mat();
    cv.applyColorMap(u8, heat, cv.COLORMAP_JET);  // BGR
    const out = rgb.clone();
    const px = out.data, hp = heat.data, n = cam.rows * cam.cols;
    for (let i = 0; i < n; i++) {
      const a = 0.65 * Math.min(Math.max(cam.data[i], 0), 1);
      px[3 * i] = px[3 * i] * (1 - a) + hp[3 * i + 2] * a;
      px[3 * i + 1] = px[3 * i + 1] * (1 - a) + hp[3 * i + 1] * a;
      px[3 * i + 2] = px[3 * i + 2] * (1 - a) + hp[3 * i] * a;
    }
    const url = toDataUrl(cv, out, detections, maxSide);
    u8.delete(); heat.delete(); out.delete();
    return url;
  }

  // Image-level suggestion: class with the highest summed confidence.
  function summarize(detections) {
    const totals = {};
    detections.forEach(d => { totals[d.label] = (totals[d.label] || 0) + d.confidence; });
    const top = Object.keys(totals).sort((a, b) => totals[b] - totals[a])[0] || null;
    const topConfidence = top ? Math.max(...detections.filter(d => d.label === top).map(d => d.confidence)) : null;
    return { top, topConfidence };
  }

  // ---------------------------------------------------------------- images

  async function loadRgb(cv, source) {
    const img = new Image();
    img.src = source;
    await img.decode();  // EXIF orientation is applied by the browser
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const rgba = cv.matFromImageData(ctx.getImageData(0, 0, canvas.width, canvas.height));
    const rgb = new cv.Mat();
    cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);
    rgba.delete();
    const limited = resizeMax(cv, rgb, MAX_SIDE);
    rgb.delete();
    return limited;
  }

  function toDataUrl(cv, rgb, detections, maxSide = 800) {
    const scale = Math.min(1, maxSide / Math.max(rgb.rows, rgb.cols));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(rgb.cols * scale);
    canvas.height = Math.round(rgb.rows * scale);
    const rgba = new cv.Mat();
    cv.cvtColor(rgb, rgba, cv.COLOR_RGB2RGBA);
    const full = document.createElement('canvas');
    full.width = rgb.cols;
    full.height = rgb.rows;
    full.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(rgba.data), rgb.cols, rgb.rows), 0, 0);
    rgba.delete();
    const ctx = canvas.getContext('2d');
    ctx.drawImage(full, 0, 0, canvas.width, canvas.height);
    if (detections) {
      ctx.lineWidth = 2;
      ctx.font = 'bold 11px sans-serif';
      detections.forEach(d => {
        const [x1, y1, x2, y2] = [d.box[0] * canvas.width, d.box[1] * canvas.height, d.box[2] * canvas.width, d.box[3] * canvas.height];
        ctx.strokeStyle = '#E11D48';
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const text = `${d.label} ${(d.confidence * 100).toFixed(0)}%`;
        const tw = ctx.measureText(text).width + 6;
        ctx.fillStyle = '#E11D48';
        ctx.fillRect(x1, Math.max(0, y1 - 14), tw, 14);
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(text, x1 + 3, Math.max(11, y1 - 3));
      });
    }
    return canvas.toDataURL('image/jpeg', 0.85);
  }

  // ---------------------------------------------------------------- public API

  // Same response shape as the former /detect server endpoint.
  async function runDetect(source) {
    const [cv, manifest] = await Promise.all([loadCv(), loadManifest()]);
    const modelD = manifest.models.find(m => m.id === 'D');
    const rgb = await loadRgb(cv, source);
    try {
      const itaResult = computeIta(cv, rgb, manifest.masking);
      const { beta, source: betaSource } = betaForD(itaResult.ita, manifest.calibration);
      const enhanced = lClahe(cv, rgb, beta, manifest.tile_grid_size);
      try {
        const camSpec = modelD.cam ? await loadCamSpec(modelD.cam).catch(() => null) : null;
        const { detections, cam } = await detect(cv, enhanced, modelD.file, manifest, camSpec);
        const { top, topConfidence } = summarize(detections);
        return {
          enhanced_image: toDataUrl(cv, enhanced),
          boxed_image: toDataUrl(cv, rgb, detections, 800),
          gradcam_image: cam ? camToDataUrl(cv, rgb, cam, detections) : null,
          gradcam_heatmap: cam ? camHeatmapUrl(cv, cam) : null,
          gradcam_targets: cam ? cam.targets : [],
          class_scores: cam ? cam.classScores : [],
          lesion_measures: detections.length ? measureLesion(cv, rgb, detections, cam) : null,
          ita_inputs: itaResult.ita === null ? null : { meanL: itaResult.meanL, meanB: itaResult.meanB, k: itaResult.k },
          top_label: top,
          top_confidence: topConfidence,
          detections,
          ita: itaResult.ita,
          bracket: itaResult.bracket,
          mask_status: itaResult.status,
          beta,
          beta_source: betaSource,
          weights: modelD.file,
          calibration_status: manifest.calibration.status,
        };
      } finally {
        enhanced.delete();
      }
    } finally {
      rgb.delete();
    }
  }

  // Same response shape as the former /compare server endpoint.
  async function runCompare(source) {
    const [cv, manifest] = await Promise.all([loadCv(), loadManifest()]);
    const rgb = await loadRgb(cv, source);
    try {
      const itaResult = computeIta(cv, rgb, manifest.masking);
      const { beta: betaD, source: betaSource } = betaForD(itaResult.ita, manifest.calibration);
      const models = [];
      for (const model of manifest.models) {
        const { image, beta } = preprocess(cv, rgb, model.preprocessing, manifest, itaResult);
        try {
          const camSpec = model.cam ? await loadCamSpec(model.cam).catch(() => null) : null;
          const { detections, cam } = await detect(cv, image, model.file, manifest, camSpec);
          const { top, topConfidence } = summarize(detections);
          models.push({
            id: model.id, description: model.description, available: true, beta,
            top_label: top, top_confidence: topConfidence, boxes: detections.length,
            image: toDataUrl(cv, image, detections, 640),
            gradcam_image: cam ? camToDataUrl(cv, rgb, cam, null) : null,
          });
        } finally {
          image.delete();
        }
      }
      return { ita: itaResult.ita, bracket: itaResult.bracket, mask_status: itaResult.status, beta_d: betaD, beta_source: betaSource, models };
    } finally {
      rgb.delete();
    }
  }

  return { runDetect, runCompare, _internal: { loadCv, loadManifest, computeIta, lClahe, rgbClahe, letterbox, detect, loadRgb } };
})();
