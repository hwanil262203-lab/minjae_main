/*
  YOLO 객체 감지 (ONNX Runtime Web) + 스마트폰 진동
  - best.onnx (Ultralytics YOLO export, opset 12) 를 브라우저에서 실행
  - WebGPU 우선, 실패 시 WASM(CPU) 폴백
  - 박스 아랫변 위치 + 카메라 높이 + 폰 기울기로 거리(m) 추정 (사람·차량은 크기도 함께), 방향은 박스 위치
  - 신뢰도 임계치 + 연속 프레임 히스테리시스 후 방향/근접도별 진동 패턴
*/

const CONFIG = {
  // ?model=yolo11n.onnx 처럼 주소로 모델 파일을 바꿔 테스트할 수 있다
  modelUrl: new URLSearchParams(location.search).get('model') || './best.onnx',
  labelsUrl: new URLSearchParams(location.search).get('labels') || './labels.json',
  // best.onnx 가 없을 때 자동으로 쓰는 시험용 모델 (YOLOv10n, COCO 80클래스 사전학습)
  fallbackModelUrl: './yolov10n.onnx',
  fallbackLabelsUrl: './labels_coco.json',
  ortVersion: '1.30.0',
  nmsIou: 0.45,
  maxDraw: 8,
  vibrateRepeatMs: 1500, // 위험이 계속되면 이 간격으로 진동을 다시 울림
  // 감지 루프 자가 복구
  frameTimeoutMs: 8000, // 한 프레임 추론이 이보다 오래 걸리면 멈춘 것으로 보고 세션 재생성
  loadTimeoutMs: 60000, // 모델 로드·워밍업 제한 시간
  maxConsecutiveErrors: 3, // 연속 오류가 이만큼이면 세션 재생성
  maxRecoveries: 3, // 빠른 세션 재생성 한도 (넘으면 모델 전체 재로드)
  maxTotalRecoveries: 30, // 전체 재로드 한도 (넘으면 오류 창)
  modelLoadAttempts: 30, // 모델 로드 자동 재시도 횟수
  modelLoadRetryMs: 3000, // 재시도 간격
};

// 클래스별 한국어 이름, 위험 가중치(0~1), 색. 키는 labels.json 의 영어 이름.
const CLASS_INFO = {
  crosswalk: { ko: '횡단보도', weight: 0.8, color: '#facc15' },
  stairs: { ko: '계단', weight: 1.0, color: '#f97316' },
  pole: { ko: '기둥·가로등', weight: 1.0, color: '#ef4444' },
  person: { ko: '사람', weight: 0.6, color: '#38bdf8' },
  vehicle: { ko: '차량', weight: 0.9, color: '#a855f7' },
  // COCO 사전학습 모델로 파이프라인을 테스트할 때 쓰이는 보조 항목
  car: { ko: '자동차', weight: 0.9, color: '#a855f7' },
  bus: { ko: '버스', weight: 0.9, color: '#a855f7' },
  truck: { ko: '트럭', weight: 0.9, color: '#a855f7' },
  bicycle: { ko: '자전거', weight: 0.7, color: '#a855f7' },
  motorcycle: { ko: '오토바이', weight: 0.8, color: '#a855f7' },
  'fire hydrant': { ko: '소화전', weight: 0.7, color: '#ef4444' },
  bench: { ko: '벤치', weight: 0.6, color: '#ef4444' },
  'traffic light': { ko: '신호등', weight: 0.2, color: '#94a3b8' },
  'stop sign': { ko: '정지표지', weight: 0.2, color: '#94a3b8' },
  dog: { ko: '개', weight: 0.5, color: '#38bdf8' },
  chair: { ko: '의자', weight: 0.5, color: '#ef4444' },
};
const DEFAULT_CLASS = { weight: 0.5, color: '#94a3b8' };
const DIR_KO = { left: '왼쪽', center: '정면', right: '오른쪽' };

const ui = {
  video: document.getElementById('video'),
  overlay: document.getElementById('overlay'),
  btnLoadModel: document.getElementById('btnLoadModel'),
  btnStart: document.getElementById('btnStart'),
  btnStop: document.getElementById('btnStop'),
  threshold: document.getElementById('threshold'),
  steadyFrames: document.getElementById('steadyFrames'),
  proximity: document.getElementById('proximity'),
  status: document.getElementById('status'),
  engine: document.getElementById('engine'),
  log: document.getElementById('log'),
  detections: document.getElementById('detections'),
  floatingIndicator: document.getElementById('floatingIndicator'),
};

let session = null;
let labels = { names: [], imgsz: 640, nms_in_model: null };
let inputName = null;
let outputName = null;
let inputSize = 640;
let engineName = '-';
let usingFallbackModel = false;
let webcamStream = null;
let testImage = null; // ?image=... 로 지정한 정지 이미지 (카메라 없이 테스트)
let isRunning = false;
let animationHandle = null;

const log = (msg) => {
  const t = new Date().toLocaleTimeString();
  ui.log.textContent += `[${t}] ${msg}\n`;
  ui.log.scrollTop = ui.log.scrollHeight;
};

function setStatus(text) {
  ui.status.textContent = text;
}

function classInfo(label) {
  return CLASS_INFO[label] || { ...DEFAULT_CLASS, ko: label };
}

// ============================================
// 카메라
// ============================================
async function openCamera() {
  if (webcamStream) {
    log('카메라가 이미 실행 중입니다.');
    return;
  }
  try {
    log('카메라 권한 요청 중...');
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    ui.video.srcObject = webcamStream;
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('비디오 로드 타임아웃')), 10000);
      const ready = () => {
        clearTimeout(timeout);
        ui.overlay.width = ui.video.videoWidth;
        ui.overlay.height = ui.video.videoHeight;
        resolve();
      };
      ui.video.onloadedmetadata = ready;
      if (ui.video.readyState >= 1) ready();
    });
    await ui.video.play();
    log(`✓ 카메라 시작됨 (${ui.video.videoWidth}x${ui.video.videoHeight})`);
  } catch (e) {
    log(`✗ 카메라 오류: ${e.message}`);
    console.error('카메라 오류:', e);
  }
}

// 테스트 모드: ?image=sample.jpg 로 정지 이미지를 카메라 대신 사용
async function loadTestImage(url) {
  try {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () => reject(new Error('이미지를 불러오지 못했습니다'));
      img.src = url;
    });
    testImage = img;
    ui.video.style.display = 'none';
    img.style.cssText = 'width:100%;height:auto;display:block;max-width:640px';
    ui.video.parentElement.insertBefore(img, ui.overlay);
    ui.overlay.width = img.naturalWidth;
    ui.overlay.height = img.naturalHeight;
    log(`✓ 테스트 이미지 사용 (${img.naturalWidth}x${img.naturalHeight}): ${url}`);
  } catch (e) {
    log(`✗ 테스트 이미지 오류: ${e.message}`);
  }
}

// ============================================
// 모델 로드 (ONNX Runtime Web)
// ============================================
async function loadModel(opts = {}) {
  if (modelLoading) {
    log('모델을 이미 불러오는 중입니다.');
    return false;
  }
  if (session && !opts.force) {
    log('모델이 이미 준비되어 있습니다.');
    return true;
  }
  modelLoading = true;
  setModelButton('loading');
  try {
    setStatus('모델 로딩 중...');
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${CONFIG.ortVersion}/dist/`;
    ort.env.wasm.numThreads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1));
    if (session) {
      const old = session;
      session = null;
      try { await old.release(); } catch (_) { /* 무시 */ }
    }

    // labels.json (없어도 동작: 클래스 이름이 숫자로 표시됨)
    try {
      const r = await fetch(CONFIG.labelsUrl, { cache: 'no-store' });
      if (r.ok) labels = { ...labels, ...(await r.json()) };
    } catch (_) {
      /* 무시 */
    }
    // 모델 파일이 없으면(아직 학습 전) 시험용 사전학습 모델로 대체
    let modelUrl = CONFIG.modelUrl;
    usingFallbackModel = false;
    try {
      const head = await fetch(modelUrl, { method: 'HEAD', cache: 'no-store' });
      if (!head.ok) throw new Error(`HTTP ${head.status}`);
    } catch (e) {
      log(`⚠ ${modelUrl} 없음 (${e.message}) → 시험용 모델 ${CONFIG.fallbackModelUrl} 사용`);
      modelUrl = CONFIG.fallbackModelUrl;
      usingFallbackModel = true;
      try {
        const r = await fetch(CONFIG.fallbackLabelsUrl, { cache: 'no-store' });
        if (r.ok) labels = { ...labels, ...(await r.json()) };
      } catch (_) {
        /* 무시 */
      }
    }
    activeModelUrl = modelUrl;
    log(`모델 파일: ${modelUrl}`);

    // WebGPU 우선, 실패하면 WASM. (opts.wasmOnly 면 처음부터 WASM)
    const providers = opts.providers || (navigator.gpu && settings.useWebGpu && !opts.wasmOnly ? ['webgpu', 'wasm'] : ['wasm']);
    if (navigator.gpu && !settings.useWebGpu) log('이 기기에서는 WebGPU 를 건너뛰고 WASM 으로 실행합니다 (설정에서 바꿀 수 있음)');
    await createSession(modelUrl, providers);

    ui.engine.textContent = engineLabel();
    log(`클래스: ${labels.names.join(', ') || '(labels.json 없음)'}`);
    setStatus(usingFallbackModel ? '시험용 모델 준비 완료' : '모델 준비 완료');
    setModelButton('ready');
    loopState.recoveries = 0;
    return true;
  } catch (e) {
    log(`✗ 모델 로드 실패: ${e.message}`);
    console.error('모델 로드 오류:', e);
    setStatus('모델 로드 실패 — 🔄 모델 로드 버튼을 눌러 다시 시도');
    session = null;
    setModelButton('retry');
    return false;
  } finally {
    modelLoading = false;
  }
}

let modelLoading = false;
let activeModelUrl = null;

function withTimeout(promise, ms, label) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`${label} 시간 초과 (${ms}ms)`)), ms);
    promise.then(
      (v) => { clearTimeout(t); resolve(v); },
      (e) => { clearTimeout(t); reject(e); }
    );
  });
}

function engineLabel() {
  return `${engineName.toUpperCase()} · ${inputSize}px · NMS ${labels.nms_in_model ? '내장' : 'JS'}` +
    (usingFallbackModel ? ' · 시험용 모델' : '');
}

function setModelButton(state) {
  const b = ui.btnLoadModel;
  if (!b) return;
  if (state === 'loading') {
    b.disabled = true;
    b.textContent = '⏳ 모델 로드 중...';
  } else if (state === 'ready') {
    b.disabled = false;
    b.textContent = '🔄 모델 다시 로드';
    b.classList.add('secondary');
  } else {
    b.disabled = false;
    b.textContent = '🔄 모델 로드';
    b.classList.remove('secondary'); // 실패 시 눈에 띄게
  }
}

// 세션 생성 + 워밍업. providers 를 순서대로 시도해 처음 성공한 것을 쓴다
async function createSession(modelUrl, providers) {
  let lastErr = null;
  for (const ep of providers) {
    try {
      const s = await withTimeout(
        ort.InferenceSession.create(modelUrl, { executionProviders: [ep], graphOptimizationLevel: 'all' }),
        CONFIG.loadTimeoutMs, `${ep} 세션 생성`
      );
      const inName = s.inputNames[0];
      const outName = s.outputNames[0];
      // 입력 크기: 모델 메타데이터 → labels.json → 640 순서로 결정
      let size = labels.imgsz || 640;
      const meta = s.inputMetadata && s.inputMetadata[0];
      const dims = meta && (meta.shape || meta.dims);
      if (dims && typeof dims[2] === 'number' && dims[2] > 0) size = dims[2];
      // 워밍업 겸 출력 형식 확인
      const zeros = new ort.Tensor('float32', new Float32Array(3 * size * size), [1, 3, size, size]);
      const t0 = performance.now();
      const out = await withTimeout(s.run({ [inName]: zeros }), CONFIG.loadTimeoutMs, `${ep} 워밍업`);
      const od = out[outName].dims;
      session = s;
      engineName = ep;
      inputName = inName;
      outputName = outName;
      inputSize = size;
      labels.nms_in_model = od.length === 3 && od[2] === 6;
      log(`✓ 모델 로드 완료 (${ep}, 입력 ${size}px, 출력 [${od.join(',')}], 워밍업 ${Math.round(performance.now() - t0)}ms)`);
      return true;
    } catch (e) {
      lastErr = e;
      log(`${ep} 초기화 실패 → 다음 백엔드 시도 (${String((e && e.message) || e).split('\n')[0]})`);
    }
  }
  throw lastErr || new Error('세션 생성 실패');
}

// 모델이 로드될 때까지 반복 시도. 한도를 넘으면 오류 창을 띄운다.
let modelRetryPromise = null;

function showErrorModal(text) {
  const m = document.getElementById('errorModal');
  if (!m) return;
  document.getElementById('errorModalText').textContent = text;
  m.hidden = false;
}

function ensureModel(maxAttempts = CONFIG.modelLoadAttempts, delayMs = CONFIG.modelLoadRetryMs) {
  if (session) return Promise.resolve(true);
  if (modelRetryPromise) return modelRetryPromise; // 이미 재시도 중이면 그 결과를 함께 기다림
  modelRetryPromise = (async () => {
    try {
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        if (session) return true;
        if (attempt > 1) {
          setStatus(`모델 로드 재시도 ${attempt}/${maxAttempts}...`);
          log(`모델 로드 재시도 ${attempt}/${maxAttempts}`);
        }
        const ok = await loadModel();
        if (ok && session) return true;
        await new Promise((r) => setTimeout(r, delayMs));
      }
      log(`✗ 모델 로드 ${maxAttempts}회 실패`);
      setStatus('모델 로드 실패');
      showErrorModal('모델 로드에 실패했습니다. 다시 접속하여 주십시오.');
      return false;
    } finally {
      modelRetryPromise = null;
    }
  })();
  return modelRetryPromise;
}

// 추론이 멈추거나 오류가 반복되면 세션을 새로 만든다 (WebGPU 가 문제면 WASM 으로 내려감)
async function recoverSession(reason) {
  loopState.recoveries += 1;
  if (loopState.recoveries > CONFIG.maxRecoveries) {
    log(`✗ 복구 한도(${CONFIG.maxRecoveries}회) 초과`);
    return false;
  }
  log(`⚠ ${reason} → 모델을 WASM 으로 다시 불러옵니다 (복구 ${loopState.recoveries}/${CONFIG.maxRecoveries})`);
  const old = session;
  session = null;
  try { if (old) await old.release(); } catch (_) { /* 무시 */ }
  try {
    await createSession(activeModelUrl || CONFIG.modelUrl, ['wasm']);
    ui.engine.textContent = engineLabel() + ' · 복구됨';
    return true;
  } catch (e) {
    log(`✗ 복구 실패: ${e.message}`);
    return false;
  }
}

// ============================================
// 전처리: letterbox → NCHW float32 (0~1)
// ============================================
const pre = { canvas: document.createElement('canvas'), ctx: null };

// 현재 프레임 소스 (카메라 또는 테스트 이미지)
function frameSource() {
  if (testImage) return { el: testImage, w: testImage.naturalWidth, h: testImage.naturalHeight };
  return { el: ui.video, w: ui.video.videoWidth, h: ui.video.videoHeight };
}

function preprocess(src, S) {
  if (pre.canvas.width !== S) {
    pre.canvas.width = S;
    pre.canvas.height = S;
    pre.ctx = pre.canvas.getContext('2d', { willReadFrequently: true });
  }
  const vw = src.w;
  const vh = src.h;
  const scale = Math.min(S / vw, S / vh);
  const nw = Math.round(vw * scale);
  const nh = Math.round(vh * scale);
  const dx = Math.floor((S - nw) / 2);
  const dy = Math.floor((S - nh) / 2);
  const ctx = pre.ctx;
  ctx.fillStyle = '#727272';
  ctx.fillRect(0, 0, S, S);
  ctx.drawImage(src.el, 0, 0, vw, vh, dx, dy, nw, nh);
  const { data } = ctx.getImageData(0, 0, S, S);
  const area = S * S;
  const f = new Float32Array(3 * area);
  for (let i = 0, p = 0; i < area; i++, p += 4) {
    f[i] = data[p] / 255;
    f[area + i] = data[p + 1] / 255;
    f[2 * area + i] = data[p + 2] / 255;
  }
  return { tensor: new ort.Tensor('float32', f, [1, 3, S, S]), scale, dx, dy };
}

// ============================================
// 후처리: 출력 텐서 → [{label, score, x1,y1,x2,y2}] (비디오 좌표)
// ============================================
function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0;
}

function nms(cands, thr) {
  cands.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const c of cands) {
    if (keep.every((k) => k.cls !== c.cls || iou(k, c) < thr)) keep.push(c);
  }
  return keep;
}

function postprocess(output, lb, conf, vw, vh) {
  const d = output.dims;
  const data = output.data;
  let boxes = [];

  if (d.length === 3 && d[2] === 6) {
    // Ultralytics nms=True 출력: [1, max_det, 6] = x1,y1,x2,y2,conf,cls
    for (let i = 0; i < d[1]; i++) {
      const o = i * 6;
      const s = data[o + 4];
      if (s < conf) continue;
      boxes.push({ x1: data[o], y1: data[o + 1], x2: data[o + 2], y2: data[o + 3], score: s, cls: data[o + 5] | 0 });
    }
  } else if (d.length === 3) {
    // 원시 출력: [1, 4+nc, N] (Ultralytics 기본) 또는 [1, N, 4+nc]
    const transposed = d[1] > d[2];
    const C = transposed ? d[2] : d[1];
    const N = transposed ? d[1] : d[2];
    const nc = C - 4;
    const at = transposed ? (c, i) => data[i * C + c] : (c, i) => data[c * N + i];
    const cands = [];
    for (let i = 0; i < N; i++) {
      let best = 0;
      let bc = -1;
      for (let c = 0; c < nc; c++) {
        const s = at(4 + c, i);
        if (s > best) {
          best = s;
          bc = c;
        }
      }
      if (best < conf) continue;
      const cx = at(0, i);
      const cy = at(1, i);
      const w = at(2, i);
      const h = at(3, i);
      cands.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score: best, cls: bc });
    }
    boxes = nms(cands, CONFIG.nmsIou);
  } else {
    throw new Error(`지원하지 않는 출력 형식: [${d.join(',')}]`);
  }

  // letterbox 좌표 → 비디오 좌표
  const clampX = (v) => Math.max(0, Math.min(vw, v));
  const clampY = (v) => Math.max(0, Math.min(vh, v));
  return boxes.map((b) => ({
    label: labels.names[b.cls] !== undefined ? labels.names[b.cls] : String(b.cls),
    score: b.score,
    x1: clampX((b.x1 - lb.dx) / lb.scale),
    y1: clampY((b.y1 - lb.dy) / lb.scale),
    x2: clampX((b.x2 - lb.dx) / lb.scale),
    y2: clampY((b.y2 - lb.dy) / lb.scale),
  }));
}

// ============================================
// 거리 추정(미터) — 김민재 진로세특 '카메라 한 개로 거리 측정' 원리(닮은 삼각형 비례식) 적용
//  · 바닥 기준: 카메라 높이 h 와 폰 기울기를 알면, 박스 아랫변 픽셀 y 를 내려다본 각 α = (아래로 기운 각) + atan((y − cy) ÷ f),
//    거리 d = h ÷ tan α.  보고서의 '물체의 실제 크기' 자리에 카메라 높이를 넣은 것이라 종류와 상관없이 쓸 수 있다.
//  · 크기 기준: 실제 높이를 아는 종류(사람·차량)는 보고서 공식 그대로 d = f × 실제 높이 ÷ 박스 픽셀 높이.
//  · 초점거리 f(픽셀)는 웹 브라우저가 알려 주지 않는다 → 화각 66° 가정값(긴 변 × 0.77)을 쓰고 설정의 '1m 보정'으로 기기별로 맞춘다.
//  · 기울기는 DeviceOrientation(beta) 로 읽는다. 없으면 가슴 높이에서 조금 숙인 −12° 로 가정.
// ============================================
const REAL_HEIGHT = { person: 1.6, vehicle: 1.5, car: 1.5, bus: 3.0, truck: 2.5, bicycle: 1.0, motorcycle: 1.1 };
const Range = {
  pitchDeg: null, // 카메라 광축이 수평에서 기운 각 (아래로 향하면 음수)
  src: 'none', // 'sensor' | 'none'
  DEFAULT_PITCH: -12,
  calibrating: false,
  focal(vw, vh) { return settings.focalScale * Math.max(vw, vh); },
  pitch() { return this.pitchDeg == null ? this.DEFAULT_PITCH : this.pitchDeg; },
  downRad() { return (-this.pitch() * Math.PI) / 180; },
  // 박스 아랫변 y(픽셀) → 바닥 위 그 지점까지의 거리(m). 지평선 위면 null
  groundDist(y2, vw, vh) {
    const alpha = this.downRad() + Math.atan((y2 - vh / 2) / this.focal(vw, vh));
    if (alpha < 0.015) return null; // 거의 수평 → 너무 멀거나 지평선 위
    return settings.camHeight / Math.tan(alpha);
  },
  sizeDist(label, hPx, vw, vh) {
    const H = REAL_HEIGHT[label];
    if (!H || hPx < 4) return null;
    return (this.focal(vw, vh) * H) / hPx;
  },
  estimate(det, vw, vh) {
    const g = this.groundDist(det.y2, vw, vh);
    const s = this.sizeDist(det.label, det.y2 - det.y1, vw, vh);
    let d = null, src = '';
    if (g != null && s != null) { d = Math.sqrt(g * s); src = '바닥+크기'; } // 기하 평균
    else if (g != null) { d = g; src = '바닥'; }
    else if (s != null) { d = s; src = '크기'; }
    if (d != null) d = Math.max(0.3, Math.min(50, d));
    return { d, src };
  },
  // 폰 기울기: 세로로 들었을 때 beta=90° 가 카메라 수평. 숙이면 작아진다
  onOrient(e) {
    if (typeof e.beta !== 'number') return;
    if (Math.abs(e.gamma || 0) > 60) return; // 옆으로 눕힌 상태는 무시
    const p = e.beta - 90;
    Range.pitchDeg = Range.pitchDeg == null ? p : Range.pitchDeg * 0.7 + p * 0.3; // 떨림 완화
    Range.src = 'sensor';
  },
  async start() {
    try {
      if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
        const r = await DeviceOrientationEvent.requestPermission(); // iOS: 사용자 탭 안에서만
        if (r !== 'granted') return false;
      }
    } catch (_) { return false; }
    window.addEventListener('deviceorientation', this.onOrient, true);
    return true;
  },
  // 1m 보정: 사용자가 D m 앞 바닥 지점을 화면에서 누르면 f 를 역산
  calibrateFromTap(yTap, vw, vh, D) {
    const alpha = Math.atan(settings.camHeight / D); // 그 지점을 내려다본 각
    const rel = alpha - this.downRad(); // 화면 가운데(광축)에서 더 내려간 각
    const dy = yTap - vh / 2;
    if (rel <= 0.01 || dy <= 2) return null;
    const f = dy / Math.tan(rel);
    const scale = f / Math.max(vw, vh);
    if (scale < 0.3 || scale > 3) return null;
    return scale;
  },
};

// ============================================
// 위험 판단: 방향 + 거리 + 클래스 가중치
// ============================================
function assess(dets, vw, vh, minProximity) {
  for (const det of dets) {
    const info = classInfo(det.label);
    det.ko = info.ko;
    det.color = info.color;
    // 거리(m): 박스 아랫변 위치(+사람·차량은 크기)로 추정 → 근접도 0~1 로 바꿔 위험도에 반영
    //   1m 이하 → 1.0, 6m 이상 → 0. 원래 큰 물체가 멀리 있으면 아랫변이 위쪽이라 거리가 멀게 나온다.
    const est = Range.estimate(det, vw, vh);
    det.dist = est.d;
    det.distSrc = est.src;
    det.prox = est.d == null ? 0 : Math.max(0, Math.min(1, (6 - est.d) / 5));
    const cx = (det.x1 + det.x2) / 2 / vw;
    det.dir = cx < 0.35 ? 'left' : cx > 0.65 ? 'right' : 'center';
    det.near = est.d != null && est.d <= 2.5;
    det.risk = det.score * info.weight * Math.min(1, det.prox / 0.5);
  }
  dets.sort((a, b) => b.risk - a.risk);
  const best = dets.find((d) => d.prox >= minProximity && classInfo(d.label).weight >= 0.5);
  return best || null;
}

const hysteresisState = { lastLabel: null, steadyCount: 0 };

function updateHysteresis(currentLabel, requiredFrames) {
  if (currentLabel && currentLabel === hysteresisState.lastLabel) {
    hysteresisState.steadyCount += 1;
  } else if (currentLabel) {
    hysteresisState.lastLabel = currentLabel;
    hysteresisState.steadyCount = 1;
  } else {
    hysteresisState.lastLabel = null;
    hysteresisState.steadyCount = 0;
  }
  return hysteresisState.steadyCount >= requiredFrames ? hysteresisState.lastLabel : null;
}

// ============================================
// 사용자 설정 (localStorage 에 저장)
// ============================================
const SETTINGS_KEY = 'hazard-settings-v1';
const settings = {
  vibStrength: 'mid', // weak | mid | strong
  flash: true, // 위험 알림 중 화면 깜빡임
  flashLevel: 'light', // light | mid | strong
  onboarded: false, // 첫 접속 안내를 봤는지
  useWebGpu: true, // WebGPU 시도 여부. 기기에서 실패하면 자동으로 false 로 저장
  camHeight: 1.3, // 카메라(폰)의 지면 높이 m. 가슴 높이 기준
  focalScale: 0.77, // 픽셀 초점거리 ÷ 긴 변 픽셀 수. 0.77 ≈ 화각 66° (일반 폰 기본 카메라). '1m 보정'으로 기기별 조정
  focalCalibrated: false, // 1m 보정을 했는지
};

function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) Object.assign(settings, JSON.parse(raw));
  } catch (_) {
    /* 사생활 모드 등에서는 무시 */
  }
}

function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch (_) {
    /* 무시 */
  }
}

const VIB_STRENGTH = {
  // 웹 진동 API 는 강도 조절이 없어 펄스 길이·반복 간격으로 흉내 낸다
  weak: { scale: 0.5, repeatMs: 2200 },
  mid: { scale: 1.0, repeatMs: 1500 },
  strong: { scale: 1.6, repeatMs: 1000 },
};
const FLASH_OPACITY = { light: 0.22, mid: 0.38, strong: 0.55 };

// ============================================
// 진동: 방향과 무관. 가장 위험한 대상 하나를 골라(assess 가 위험도 순으로 정렬) 그 대상의
//   가중 위험도(risk = 신뢰도 × 장벽 가중치 × 근접도)만으로 세기를 정하고, 사라질 때까지 계속 울린다.
//   위험도가 높을수록 진동이 길고 쉬는 시간이 짧다 (웹 진동 API 는 강도 조절이 없어 이렇게 흉내 낸다).
// ============================================
const vib = { key: null, t: 0, cycleMs: 0 };
const canVibrate = 'vibrate' in navigator;

// risk: 0 ~ 1 (예: 계단 가중치 1.0 × 신뢰도 0.8 × 근접 1.0 = 0.8). 한 사이클 [울림, 쉼] 을 이어 붙여 약 1.5초 패턴.
const RISK_FULL = 0.7; // 이 위험도 이상이면 최대 세기
function vibrationPattern(risk, strength = settings.vibStrength) {
  const s = VIB_STRENGTH[strength] || VIB_STRENGTH.mid;
  const level = Math.max(0, Math.min(1, risk / RISK_FULL));
  const on = Math.round((180 + 320 * level) * s.scale); // 180~500ms
  const off = Math.round(320 - 220 * level); // 320~100ms
  const pattern = [];
  let total = 0;
  while (total < 1500) {
    pattern.push(on, off);
    total += on + off;
  }
  return pattern;
}

// 사용자가 원(인디케이터)을 눌러 "이 대상은 그만 울려" 로 만든 상태
// label: 무음 처리한 위험물 종류, lastSeen: 마지막으로 그 종류가 확정 감지된 시각
const mute = { label: null, lastSeen: 0 };
const MUTE_CLEAR_MS = 1500; // 이 시간 동안 안 보이면 화면 밖으로 나간 것으로 보고 무음 해제

function muteCurrent(label) {
  mute.label = label;
  mute.lastSeen = performance.now();
  stopVibration();
  log(`🔇 ${classInfo(label).ko} 진동 끔 — 화면에서 사라졌다가 다시 나타나면 다시 울립니다`);
}

function clearMute(why) {
  if (!mute.label) return;
  log(`🔔 ${classInfo(mute.label).ko} 진동 다시 켬 (${why})`);
  mute.label = null;
}

function vibrateFor(det) {
  if (mute.label === det.label) {
    mute.lastSeen = performance.now();
    return; // 같은 대상이 계속 보이는 동안은 조용히
  }
  if (mute.label) clearMute('다른 위험물 감지'); // 다른 종류가 나타나면 무음은 끝
  const key = `${det.label}|${Math.round(det.risk * 10)}`;
  const now = performance.now();
  if (key !== vib.key || now - vib.t > vib.cycleMs) {
    const pattern = vibrationPattern(det.risk);
    if (canVibrate) navigator.vibrate(pattern);
    vib.key = key;
    vib.t = now;
    vib.cycleMs = pattern.reduce((a, b) => a + b, 0) - 50; // 끊김 없이 이어서 울리도록 조금 일찍 다시 보냄
  }
  setFlash(true);
}

function stopVibration() {
  if (vib.key && canVibrate) navigator.vibrate(0);
  vib.key = null;
  vib.t = 0;
  setFlash(false);
  if (mute.label && performance.now() - mute.lastSeen > MUTE_CLEAR_MS) clearMute('화면에서 사라짐');
}

// 화면 깜빡임 (진동 모드/무음 대비)
const flashOverlay = document.getElementById('flashOverlay');

function setFlash(on) {
  if (!flashOverlay) return;
  if (on && settings.flash) {
    flashOverlay.style.setProperty('--flash-opacity', FLASH_OPACITY[settings.flashLevel] || FLASH_OPACITY.light);
    flashOverlay.classList.add('active');
  } else {
    flashOverlay.classList.remove('active');
  }
}

function testVibration() {
  if (!canVibrate) return false;
  // 위험도 0.6 (가까운 계단을 60% 신뢰도로 본 정도) 세기로 1.5초 체감
  return navigator.vibrate(vibrationPattern(0.6));
}

function testFlash() {
  const prev = settings.flash;
  settings.flash = true;
  setFlash(true);
  setTimeout(() => {
    setFlash(false);
    settings.flash = prev;
  }, 2000);
}

// ============================================
// 화면 표시
// ============================================
function drawDetections(ctx, dets, best) {
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.font = '14px ui-monospace, monospace';
  ctx.textBaseline = 'top';
  for (const det of dets.slice(0, CONFIG.maxDraw)) {
    const isBest = det === best;
    ctx.lineWidth = isBest ? 4 : 2;
    ctx.strokeStyle = det.color;
    ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
    const text = `${det.ko} ${(det.score * 100).toFixed(0)}%`;
    const tw = ctx.measureText(text).width + 8;
    ctx.fillStyle = det.color;
    ctx.fillRect(det.x1, Math.max(0, det.y1 - 18), tw, 18);
    ctx.fillStyle = '#0b1020';
    ctx.fillText(text, det.x1 + 4, Math.max(0, det.y1 - 18) + 2);
  }
  if (best) {
    ctx.font = 'bold 18px system-ui, sans-serif';
    const text = `⚠ ${best.ko} · ${DIR_KO[best.dir]} · ${best.dist != null ? '약 ' + best.dist.toFixed(1) + 'm' : '접근 중'}`;
    const tw = ctx.measureText(text).width + 16;
    ctx.fillStyle = 'rgba(239, 68, 68, 0.85)';
    ctx.fillRect(8, 8, tw, 30);
    ctx.fillStyle = '#fff';
    ctx.fillText(text, 16, 14);
  }
}

function updateDetectionList(dets, best) {
  if (!dets.length) {
    ui.detections.innerHTML = '<li class="muted">감지된 물체 없음</li>';
    return;
  }
  ui.detections.innerHTML = dets
    .slice(0, CONFIG.maxDraw)
    .map((d) => {
      const cls = d === best ? 'danger' : '';
      return `<li class="${cls}"><span class="dot" style="background:${d.color}"></span>` +
        `<b>${d.ko}</b> ${(d.score * 100).toFixed(0)}% · ${DIR_KO[d.dir]} · ${d.dist != null ? '약 ' + d.dist.toFixed(1) + 'm' : '거리 모름'}</li>`;
    })
    .join('');
}

// 상태 인디케이터: ok(초록 ✓) / warn(노랑 !) / danger(빨강 !)
const INDICATOR = {
  ok: { cls: '', text: '✓', title: '정상 작동 중 (드래그하여 이동 가능)' },
  warn: { cls: 'warn', text: '!', title: '비정상: 감지가 동작하지 않습니다 (드래그하여 이동 가능)' },
  danger: { cls: 'danger', text: '!', title: '위험 감지됨! 누르면 이 대상이 사라질 때까지 진동을 끕니다 (드래그하여 이동 가능)' },
};
let indicatorState = null;

function setIndicator(state) {
  if (state === indicatorState) return;
  indicatorState = state;
  const s = INDICATOR[state] || INDICATOR.warn;
  ui.floatingIndicator.classList.remove('warn', 'danger');
  if (s.cls) ui.floatingIndicator.classList.add(s.cls);
  ui.floatingIndicator.textContent = s.text;
  ui.floatingIndicator.title = s.title;
}

// 정상 작동 감시: 감지 루프가 멈추거나 프레임이 오래 끊기면 노랑
const health = { lastFrameTs: 0, danger: false };

function updateFloatingIndicator(isDanger) {
  health.danger = isDanger;
  refreshIndicator();
}

function refreshIndicator() {
  if (health.danger) return setIndicator('danger');
  const alive = isRunning && session && performance.now() - health.lastFrameTs < 4000;
  setIndicator(alive ? 'ok' : 'warn');
}

setInterval(refreshIndicator, 1000);

// ============================================
// 추론 루프 (이전 추론이 끝난 뒤 다음 프레임 처리 → 겹치지 않음)
// ============================================
const perf = { ms: 0, frames: 0, lastReport: 0 };

// 감지 루프 상태 (멈춤·오류 복구용)
const loopState = { token: 0, inFlight: false, frameStart: 0, lastTick: 0, consecutiveErrors: 0, recoveries: 0, totalRecoveries: 0, okStreak: 0, lastVideoTime: -1, lastVideoChange: 0 };

async function inferenceLoop() {
  if (!session) {
    log('✗ 모델이 준비되지 않았습니다. 🔄 모델 로드 버튼을 눌러 주세요.');
    setStatus('모델 없음 — 🔄 모델 로드 필요');
    setModelButton('retry');
    return;
  }
  if (!(webcamStream || testImage)) {
    log('✗ 카메라가 준비되지 않았습니다. 카메라 권한을 확인한 뒤 페이지를 새로고침해 주세요.');
    return;
  }
  if (isRunning) return;
  isRunning = true;
  loopState.token += 1;
  const myToken = loopState.token;
  loopState.consecutiveErrors = 0;
  loopState.lastTick = performance.now();
  setStatus('감지 중...');
  log('감지 시작됨');
  const ctx = ui.overlay.getContext('2d');

  const step = async () => {
    if (!isRunning || myToken !== loopState.token) return;
    const t0 = performance.now();
    loopState.inFlight = true;
    loopState.frameStart = t0;
    loopState.lastTick = t0;
    try {
      const src = frameSource();
      const vw = src.w;
      const vh = src.h;
      if (vw && vh && session) {
        if (ui.overlay.width !== vw) {
          ui.overlay.width = vw;
          ui.overlay.height = vh;
        }
        const conf = parseFloat(ui.threshold.value) || 0.25;
        const requiredFrames = parseInt(ui.steadyFrames.value, 10) || 3;
        const maxDist = parseFloat(ui.proximity.value) || 5; // 설정: 이 거리(m) 안일 때만 알림
        const minProximity = Math.max(0.02, (6 - maxDist) / 5);

        const lb = preprocess(src, inputSize);
        const result = await withTimeout(session.run({ [inputName]: lb.tensor }), CONFIG.frameTimeoutMs, '추론');
        if (!isRunning || myToken !== loopState.token) return;
        const dets = postprocess(result[outputName], lb, conf, vw, vh);
        const best = assess(dets, vw, vh, minProximity);
        const confirmed = updateHysteresis(best ? best.label : null, requiredFrames);

        drawDetections(ctx, dets, best);
        updateDetectionList(dets, best);
        updateFloatingIndicator(!!confirmed);

        if (confirmed && best) vibrateFor(best);
        else stopVibration();
        health.lastFrameTs = performance.now();
        loopState.consecutiveErrors = 0;
        loopState.okStreak += 1;
        if (loopState.okStreak === 100) loopState.recoveries = 0; // 한동안 잘 돌면 복구 횟수 초기화
      }
    } catch (e) {
      loopState.consecutiveErrors += 1;
      loopState.okStreak = 0;
      const msg = String((e && e.message) || e).split('\n')[0];
      console.error('추론 오류:', e);
      log(`감지 오류 ${loopState.consecutiveErrors}회 (${engineName}): ${msg}`);
      const timedOut = /시간 초과/.test(msg);
      const gpuErr = engineName === 'webgpu' && /webgpu|kernel|gpu|unsupported/i.test(msg);
      if (gpuErr && settings.useWebGpu) {
        settings.useWebGpu = false;
        saveSettings();
        log('⚠ 이 기기의 WebGPU 가 모델을 실행하지 못합니다. 앞으로는 처음부터 WASM 으로 실행합니다.');
      }
      if (timedOut || gpuErr || loopState.consecutiveErrors >= CONFIG.maxConsecutiveErrors) {
        const ok = await recoverSession(timedOut ? '추론이 응답하지 않음' : gpuErr ? 'WebGPU 실행 실패' : '오류 반복');
        if (!ok) {
          // 빠른 복구가 막히면 정지하지 않고 모델 전체를 다시 불러온 뒤 자동으로 재시작
          loopState.inFlight = false;
          stopLoop();
          loopState.totalRecoveries += 1;
          if (loopState.totalRecoveries > CONFIG.maxTotalRecoveries) {
            setStatus('감지 중단');
            showErrorModal('감지를 계속할 수 없습니다. 다시 접속하여 주십시오.');
            return;
          }
          setStatus('모델을 다시 불러오는 중...');
          log(`↻ 모델 전체 재로드 후 자동 재시작 (${loopState.totalRecoveries}/${CONFIG.maxTotalRecoveries})`);
          loopState.recoveries = 0;
          const stale = session;
          session = null; // 기존 세션을 버리고 완전히 새로 불러온다
          if (stale) stale.release().catch(() => {});
          ensureModel().then((loaded) => {
            if (loaded && (webcamStream || testImage)) inferenceLoop();
          });
          return;
        }
        loopState.consecutiveErrors = 0;
      }
    } finally {
      loopState.inFlight = false;
    }
    if (!isRunning || myToken !== loopState.token) return;
    // 성능 표시
    perf.ms = perf.ms * 0.8 + (performance.now() - t0) * 0.2;
    perf.frames += 1;
    if (performance.now() - perf.lastReport > 1000) {
      perf.lastReport = performance.now();
      setStatus(`감지 중 · ${Math.round(perf.ms)}ms/프레임 (${(1000 / Math.max(1, perf.ms)).toFixed(1)} FPS)`);
    }
    scheduleNext(step);
  };
  step();
}

// 다음 프레임 예약: requestAnimationFrame 이 오지 않는 환경(일부 모바일 브라우저, 백그라운드)에서는 타이머로 진행
function scheduleNext(step) {
  let fired = false;
  const go = () => {
    if (fired) return;
    fired = true;
    step();
  };
  animationHandle = requestAnimationFrame(go);
  setTimeout(go, document.hidden ? 1000 : 300);
}

function stopLoop() {
  isRunning = false;
  loopState.token += 1;
  stopVibration();
  updateFloatingIndicator(false);
  setStatus('정지됨');
  if (animationHandle) cancelAnimationFrame(animationHandle);
}

function reopenCamera(why) {
  log(`⚠ ${why} → 카메라를 다시 엽니다`);
  try { if (webcamStream) webcamStream.getTracks().forEach((t) => t.stop()); } catch (_) { /* 무시 */ }
  webcamStream = null;
  ui.video.srcObject = null;
  openCamera();
}

// 감시: 루프가 멈추거나 카메라 영상이 얼면 스스로 복구
setInterval(() => {
  if (!isRunning) return;
  const now = performance.now();
  // 1) 추론 Promise 가 끝내 돌아오지 않음 (withTimeout 도 못 잡는 극단 상황)
  if (loopState.inFlight && now - loopState.frameStart > CONFIG.frameTimeoutMs + 15000) {
    log('⚠ 감지 루프가 응답하지 않아 다시 시작합니다');
    isRunning = false;
    loopState.token += 1;
    loopState.inFlight = false;
    inferenceLoop();
    return;
  }
  // 2) 다음 프레임이 예약되지 않은 채 멈춤
  if (!loopState.inFlight && now - loopState.lastTick > 6000 && !document.hidden) {
    log('⚠ 감지가 멈춘 것을 감지해 다시 시작합니다');
    isRunning = false;
    loopState.token += 1;
    inferenceLoop();
    return;
  }
  // 3) 카메라 스트림이 끊기거나 영상이 얼어 있음
  if (webcamStream && !testImage) {
    const v = ui.video;
    const track = webcamStream.getVideoTracks()[0];
    if (track && track.readyState === 'ended') {
      reopenCamera('카메라 스트림이 끊김');
      return;
    }
    if (v.paused && v.readyState >= 2) v.play().catch(() => {});
    if (v.currentTime !== loopState.lastVideoTime) {
      loopState.lastVideoTime = v.currentTime;
      loopState.lastVideoChange = now;
    } else if (loopState.lastVideoChange && now - loopState.lastVideoChange > 5000 && !document.hidden) {
      loopState.lastVideoChange = now;
      reopenCamera('카메라 영상이 5초 이상 멈춤');
    }
  }
}, 2000);

ui.btnLoadModel.addEventListener('click', () => (session ? loadModel({ force: true }) : ensureModel()));
document.getElementById('btnErrorReload').addEventListener('click', () => location.reload());
ui.btnStart.addEventListener('click', inferenceLoop);
ui.btnStop.addEventListener('click', stopLoop);

// ============================================
// 드래그 가능한 Floating Indicator
// ============================================
let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;
let indicatorStartX = 0;
let indicatorStartY = 0;

let dragStartT = 0;
let dragMoved = 0;

function beginDrag(clientX, clientY) {
  isDragging = true;
  dragStartX = clientX;
  dragStartY = clientY;
  dragStartT = performance.now();
  dragMoved = 0;
  const rect = ui.floatingIndicator.getBoundingClientRect();
  indicatorStartX = rect.left;
  indicatorStartY = rect.top;
  ui.floatingIndicator.classList.add('dragging');
}

function moveDrag(clientX, clientY) {
  if (!isDragging) return;
  dragMoved = Math.max(dragMoved, Math.hypot(clientX - dragStartX, clientY - dragStartY));
  ui.floatingIndicator.style.left = indicatorStartX + (clientX - dragStartX) + 'px';
  ui.floatingIndicator.style.top = indicatorStartY + (clientY - dragStartY) + 'px';
  ui.floatingIndicator.style.right = 'auto';
}

function endDrag() {
  if (!isDragging) return;
  isDragging = false;
  ui.floatingIndicator.classList.remove('dragging');
  // 거의 움직이지 않고 짧게 뗐으면 '탭' → 위험 감지 중이면 이 대상 무음
  if (dragMoved < 8 && performance.now() - dragStartT < 600) onIndicatorTap();
}

function onIndicatorTap() {
  const label = hysteresisState.lastLabel;
  if (!health.danger || !label) return; // 위험 상태가 아닐 때는 아무 일도 없음
  if (mute.label === label) return; // 이미 무음이면 그대로 (수동 해제 없음, 대상이 사라져야 풀림)
  muteCurrent(label);
}

ui.floatingIndicator.addEventListener('mousedown', (e) => {
  beginDrag(e.clientX, e.clientY);
  e.preventDefault();
});
document.addEventListener('mousemove', (e) => moveDrag(e.clientX, e.clientY));
document.addEventListener('mouseup', endDrag);
ui.floatingIndicator.addEventListener('touchstart', (e) => {
  beginDrag(e.touches[0].clientX, e.touches[0].clientY);
  e.preventDefault();
});
document.addEventListener('touchmove', (e) => {
  if (!isDragging) return;
  moveDrag(e.touches[0].clientX, e.touches[0].clientY);
  e.preventDefault();
}, { passive: false });
document.addEventListener('touchend', endDrag);

// ============================================
// 시작 안내 창 + 설정 창
// ============================================
const panels = {
  start: document.getElementById('startOverlay'),
  settings: document.getElementById('settingsPanel'),
};

function syncSettingsUI() {
  document.querySelectorAll('input[name="vibStrength"]').forEach((r) => (r.checked = r.value === settings.vibStrength));
  document.querySelectorAll('input[name="flashLevel"]').forEach((r) => (r.checked = r.value === settings.flashLevel));
  document.getElementById('flashToggle').checked = settings.flash;
  document.getElementById('startFlashToggle').checked = settings.flash;
  document.getElementById('webgpuToggle').checked = settings.useWebGpu;
  document.getElementById('camHeight').value = Math.round(settings.camHeight * 100);
  renderRangeInfo();
}

function bindSettingsUI() {
  document.querySelectorAll('input[name="vibStrength"]').forEach((r) =>
    r.addEventListener('change', () => {
      if (r.checked) {
        settings.vibStrength = r.value;
        saveSettings();
      }
    })
  );
  document.querySelectorAll('input[name="flashLevel"]').forEach((r) =>
    r.addEventListener('change', () => {
      if (r.checked) {
        settings.flashLevel = r.value;
        saveSettings();
      }
    })
  );
  document.getElementById('flashToggle').addEventListener('change', (e) => {
    settings.flash = e.target.checked;
    saveSettings();
    if (!settings.flash) setFlash(false);
  });
  document.getElementById('webgpuToggle').addEventListener('change', (e) => {
    settings.useWebGpu = e.target.checked;
    saveSettings();
    log(`GPU 가속 ${settings.useWebGpu ? '사용' : '사용 안 함'} — 🔄 모델 다시 로드 를 누르면 적용됩니다`);
  });
  document.getElementById('btnVibTest').addEventListener('click', () => {
    if (!testVibration()) log('진동을 지원하지 않거나 차단된 기기입니다.');
  });
  document.getElementById('btnFlashTest').addEventListener('click', testFlash);
  document.getElementById('camHeight').addEventListener('change', (e) => {
    const cm = parseFloat(e.target.value);
    if (cm >= 50 && cm <= 250) { settings.camHeight = cm / 100; saveSettings(); renderRangeInfo(); }
  });
  document.getElementById('btnCalib').addEventListener('click', () => {
    Range.calibrating = !Range.calibrating;
    renderRangeInfo();
    if (Range.calibrating) {
      panels.settings.hidden = true;
      log('📏 보정: 줄자로 잰 거리(기본 1m) 앞 바닥에 물건을 두고, 폰을 평소보다 조금 더 숙여 그 지점이 화면 아래쪽 1/3쯤에 오게 한 뒤 화면에서 그 지점을 한 번 누르세요.');
    }
  });
  document.getElementById('btnCalibReset').addEventListener('click', () => {
    settings.focalScale = 0.77; settings.focalCalibrated = false; saveSettings(); renderRangeInfo();
    log('거리 보정을 기본값(화각 66°)으로 되돌렸습니다');
  });
  ui.overlay.addEventListener('click', (e) => {
    if (!Range.calibrating) return;
    const rect = ui.overlay.getBoundingClientRect();
    const y = ((e.clientY - rect.top) / rect.height) * ui.overlay.height;
    const D = (parseFloat(document.getElementById('calibDist').value) || 100) / 100;
    const scale = Range.calibrateFromTap(y, ui.overlay.width, ui.overlay.height, D);
    Range.calibrating = false;
    if (scale == null) { log('✗ 보정 실패: 누른 지점이 화면 가운데보다 아래(바닥)여야 합니다. 폰을 조금 더 숙이고 다시 해 보세요.'); renderRangeInfo(); return; }
    settings.focalScale = scale; settings.focalCalibrated = true; saveSettings(); renderRangeInfo();
    log(`✓ 거리 보정 완료: 초점거리 계수 ${scale.toFixed(2)} (기울기 ${Range.pitch().toFixed(0)}°, 카메라 높이 ${settings.camHeight}m)`);
  });
  setInterval(renderRangeInfo, 1000);
  document.getElementById('btnSettings').addEventListener('click', () => {
    syncSettingsUI();
    panels.settings.hidden = false;
  });
  document.getElementById('btnSettingsClose').addEventListener('click', () => {
    panels.settings.hidden = true;
  });

  // 시작 안내 창
  if (!canVibrate) document.getElementById('noVibrateNote').hidden = false;
  document.getElementById('btnStartApp').addEventListener('click', startApp);
}

function renderRangeInfo() {
  const el = document.getElementById('rangeInfo');
  if (!el) return;
  const p = Range.pitch();
  el.textContent = `기울기 ${p.toFixed(0)}° (${Range.src === 'sensor' ? '센서' : '기본값'}) · 초점거리 계수 ${settings.focalScale.toFixed(2)} (${settings.focalCalibrated ? '보정됨' : '기본 화각 66°'})` +
    (Range.calibrating ? ' · 📏 보정 중: 화면의 바닥 지점을 누르세요' : '');
}

function showStartOverlay() {
  panels.start.classList.toggle('first', !settings.onboarded);
  panels.start.classList.toggle('compact', settings.onboarded);
  syncSettingsUI();
  panels.start.hidden = false;
}

// 사용자의 탭(시작하기) 이후에만 카메라·감지를 시작한다.
// Chrome 은 사용자가 페이지를 한 번도 누르지 않으면 navigator.vibrate 를 막는다.
async function startApp() {
  settings.flash = document.getElementById('startFlashToggle').checked;
  settings.onboarded = true;
  saveSettings();
  panels.start.hidden = true;
  Range.start().then((ok) => log(ok ? '✓ 기울기 센서 사용 (거리 계산에 반영)' : '기울기 센서 없음 → 기본 기울기(−12°)로 거리 계산'));

  const imageUrl = new URLSearchParams(location.search).get('image');
  if (imageUrl) {
    if (!testImage) await loadTestImage(imageUrl);
  } else if (!webcamStream) {
    await openCamera();
  }
  // 모델이 아직 없으면(로드 실패·지연) 한 번 더 시도
  if (!session) await ensureModel();
  if (session && (webcamStream || testImage)) {
    log('🚀 감지를 시작합니다...');
    inferenceLoop();
  } else if (!session) {
    log('✗ 모델이 없어 시작하지 못했습니다. 🔄 모델 로드 버튼을 눌러 주세요.');
    setModelButton('retry');
  } else {
    log('✗ 카메라가 준비되지 않았습니다. 카메라 권한을 확인한 뒤 페이지를 새로고침해 주세요.');
  }
}

// ============================================
// 초기화: 설정 로드 → 모델 로드 → 시작 안내 창
// ============================================
window.addEventListener('load', async () => {
  loadSettings();
  bindSettingsUI();
  log('앱 시작됨');
  if (typeof ort === 'undefined') {
    log('✗ ONNX Runtime Web 로드 실패 (인터넷 연결 확인)');
    setStatus('ORT 로드 실패');
    return;
  }
  log(`✓ ONNX Runtime Web ${ort.env.versions ? ort.env.versions.web || '' : ''} 로드 완료`);
  showStartOverlay();
  await ensureModel();
});
