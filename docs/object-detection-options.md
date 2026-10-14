# 객체 감지(Object Detection) 전환 방안 탐색

작성일: 2026-10-14

## 1. 현재 구조 요약

| 항목 | 현재 상태 |
|---|---|
| 모델 종류 | Teachable Machine **이미지 분류** (MobileNet 기반, 입력 224×224) |
| 클래스 | 횡단보도 / 계단 / 기둥, 가로등 / 평지(안전) — `metadata.json` |
| 런타임 | TensorFlow.js 4.2.0 + `@teachablemachine/image` 0.8 (CDN) |
| 추론 흐름 | `model.predict(video)` → `[{className, probability}]` → 임계치(0.90) + 연속 프레임(4) 히스테리시스 → `navigator.vibrate([400,200,400])` |
| 배포 | 정적 HTML/JS, Netlify (HTTPS 필수) |

### 분류 방식의 한계 (전환 동기)

- 화면 **전체**를 한 장으로 판단하므로 위험물이 **어디에**(좌/중/우), **얼마나 가까이** 있는지 알 수 없다.
- 한 화면에 여러 위험(계단 + 기둥)이 있어도 하나의 라벨만 나온다.
- "평지"라는 음성(negative) 클래스를 억지로 넣어야 하고, 배경이 바뀌면 오탐이 잦다.
- Teachable Machine 자체는 **객체 감지 학습을 지원하지 않는다** → 학습 파이프라인을 통째로 바꿔야 한다.

### 객체 감지로 바꾸면 얻는 것

- 바운딩 박스 → **방향**(박스 중심 x) + **근접도**(박스 면적·하단 y) 계산 가능
- 방향/거리/종류별로 **진동 패턴을 다르게** 줄 수 있음 (예: 가까울수록 짧고 빠르게)
- 기존 히스테리시스·진동·UI 코드는 거의 그대로 재사용 가능. 바뀌는 곳은 `loadModel()`, `model.predict()`, `getDangerFromPredictions()` 세 군데가 핵심.

---

## 2. 후보 방안 비교

| | A. COCO-SSD (사전학습) | B. YOLO + ONNX Runtime Web | C. YOLO + LiteRT.js | D. MediaPipe Object Detector | E. Roboflow + inferencejs |
|---|---|---|---|---|---|
| 학습 필요 | 없음 | Colab에서 직접 | Colab에서 직접 | Model Maker (Colab) | Roboflow 웹 UI (노코드) |
| 계단/횡단보도/기둥 감지 | **불가** (COCO 80클래스에 없음) | 가능 | 가능 | 가능 | 가능 |
| 브라우저 가속 | WebGL | **WebGPU** + WASM 폴백 | **WebGPU** + WASM/WebNN | WASM/GPU | WebGL(TF.js) |
| 빌드 도구 | 불필요 (script 태그) | 불필요 (CDN 가능) | npm 번들러 필요 | 불필요 (script 태그) | npm 번들러 필요 |
| 계정/키 | 없음 | 없음 | 없음 | 없음 | **필요** (publishable key) |
| 오프라인 동작 | 가능 | 가능 | 가능 | 가능 | 모델 로드 시 네트워크 필요 |
| 생태계 성숙도 | 패키지 3년간 미갱신 | 매우 성숙 | 2026-07 출시, 초기 | Model Maker **유지보수 중단** 고지 | 성숙, 다만 요금제 변경(2026-07-30) |
| 난이도 | ★ | ★★★ | ★★★★ | ★★ | ★★ |

### A. `@tensorflow-models/coco-ssd` — 파이프라인 검증용

- 설치 없이 script 태그 한 줄. `model.detect(video)` → `[{class, score, bbox:[x,y,w,h]}]`.
- 우리가 원하는 클래스가 없으므로 최종 해법은 아님. 다만 **"박스 → 방향/거리 → 진동" 로직과 스마트폰 성능을 하루 만에 검증**하는 0단계로 가치가 있다.
- `person`, `car`, `bicycle`, `bus`, `traffic light`, `fire hydrant`, `bench` 등은 보조 위험물로 하이브리드 구성에 쓸 수 있다.

### B. Ultralytics YOLO(11n/26n) 커스텀 학습 + ONNX Runtime Web — **1순위 추천**

1. 데이터: Roboflow Universe에 시각장애인 보행 보조용 공개 데이터셋이 이미 여럿 있다 (crosswalk / stairs / pole / obstacle 클래스 포함, 수천~2.5만 장). 필요 시 직접 촬영한 사진을 추가 라벨링.
2. 학습: Colab 무료 GPU에서 `yolo26n` 또는 `yolo11n`을 imgsz 320~416으로 파인튜닝 (수십 분).
3. 내보내기: `model.export(format="onnx", opset=12, dynamic=True)` (WebGPU 호환을 위해 opset 12 권장). NMS를 그래프에 포함시키는 옵션도 있어 후처리를 줄일 수 있다.
4. 브라우저: `onnxruntime-web`을 CDN으로 로드, `executionProviders: ['webgpu','wasm']`로 자동 폴백. 전처리(letterbox 리사이즈)와 후처리(NMS)는 JS로 구현. 공개 레퍼런스 구현이 많다 (예: nomi30701/yolo-object-detection-onnxruntime-web).
5. 성능: YOLO11n 기준 WebGPU에서 수십 ms, WASM 폴백은 수백 ms. 스마트폰에서는 imgsz 320 + 프레임 스킵 + Web Worker 권장.

- 장점: 벤더 종속 없음, 오프라인, 모델 교체 자유, 커뮤니티 자료 최다.
- 단점: Python 학습 과정 필요, 전·후처리 코드 직접 작성.
- 주의: **Ultralytics의 `format="tfjs"` export는 8.4.83부터 폐기**되어 `.tflite`(LiteRT)로 대체되었다. 기존 TF.js 런타임을 유지하려면 ONNX 경로가 안전하다.

### C. YOLO + LiteRT.js (`@litertjs/core`, `@ultralytics/yolo`)

- Google이 2026-07-09 공개한 새 웹 런타임. WebGPU/WebNN 가속, TF.js 대비 최대 3배 빠르다고 발표. Ultralytics가 공식 통합을 제공한다.
- 단점: 출시 3개월, 예제 부족, npm 번들러 필요, export가 Linux/macOS에서만 지원(Colab은 OK).
- 판단: 지금 당장 주력으로 쓰기엔 이르지만, B에서 학습한 **같은 `.pt` 모델을 `.tflite`로 다시 export하면 그대로 옮길 수 있으므로** 추후 성능 업그레이드 경로로 남겨둔다.

### D. MediaPipe Object Detector (`@mediapipe/tasks-vision`)

- 브라우저 API가 가장 단순하다: `ObjectDetector.createFromOptions({modelAssetPath})` → `detectForVideo(video, ts)` → `detections[].boundingBox{originX,originY,width,height}` + `categories[]{score,categoryName}`. script 태그만으로 동작.
- 커스텀 학습은 **MediaPipe Model Maker**(EfficientDet-Lite0)로 하는데, 공식 문서에 "still available, but is no longer actively maintained" 고지가 붙어 있다. Colab에서 버전 고정으로 아직 돌릴 수는 있으나 장기 리스크.
- 판단: 코드가 가장 적게 드는 대안. 학습 도구 중단 리스크를 감수할 수 있다면 2순위.

### E. Roboflow 노코드 학습 + `inferencejs`

- Teachable Machine과 가장 비슷한 경험: 웹에서 사진 업로드 → 박스 그리기 → 자동 학습 → 배포.
- 브라우저 추론은 `inferencejs`(내부적으로 TF.js)로 `startWorkerByModelId(modelId, publishableKey)` → `infer(workerId, new CVImage(video))` → `[{class, confidence, bbox{x,y,width,height}}]`.
- 제약: 계정 + publishable key 필수, 무료(Public) 플랜은 데이터셋·모델을 **공개**해야 하며 2026-07-30부터 월 15 크레딧으로 축소. 브라우저(온디바이스) 추론 자체는 크레딧을 쓰지 않지만 학습은 쓴다. 모델 로드에 네트워크 필요.
- 판단: Python을 피하고 싶을 때 현실적인 선택. 다만 데이터셋 라벨링만 Roboflow에서 하고, **export한 YOLO 가중치로 B 경로를 타는 혼합 방식**이 가장 유연하다.

### F. 하이브리드 (기존 분류기 유지 + 감지기 추가)

- 횡단보도·계단은 사실 "장면(scene)" 성격이라 분류기로도 꽤 잘 잡힌다. 기둥·사람·차량 같은 "물체"만 감지기로 보완하는 구성.
- 두 모델을 동시에 돌리면 스마트폰 부하가 커지므로, 프레임을 번갈아 처리하는 식의 스케줄링이 필요하다.
- 거리 추정이 꼭 필요해지면 `transformers.js`의 Depth Anything(small) 같은 단안 깊이 추정 모델을 추가하는 확장도 가능하다.

---

## 3. 추천 로드맵

1. **0단계 (반나절)** — A(COCO-SSD)로 현재 앱에 박스 오버레이 + "방향/근접도 → 진동 패턴" 로직을 먼저 붙여 스마트폰에서 FPS와 진동 UX를 검증한다.
2. **1단계 (1~2일)** — Roboflow Universe에서 crosswalk/stairs/pole 데이터셋을 받아 Colab에서 YOLO11n(또는 26n)을 파인튜닝, ONNX(opset 12)로 export.
3. **2단계 (1~2일)** — `onnxruntime-web`로 교체. `loadModel()`·`predict`·`getDangerFromPredictions()`만 바꾸고 히스테리시스·진동·플로팅 인디케이터는 유지. 추론은 Web Worker로 분리.
4. **3단계 (선택)** — 직접 촬영한 학교 주변 사진을 추가 라벨링해 재학습. WebGPU 미지원 기기는 WASM 폴백으로 커버. 성능이 아쉬우면 LiteRT.js(C)로 런타임 교체 실험.

## 4. 코드 변경 지점 (main.js 기준)

| 함수 | 변경 내용 |
|---|---|
| `loadModel()` | `tmImage.load` → ORT 세션 생성 (`ort.InferenceSession.create(url, {executionProviders:['webgpu','wasm']})`) |
| `inferenceLoop()` 내 `model.predict` | 비디오 → 텐서 전처리 → `session.run` → NMS → `[{label, score, box}]` |
| `getDangerFromPredictions()` | 확률 1개 → 박스 배열. 가장 위험한 박스 선택 기준: 클래스 가중치 × 근접도(박스 높이 / 화면 높이) |
| `vibrateDevice()` | 고정 패턴 → 방향/근접도별 패턴 (좌: 1회, 중앙: 2회, 우: 3회 등) |
| 오버레이 | 전체 반투명 칠하기 → 박스 + 라벨 그리기 |
| `index.html` | TM 스크립트 제거, `onnxruntime-web` CDN 추가, 예측 막대 → 감지 목록 |

## 5. 참고 링크

- Ultralytics TF.js export 폐기 / LiteRT 안내: https://docs.ultralytics.com/integrations/tfjs/ , https://docs.ultralytics.com/integrations/litert/
- ONNX Runtime Web: https://onnxruntime.ai/docs/tutorials/web/
- YOLO + onnxruntime-web 레퍼런스: https://github.com/nomi30701/yolo-object-detection-onnxruntime-web
- LiteRT.js: https://developers.google.com/edge/litert/web
- MediaPipe Object Detector (Web): https://developers.google.com/edge/mediapipe/solutions/vision/object_detector/web_js
- MediaPipe Model Maker (유지보수 중단 고지): https://developers.google.com/edge/mediapipe/solutions/model_maker
- Roboflow inferencejs 레퍼런스: https://docs.roboflow.com/deploy/sdks/web-browser/web-inference.js/inferencejs-reference
- Roboflow 요금: https://roboflow.com/pricing
- Roboflow Universe 데이터셋 검색: https://universe.roboflow.com/search?q=class:crosswalk , https://universe.roboflow.com/search?q=class:stairs
- COCO-SSD: https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd
