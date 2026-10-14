# 위험지대 감지 + 진동 알림 (스마트폰용)

실시간으로 위험지대를 감지하고 스마트폰 진동으로 알림을 제공하는 시각장애인 보조 웹앱입니다.

## 🎯 주요 기능

- 📱 스마트폰 후면 카메라로 실시간 위험지대 감지
- 🔔 85% 이상 정확도로 위험 감지 시 진동 알림
- 🧠 YOLO 객체 감지 모델 (ONNX Runtime Web, WebGPU/WASM) 사용
- 📍 박스 위치로 방향(좌/정면/우)과 근접도를 계산해 진동 패턴을 다르게 알림
- 🌐 클라우드 기반 - 설치 불필요

## 🏷️ 감지 가능한 위험

- **계단** - 낙상 위험
- **횡단보도** - 신호 확인 필요
- **기둥&가로등** - 충돌 위험

## 🚀 사용 방법

1. 스마트폰 브라우저(Chrome/Safari)에서 배포된 URL 접속
2. **📷 카메라 시작** 클릭 → 카메라 권한 허용
3. **▶️ 감지 시작** 클릭
4. 스마트폰을 전방으로 향하게 유지
5. 위험 감지 시 자동으로 진동 발생!

## 🛠️ 기술 스택

- **ONNX Runtime Web 1.30** - 브라우저 AI 추론 (WebGPU 우선, WASM 폴백)
- **Ultralytics YOLO11n** - 객체 감지 (학습 방법: `docs/training-guide.md`)
- **Vibration API** - 스마트폰 진동
- **MediaDevices API** - 카메라 접근

## ⚙️ 설정

### 감지 임계치
- 기본값: **0.85 (85%)**
- 이 확률 이상일 때만 위험으로 판단

### 확정 프레임
- 기본값: **4프레임**
- 동일 위험이 연속으로 감지되어야 진동 (오감지 방지)

## 🌐 배포

### Netlify 배포 (권장)

1. GitHub 저장소 생성 및 코드 푸시
2. [Netlify](https://app.netlify.com/) 접속
3. **Import from Git** → 저장소 선택
4. **Deploy site** 클릭

### 로컬 테스트

```bash
# Python 서버 사용
python -m http.server 8000

# 또는 start_server.bat 실행
start_server.bat

# 브라우저에서 접속
http://localhost:8000
```

## 📱 브라우저 호환성

- ✅ Chrome (Android/Desktop)
- ✅ Safari (iOS 13+)
- ✅ Edge (Windows/Android)
- ⚠️ Firefox (진동 API 제한적)

## 🔒 권한 요구사항

- **카메라 접근** - 위험 감지용 (필수)
- **인터넷 연결** - AI 모델 로드용 (필수)
- ⚠️ **HTTPS 환경** 필수 (Netlify는 자동 제공)

## 📊 AI 모델 정보

- **모델 파일**: `best.onnx` + `labels.json` (Colab 노트북 `tools/train_colab.ipynb` 로 생성)
- **입력 크기**: 320x320 pixels (letterbox)
- **클래스**: crosswalk, stairs, pole, person, vehicle
- **테스트 모드**: `index.html?model=다른모델.onnx&labels=labels_coco.json&image=사진.jpg` 로 카메라 없이 정지 이미지 검사 가능
- 이전 Teachable Machine 모델 파일(`model.json`, `weights.bin`, `metadata.json`)은 새 모델 검증 후 삭제 예정

## 🐛 문제 해결

### 카메라가 작동하지 않아요
- HTTPS 환경인지 확인
- 브라우저에서 카메라 권한 허용 확인
- 다른 앱이 카메라를 사용중이지 않은지 확인

### 진동이 작동하지 않아요
- iOS Safari는 진동 API를 지원하지 않음
- Android Chrome 사용 권장
- 무음 모드에서도 진동은 작동함

### 모델 로드가 실패해요
- 인터넷 연결 확인
- F12 → Console 탭에서 오류 확인
- Teachable Machine CDN 접근 가능 여부 확인

## 📝 라이선스

MIT License

## 🙏 감사의 말

- [Teachable Machine](https://teachablemachine.withgoogle.com/) - Google
- [TensorFlow.js](https://www.tensorflow.org/js) - Google







