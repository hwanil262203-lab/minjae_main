/*
  Teachable Machine + Smartphone Vibration
  - Loads TM image model from server files (keras_model.h5 + labels.txt)
  - Runs webcam inference loop
  - Applies threshold (85%) + steady-frame hysteresis
  - Vibrates smartphone when danger detected
*/

const ui = {
  video: document.getElementById('video'),
  overlay: document.getElementById('overlay'),
  btnCamera: document.getElementById('btnCamera'),
  btnStart: document.getElementById('btnStart'),
  btnStop: document.getElementById('btnStop'),
  threshold: document.getElementById('threshold'),
  steadyFrames: document.getElementById('steadyFrames'),
  status: document.getElementById('status'),
  log: document.getElementById('log'),
  floatingIndicator: document.getElementById('floatingIndicator'),
};

let DANGER_LABELS = []; // 위험 라벨들 (labels.txt에서 자동 로드)

let model = null;
let webcamStream = null;
let isRunning = false;
let animationHandle = null;
let isVibrating = false;

const log = (msg) => {
  const t = new Date().toLocaleTimeString();
  ui.log.textContent += `[${t}] ${msg}\n`;
  ui.log.scrollTop = ui.log.scrollHeight;
};

function setStatus(text) {
  ui.status.textContent = text;
}

async function openCamera() {
  if (webcamStream) {
    log('카메라가 이미 실행 중입니다.');
    return;
  }
  try {
    log('카메라 권한 요청 중...');
    webcamStream = await navigator.mediaDevices.getUserMedia({ 
      video: { 
        facingMode: 'environment',
        width: { ideal: 640 },
        height: { ideal: 480 }
      }, 
      audio: false 
    });
    ui.video.srcObject = webcamStream;
    
    // 비디오가 준비될 때까지 대기
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('비디오 로드 타임아웃'));
      }, 10000);
      
      ui.video.onloadedmetadata = () => {
        clearTimeout(timeout);
        ui.overlay.width = ui.video.videoWidth;
        ui.overlay.height = ui.video.videoHeight;
        resolve();
      };
      
      // 이미 메타데이터가 로드된 경우
      if (ui.video.readyState >= 1) {
        clearTimeout(timeout);
        ui.overlay.width = ui.video.videoWidth;
        ui.overlay.height = ui.video.videoHeight;
        resolve();
      }
    });
    
    await ui.video.play();
    
    log(`✓ 카메라 시작됨 (${ui.video.videoWidth}x${ui.video.videoHeight})`);
    console.log('비디오 크기:', ui.video.videoWidth, 'x', ui.video.videoHeight);
  } catch (e) {
    log(`✗ 카메라 오류: ${e.message}`);
    console.error('카메라 오류:', e);
  }
}

async function loadModel() {
  try {
    setStatus('모델 로딩 중...');
    log('로컬 모델 파일 로딩 중...');
    
    // 로컬 모델 파일 사용
    const modelURL = './model.json';
    const metadataURL = './metadata.json';
    
    model = await tmImage.load(modelURL, metadataURL);
    
    // metadata에서 라벨 추출
    DANGER_LABELS = model.getClassLabels();
    
    log(`✓ 모델 로드 완료!`);
    log(`감지 라벨: ${DANGER_LABELS.join(', ')}`);
    setStatus('모델 준비 완료');
  } catch (e) {
    log(`✗ 모델 로드 실패: ${e.message}`);
    console.error('모델 로드 오류:', e);
    setStatus('모델 로드 실패');
  }
}

function vibrateDevice() {
  if (!('vibrate' in navigator)) {
    return;
  }
  // 2번 진동 패턴: [진동, 멈춤, 진동]
  if (!isVibrating) {
    isVibrating = true;
    const success = navigator.vibrate([400, 200, 400]); // 400ms 진동, 200ms 멈춤, 400ms 진동
    if (!success) {
      isVibrating = false;
    }
  }
}

function stopVibration() {
  if (isVibrating) {
    navigator.vibrate(0);
    isVibrating = false;
  }
}

function getDangerFromPredictions(predictions, threshold) {
  // predictions: [{className, probability}] from Teachable Machine
  // '평지'는 제외하고 위험 지대만 감지
  const dangerClasses = ['횡단보도', '계단', '기둥, 가로등'];
  
  let best = { label: null, prob: 0 };
  for (let i = 0; i < predictions.length; i++) {
    const p = predictions[i];
    // '평지'는 무시
    if (p.className === '평지') continue;
    
    // 위험 지대만 체크
    if (dangerClasses.includes(p.className) && p.probability > best.prob) {
      best = { label: p.className, prob: p.probability };
    }
  }
  
  // 임계치 이상일 때만 위험으로 판단
  if (best.label && best.prob >= threshold) return best;
  return { label: null, prob: 0 };
}

const hysteresisState = {
  lastLabel: null,
  steadyCount: 0,
};

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

async function inferenceLoop() {
  if (!model || !webcamStream) {
    log('✗ 모델 또는 카메라가 준비되지 않았습니다.');
    return;
  }
  isRunning = true;
  setStatus('감지 중...');
  log('예측 시작됨');
  const ctx = ui.overlay.getContext('2d');

  const step = async () => {
    if (!isRunning) return;
    try {
      // Teachable Machine 모델로 예측
      const predictions = await model.predict(ui.video);
      
      // 디버그: 첫 예측 시 로그
      if (!window.firstPredictionLogged) {
        console.log('첫 예측 결과:', predictions);
        log(`예측 결과 받음: ${predictions.length}개 클래스`);
        window.firstPredictionLogged = true;
      }

      // 예측 결과 시각화 업데이트
      updatePredictionBars(predictions);

      const threshold = parseFloat(ui.threshold.value) || 0.85;
      const requiredFrames = parseInt(ui.steadyFrames.value, 10) || 4;
      const best = getDangerFromPredictions(predictions, threshold);

      const confirmed = updateHysteresis(best.label, requiredFrames);

      // Draw overlay
      ctx.clearRect(0, 0, ui.overlay.width, ui.overlay.height);
      if (best.label) {
        ctx.fillStyle = 'rgba(255, 87, 34, 0.25)';
        ctx.fillRect(0, 0, ui.overlay.width, ui.overlay.height);
        ctx.fillStyle = '#ffb199';
        ctx.font = '16px ui-monospace, monospace';
        ctx.fillText(`⚠️ ${best.label} ${(best.prob * 100).toFixed(1)}%`, 12, 28);
      }

      // Floating Indicator 업데이트 (90% 이상일 때 위험으로 표시)
      const isDangerForIndicator = best.label && best.prob >= 0.90;
      updateFloatingIndicator(isDangerForIndicator);

      // 진동 제어 (일정한 강도)
      if (confirmed) {
        vibrateDevice();
      } else {
        stopVibration();
      }
    } catch (e) {
      console.error('예측 오류:', e);
      log(`감지 오류: ${e.message}`);
    }
    animationHandle = requestAnimationFrame(step);
  };
  animationHandle = requestAnimationFrame(step);
}

function updatePredictionBars(predictions) {
  // 예측 결과를 막대 그래프로 업데이트
  for (let i = 0; i < predictions.length && i < 4; i++) {
    const prob = predictions[i].probability;
    const percent = (prob * 100).toFixed(1);
    
    const barElement = document.getElementById(`pred-${i}-bar`);
    const percentElement = document.getElementById(`pred-${i}-percent`);
    
    if (barElement && percentElement) {
      barElement.style.width = `${percent}%`;
      percentElement.textContent = `${percent}%`;
      
      // 평지가 아니고 90% 이상이면 위험 색상
      const className = predictions[i].className;
      if (className !== '평지' && prob >= 0.90) {
        barElement.classList.add('danger');
      } else {
        barElement.classList.remove('danger');
      }
    }
  }
}

function stopLoop() {
  isRunning = false;
  stopVibration();
  setStatus('정지됨');
  if (animationHandle) cancelAnimationFrame(animationHandle);
}

ui.btnCamera.addEventListener('click', openCamera);
ui.btnStart.addEventListener('click', inferenceLoop);
ui.btnStop.addEventListener('click', stopLoop);

// ============================================
// 드래그 가능한 Floating Indicator 기능
// ============================================
let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;
let indicatorStartX = 0;
let indicatorStartY = 0;

// 마우스 이벤트
ui.floatingIndicator.addEventListener('mousedown', (e) => {
  isDragging = true;
  dragStartX = e.clientX;
  dragStartY = e.clientY;
  const rect = ui.floatingIndicator.getBoundingClientRect();
  indicatorStartX = rect.left;
  indicatorStartY = rect.top;
  ui.floatingIndicator.classList.add('dragging');
  e.preventDefault();
});

document.addEventListener('mousemove', (e) => {
  if (!isDragging) return;
  const deltaX = e.clientX - dragStartX;
  const deltaY = e.clientY - dragStartY;
  ui.floatingIndicator.style.left = (indicatorStartX + deltaX) + 'px';
  ui.floatingIndicator.style.top = (indicatorStartY + deltaY) + 'px';
  ui.floatingIndicator.style.right = 'auto';
});

document.addEventListener('mouseup', () => {
  if (isDragging) {
    isDragging = false;
    ui.floatingIndicator.classList.remove('dragging');
  }
});

// 터치 이벤트 (모바일)
ui.floatingIndicator.addEventListener('touchstart', (e) => {
  isDragging = true;
  const touch = e.touches[0];
  dragStartX = touch.clientX;
  dragStartY = touch.clientY;
  const rect = ui.floatingIndicator.getBoundingClientRect();
  indicatorStartX = rect.left;
  indicatorStartY = rect.top;
  ui.floatingIndicator.classList.add('dragging');
  e.preventDefault();
});

document.addEventListener('touchmove', (e) => {
  if (!isDragging) return;
  const touch = e.touches[0];
  const deltaX = touch.clientX - dragStartX;
  const deltaY = touch.clientY - dragStartY;
  ui.floatingIndicator.style.left = (indicatorStartX + deltaX) + 'px';
  ui.floatingIndicator.style.top = (indicatorStartY + deltaY) + 'px';
  ui.floatingIndicator.style.right = 'auto';
  e.preventDefault();
});

document.addEventListener('touchend', () => {
  if (isDragging) {
    isDragging = false;
    ui.floatingIndicator.classList.remove('dragging');
  }
});

// Floating Indicator 상태 업데이트 함수
function updateFloatingIndicator(isDanger) {
  if (isDanger) {
    ui.floatingIndicator.classList.add('danger');
    ui.floatingIndicator.textContent = '⚠';
    ui.floatingIndicator.title = '위험 감지됨! (드래그하여 이동 가능)';
  } else {
    ui.floatingIndicator.classList.remove('danger');
    ui.floatingIndicator.textContent = '✓';
    ui.floatingIndicator.title = '안전 상태 (드래그하여 이동 가능)';
  }
}

// 디버그: 스크립트 실행 확인
console.log('=== main.js 스크립트 로드됨 ===');
console.log('window 객체:', typeof window);
console.log('document 객체:', typeof document);

// 페이지 로드 시 자동으로 모델 로드
let appInitialized = false;

function checkLibraryAndInit() {
  if (appInitialized) return;
  appInitialized = true;
  
  console.log('=== 초기화 시작 ===');
  console.log('tf 타입:', typeof tf);
  console.log('tmImage 타입:', typeof tmImage);
  
  log('앱 시작됨');
  setStatus('초기화 중...');
  
  // TensorFlow.js 확인
  if (typeof tf === 'undefined') {
    log('✗ TensorFlow.js 로드 실패');
    console.error('TensorFlow.js가 로드되지 않았습니다.');
    setStatus('TF.js 로드 실패');
    return;
  }
  
  log('✓ TensorFlow.js 로드 완료');
  
  // Teachable Machine 확인
  if (typeof tmImage === 'undefined') {
    log('✗ Teachable Machine 라이브러리 로드 실패');
    console.error('Teachable Machine 라이브러리가 로드되지 않았습니다.');
    setStatus('tmImage 로드 실패');
    return;
  }
  
  log('✓ Teachable Machine 라이브러리 로드 완료');
  log('모델을 자동으로 로드합니다...');
  setStatus('모델 로딩 중...');
  loadModel();
}

// 페이지 로드 후 초기화 및 자동 시작
console.log('=== 이벤트 리스너 등록 ===');
window.addEventListener('load', () => {
  console.log('=== window.load 이벤트 발생 ===');
  setTimeout(async () => {
    console.log('=== 3초 대기 후 초기화 시작 ===');
    checkLibraryAndInit();
    
    // 모델 로드 대기
    await waitForModelLoad();
    
    // 자동으로 카메라 시작
    log('🤖 자동 시작 모드 활성화');
    await openCamera();
    
    // 카메라 시작 후 잠시 대기하고 자동으로 감지 시작
    setTimeout(() => {
      if (webcamStream && model) {
        log('🚀 자동으로 감지를 시작합니다...');
        inferenceLoop();
      }
    }, 1000);
  }, 3000);
});

// 모델이 로드될 때까지 대기하는 헬퍼 함수
function waitForModelLoad() {
  return new Promise((resolve) => {
    const checkInterval = setInterval(() => {
      if (model) {
        clearInterval(checkInterval);
        resolve();
      }
    }, 100);
    
    // 최대 30초 대기
    setTimeout(() => {
      clearInterval(checkInterval);
      resolve();
    }, 30000);
  });
}



