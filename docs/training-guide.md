# 학습 가이드: SafeWalkBD v9 → YOLO11n → ONNX

작성일: 2026-10-14

## 준비된 것

| 파일 | 역할 |
|---|---|
| `tools/merge_yolo_datasets.py` | 여러 YOLO 데이터셋을 합치고 클래스 이름을 통일 |
| `tools/train_colab.ipynb` | Colab에서 다운로드 → 병합 → 학습 → ONNX 내보내기까지 한 번에 |

## 최종 클래스 (ID 순서)

| ID | 클래스 | 앱에서의 의미 | 출처 |
|---|---|---|---|
| 0 | crosswalk | 횡단보도 | Crosswalk |
| 1 | stairs | 계단 | Stairs |
| 2 | pole | 기둥, 가로등 | Pole |
| 3 | person | 보행자 충돌 | Person |
| 4 | vehicle | 차량 | Vehicle + Train |

기존 앱의 3개 클래스에 person, vehicle을 더했다. 3개만 쓰려면 병합 시
`--classes crosswalk stairs pole` 을 주면 된다.

SafeWalkBD의 Animal, Obstacle, Over-bridge, Pothole, Railway, Road-barrier, Sidewalk,
Traffic-light, Traffic-sign, Tree 는 기본적으로 **제외**된다.
살리고 싶으면 `--map "Over-bridge=stairs"` 처럼 연결한다.

## 데이터셋 현황 (SafeWalkBD.v9i.yolov8, 로컬 dry-run 기준)

| split | 이미지(라벨 있음) | crosswalk | stairs | pole | person | vehicle |
|---|---|---|---|---|---|---|
| train | 3,747 | 176 | 971 | 3,121 | 3,044 | 5,020 |
| valid | 1,046 | 52 | 258 | 894 | 787 | 1,509 |
| test | 645 | 23 | 146 | 519 | 591 | 1,103 |

**crosswalk 박스가 176개뿐**이라 횡단보도 정확도는 낮게 나올 가능성이 크다.
정확도가 부족하면 Roboflow Universe의 crosswalk 전용 데이터셋을 하나 더 받아 `ZIPS`에 추가하면 된다.

## Roboflow 쪽에서 해야 할 것 (3-B SDK 방식을 쓸 때만)

1. 포크한 프로젝트마다 **Generate → Create** 로 버전을 하나 만든다. 버전이 없으면 SDK 다운로드가 실패한다.
   - 전처리는 Resize 정도만, Augmentation은 끄는 편이 낫다. Ultralytics가 학습 중에 증강을 한다.
2. 프로젝트 URL `https://app.roboflow.com/<WORKSPACE>/<PROJECT>/...` 에서 workspace와 project 슬러그를 적어 둔다.
3. Settings → API Keys 에서 Private API Key 를 복사해 둔다. 노트북 입력창에만 붙여넣고 파일에는 적지 않는다.

## Colab 실행 순서

1. `SafeWalkBD.v9i.yolov8.zip` 을 Google Drive "내 드라이브" 바로 아래에 올린다
2. [Colab](https://colab.research.google.com/) 에서 `tools/train_colab.ipynb` 업로드 (파일 → 노트 업로드)
3. 런타임 유형을 **T4 GPU** 로 변경
4. 2번 셀에서 `tools/merge_yolo_datasets.py` 업로드, 3-A 셀에서 Drive 연결 (3-B 는 건너뜀)
5. 셀을 순서대로 실행. 4-a(dry-run) 출력에서 매핑을 확인하고 필요하면 `EXTRA_MAP` 수정 후 재실행
6. 6번 셀에서 mAP50 확인. 0.6 이상이면 쓸 만하고, 0.7 이상이면 좋다
7. 9번 셀로 `best.onnx`, `labels.json` 다운로드 → 저장소 루트에 복사

## 로컬에서 병합 스크립트만 쓰기

Roboflow에서 YOLOv8 형식 zip을 직접 받았다면 Colab 없이도 병합할 수 있다.

```bash
py -3 tools/merge_yolo_datasets.py --src "C:\Users\user\Downloads\SafeWalkBD.v9i.yolov8" --out datasets/merged --dry-run
```

이 PC에서는 `python` 대신 `py -3` 을 써야 한다. `python` 은 Windows 스토어 스텁이라 실행되지 않는다.

## 자주 막히는 지점

- **다운로드 403/404**: 버전을 안 만들었거나 슬러그 오타. URL을 다시 확인.
- **박스가 0개인 클래스 경고**: 원본 클래스 이름이 별칭 목록에 없는 것. `--map` 으로 연결.
- **GPU 메모리 부족**: `batch=16` 으로 낮춘다.
- **nms=True 내보내기 실패**: 노트북이 자동으로 NMS 없는 ONNX로 재시도한다. `labels.json` 의 `nms_in_model` 값으로 웹앱이 구분한다.
