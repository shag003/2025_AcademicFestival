# 25학년도 심리학과 학술제 2⁷조

---

## SimpleANN 이미지 비교 도구

간단한 GUI 기반 이미지 비교/평가 도구

이미지를 입력으로 받아 사람 ROI(가능한 경우) 기반 특징을 추출하고, 색채도(colorfulness)와 사전 정의된 선호도 모델을 이용해 "발달(dev)" / "비발달(non)" 스타일로 각 이미지의 "평균 막전위"를 시뮬레이션하여 반복 실험 결과를 비교합니다. 통계적 비교(paired t-test 유무, Cohen's d 포함)와 반복 결과의 시각화를 제공합니다.

---

## 주요 기능 요약

- GUI 기반 이미지 선택/비교 (Tkinter)
- 자동 모드 감지: YOLO(person) 우선 → Haar cascade(얼굴) 대체
- 128x128 패치에서 기본 특징 4개(brightness, color_std, saturation, complexity) 추출
- Hasler 기반 근사 colorfulness 계산 (ROI 우선)
- 발달/비발달용 선호도 모델 및 조건부 색채도 부스트
- 반복 노이즈 시뮬레이션, paired t-test 및 Cohen's d 계산
- 반복별 막전위 시각화(matplotlib)
- evaluate_pair_auto 함수로 GUI 없이도 사용 가능

---

## 설치 (requirements.txt 사용)

의존 패키지는 프로젝트 루트의 requirements.txt에 정리되어 있습니다. 권장 설치 방법:

1. 가상환경(선택):
   - python -m venv .venv
   - source .venv/bin/activate  (Windows: .venv\Scripts\activate)

2. 의존성 설치:
   - pip install -r requirements.txt

requirements.txt 파일에 ultralytics (YOLO 사용 시)와 scipy(통계)를 포함했는지 확인하세요. ultralytics와 YOLO 모델은 선택적이며, 없을 경우 OpenCV Haar cascade로 얼굴 탐지 대체됩니다.

---

## 실행 방법

- GUI 실행:
  - python 2025_academic_festival_simpleANN.py
  - GUI에서 이미지 2개를 선택하고 반복 횟수 입력 후 "평가 실행" 클릭

- 코드로 호출:
  - from 2025_academic_festival_simpleANN import evaluate_pair_auto
  - summary = evaluate_pair_auto("img1.jpg", "img2.jpg", repeat=50, sigma=0.01)

(모듈명으로 import할 때 파일명에 숫자/특수문자가 문제될 수 있으므로 필요하면 파일명 변경 권장)

---

## 파일/파라미터 요약

- 주요 상수: MODEL_PATH, RESTING, THRESHOLD, ACTION, BASE_ALPHA, CF_THRESHOLD, DEFAULT_NOISE_SIGMA, DEFAULT_REPEAT, P_EXP, BETA
- 가중치 벡터: PREF_HIGH_WEIGHTS, PREF_LOW_WEIGHTS
- 난수 재현: RNG_SEED

---

## 제한 및 주의점

- YOLO 모델 미설치 시 사람 탐지 성능이 낮아질 수 있음.
- 색채도 계산은 조명·화이트밸런스에 민감.
- 통계 신뢰성은 repeat와 sigma에 의해 영향받음.
- 헤드리스 환경에서는 GUI 대신 evaluate_pair_auto를 사용하세요.

---