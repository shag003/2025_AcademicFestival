import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib
import platform
import os

# --- 그래프 폰트 깨짐 방지 설정 ---
if platform.system() == 'Windows':
    font_path = "C:/Windows/Fonts/NanumGothic.ttf"
    if os.path.exists(font_path):
        matplotlib.font_manager.fontManager.addfont(font_path)
        plt.rc('font', family='NanumGothic')
    else:
        plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='DejaVu Sans')

plt.rcParams['axes.unicode_minus'] = False

# ------------------------------
# YOLO 모델 불러오기
# ------------------------------
yolo_model = YOLO("yolov8n.pt")

# 뉴런 전위값 범위 및 임계값 설정
RESTING_POTENTIAL = -70  # 기본값(-70mV)
THRESHOLD_POTENTIAL = -55
ACTION_POTENTIAL = 40

# ------------------------------
# 특징 추출 함수
# ------------------------------
def extract_person_features(img_path):
    img = cv2.imread(img_path)
    results = yolo_model(img)
    person_boxes = [box.xyxy.cpu().numpy() for box in results[0].boxes if int(box.cls[0]) == 0]
    features_list = []
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box[0])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or (x2-x1)*(y2-y1) < 500:
            continue
        crop = cv2.resize(crop, (128, 128))
        brightness = np.mean(crop) / 255.0
        color_std = np.std(crop) / 128.0
        saturation = np.mean(np.max(crop,2) - np.min(crop,2)) / 255.0
        complexity = cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var() / 1000.0
        features_list.append([brightness, color_std, saturation, complexity])

    if features_list:
        features = np.mean(features_list, axis=0)
    else:
        img_small = cv2.resize(img, (128, 128))
        brightness = np.mean(img_small) / 255.0
        color_std = np.std(img_small) / 128.0
        saturation = np.mean(np.max(img_small,2) - np.min(img_small,2)) / 255.0
        complexity = cv2.Laplacian(cv2.cvtColor(img_small,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var() / 1000.0
        features = np.array([brightness, color_std, saturation, complexity], dtype=np.float32)
    return features

# ------------------------------
# 그룹별 선호도 계산 함수
# ------------------------------
def preference_highdev(features):
    weights = np.array([5, 5, 5, 5])
    score = np.dot(features, weights) / (np.sum(weights) + 1)
    return np.clip(score, 0, 1)

def preference_lowdev(features):
    weights = np.array([1.0, 0.1, 0.1, 0.1])
    score = np.dot(features, weights) / (np.sum(weights) + 1)
    return np.clip(score, 0, 1)

# ------------------------------
# 그룹별 gain/offset 적용
# ------------------------------
def apply_group_gain_offset(score, group):
    if group == '발달':
        return np.clip(1.1 * score + 0.05, 0, 1)
    else:  # 비발달
        return np.clip(0.8 * score - 0.05, 0, 1)

# ------------------------------
# 시그모이드 기반 점수→막전위 매핑
# ------------------------------
def score_to_membrane_potential_sigmoid(score, k=6.0, center=0.55):
    s = 1.0 / (1.0 + np.exp(-k * (score - center)))
    return RESTING_POTENTIAL + s * (ACTION_POTENTIAL - RESTING_POTENTIAL)

# ------------------------------
# 활동전위 판정
# ------------------------------
def action_potential_yn(membrane_potential):
    return 1 if membrane_potential >= THRESHOLD_POTENTIAL else 0

# ------------------------------
# 결과 그래프
# ------------------------------
def graph_results(potentials_high, potentials_low, ap_high, ap_low, repeat):
    x = np.arange(1, repeat + 1)
    plt.figure(figsize=(10,5))
    plt.plot(x, potentials_high, label='발달그룹 전위', marker='o')
    plt.plot(x, potentials_low, label='비발달그룹 전위', marker='s')
    plt.axhline(THRESHOLD_POTENTIAL, color='red', linestyle='--', label='역치전위(-55mV)')
    plt.fill_between(x, RESTING_POTENTIAL, ACTION_POTENTIAL, where=np.array(ap_high)==1, color='blue', alpha=0.08, label='발달그룹 활동전위')
    plt.fill_between(x, RESTING_POTENTIAL, ACTION_POTENTIAL, where=np.array(ap_low)==1, color='green', alpha=0.08, label='비발달그룹 활동전위')
    plt.xlabel('반복 횟수')
    plt.ylabel('막 전위 (mV)')
    plt.title('반복 평가별 막 전위 및 활동전위')
    plt.legend()
    plt.ylim(RESTING_POTENTIAL-5, ACTION_POTENTIAL+5)
    plt.grid(True)
    plt.show()

# ------------------------------
# GUI 레이아웃 (먼저 정의)
# ------------------------------
root = tk.Tk()
root.title("사람 이미지 선호도/동기(뉴런 전위 기반) 반복 그래프")
root.geometry("540x420")

frame = tk.Frame(root)
frame.pack()
btn = tk.Button(frame, text="이미지 선택 및 평가")
btn.pack(side="left", pady=10, padx=5)

repeat_label = tk.Label(frame, text="반복 횟수:")
repeat_label.pack(side="left", padx=5)
repeat_entry = tk.Entry(frame, width=5)
repeat_entry.insert(0, "10")
repeat_entry.pack(side="left", padx=5)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("맑은 고딕", 13), justify="left")
result_label.pack(pady=5)

# ------------------------------
# 메인 함수 (위젯 참조 가능)
# ------------------------------
def open_and_score():
    file_path = filedialog.askopenfilename(
        title="이미지 파일 선택", 
        filetypes=[("이미지 파일", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        result_label.config(text="이미지를 선택하지 않았습니다.")
        image_label.config(image='')
        return

    pil_img = Image.open(file_path)
    pil_img = pil_img.resize((180, 180))
    tk_img = ImageTk.PhotoImage(pil_img)
    image_label.config(image=tk_img)
    image_label.image = tk_img

    try:
        repeat = int(repeat_entry.get())
    except:
        result_label.config(text="반복 횟수를 정확히 입력하세요.")
        return

    potentials_high, potentials_low = [], []
    ap_high, ap_low = [], []

    for _ in range(repeat):
        features = extract_person_features(file_path)
        noise = np.random.normal(0, 0.02, features.shape)
        features_noisy = np.clip(features + noise, 0, 1)

        # 1) 그룹별 score 계산
        high_score = preference_highdev(features_noisy)
        low_score  = preference_lowdev(features_noisy)

        # 2) 그룹별 gain/offset 적용
        high_score_adj = apply_group_gain_offset(high_score, '발달')
        low_score_adj  = apply_group_gain_offset(low_score,  '비발달')

        # 3) 시그모이드 매핑 → 막전위 변환
        mem_high = score_to_membrane_potential_sigmoid(high_score_adj, k=6.0, center=0.55)
        mem_low  = score_to_membrane_potential_sigmoid(low_score_adj,  k=6.0, center=0.55)

        potentials_high.append(mem_high)
        potentials_low.append(mem_low)
        ap_high.append(action_potential_yn(mem_high))
        ap_low.append(action_potential_yn(mem_low))

    avg_low  = np.mean(potentials_low)
    avg_high = np.mean(potentials_high)

    # spike 발생 여부도 평균을 기준으로 표시 가능 (평균 전위가 역치 넘는지)
    ap_low_avg  = 'O' if avg_low  >= THRESHOLD_POTENTIAL else 'X'
    ap_high_avg = 'O' if avg_high >= THRESHOLD_POTENTIAL else 'X'

    result = (
        f"파일: {os.path.basename(file_path)}\n"
        f"[비발달 그룹]\n"
        f"  평균 막 전위: {avg_low:.1f} mV\n"
        f"  활동전위 유발: {ap_low_avg}\n"
        f"[발달 그룹]\n"
        f"  평균 막 전위: {avg_high:.1f} mV\n"
        f"  활동전위 유발: {ap_high_avg}\n"
        f"(그래프에서 반복별 변동 확인)"
    )
    result_label.config(text=result)
    graph_results(potentials_high, potentials_low, ap_high, ap_low, repeat)

# 버튼에 함수 연결
btn.config(command=open_and_score)

# 실행
root.mainloop()
