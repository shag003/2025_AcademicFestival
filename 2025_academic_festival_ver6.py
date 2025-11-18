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
# YOLO 모델 불러오기 (원래 코드와 동일 경로)
# ------------------------------
MODEL_PATH = "yolov8n.pt"
try:
    if not os.path.exists(MODEL_PATH):
        print(f"[경고] 모델 파일이 없습니다: {MODEL_PATH}. 모델 없이도 동작하지만 사람 검출이 안 됩니다.")
    yolo_model = YOLO(MODEL_PATH)
except Exception as e:
    print("YOLO 모델 로드 실패:", e)
    yolo_model = None

# 뉴런 전위값 범위 및 임계값 설정
RESTING_POTENTIAL = -70  # 기본값(-70mV)
THRESHOLD_POTENTIAL = -55
ACTION_POTENTIAL = 40

# ------------------------------
# 특징 추출 함수 (원본 코드 로직 유지)
# ------------------------------
def extract_person_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다.")
    # 모델이 없거나 문제 있을 경우 전체 이미지 대체 경로
    if yolo_model is None:
        img_small = cv2.resize(img, (128, 128))
        brightness = np.mean(img_small) / 255.0
        color_std = np.std(img_small) / 128.0
        saturation = np.mean(np.max(img_small,2) - np.min(img_small,2)) / 255.0
        complexity = cv2.Laplacian(cv2.cvtColor(img_small,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var() / 1000.0
        return np.array([brightness, color_std, saturation, complexity], dtype=np.float32)

    results = yolo_model(img)
    person_boxes = [box.xyxy.cpu().numpy() for box in results[0].boxes if int(box.cls[0]) == 0]
    features_list = []
    h, w = img.shape[:2]
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box[0])
        # 좌표 클램프
        x1 = max(0, min(w-1, x1))
        y1 = max(0, min(h-1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue
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
        # 사람 검출 못하면 이미지 전체로 대체
        img_small = cv2.resize(img, (128, 128))
        brightness = np.mean(img_small) / 255.0
        color_std = np.std(img_small) / 128.0
        saturation = np.mean(np.max(img_small,2) - np.min(img_small,2)) / 255.0
        complexity = cv2.Laplacian(cv2.cvtColor(img_small,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var() / 1000.0
        features = np.array([brightness, color_std, saturation, complexity], dtype=np.float32)
    return features

# ------------------------------
# 그룹별 선호도 계산 및 매핑 함수 (원본 유지)
# ------------------------------
def preference_highdev(features):
    weights = np.array([5, 5, 5, 5])
    score = np.dot(features, weights) / (np.sum(weights) + 1)
    return np.clip(score, 0, 1)

def preference_lowdev(features):
    weights = np.array([1.0, 0.1, 0.1, 0.1])
    score = np.dot(features, weights) / (np.sum(weights) + 1)
    return np.clip(score, 0, 1)

def apply_group_gain_offset(score, group):
    if group == '발달':
        return np.clip(1.1 * score + 0.05, 0, 1)
    else:  # 비발달
        return np.clip(0.8 * score - 0.05, 0, 1)

def score_to_membrane_potential_sigmoid(score, k=6.0, center=0.55):
    s = 1.0 / (1.0 + np.exp(-k * (score - center)))
    return RESTING_POTENTIAL + s * (ACTION_POTENTIAL - RESTING_POTENTIAL)

def action_potential_yn(membrane_potential):
    return 1 if membrane_potential >= THRESHOLD_POTENTIAL else 0

# ------------------------------
# 결과 그래프 (두 이미지 비교)
# ------------------------------
def graph_results(img1_high, img1_low, img2_high, img2_low, ap1_high, ap1_low, ap2_high, ap2_low, repeat):
    x = np.arange(1, repeat + 1)
    plt.figure(figsize=(11,6))
    plt.plot(x, img1_high, label='이미지1 - 발달 전위', marker='o')
    plt.plot(x, img1_low,  label='이미지1 - 비발달 전위', marker='s')
    plt.plot(x, img2_high, label='이미지2 - 발달 전위', marker='^')
    plt.plot(x, img2_low,  label='이미지2 - 비발달 전위', marker='x')
    plt.axhline(THRESHOLD_POTENTIAL, color='red', linestyle='--', label='역치전위(-55mV)')
    # fill activity areas for each image/group with different alphas/colors
    plt.fill_between(x, RESTING_POTENTIAL, ACTION_POTENTIAL, where=np.array(ap1_high)==1, color='blue', alpha=0.06)
    plt.fill_between(x, RESTING_POTENTIAL, ACTION_POTENTIAL, where=np.array(ap1_low)==1,  color='cyan', alpha=0.04)
    plt.fill_between(x, RESTING_POTENTIAL, ACTION_POTENTIAL, where=np.array(ap2_high)==1, color='green', alpha=0.06)
    plt.fill_between(x, RESTING_POTENTIAL, ACTION_POTENTIAL, where=np.array(ap2_low)==1,  color='lime', alpha=0.04)
    plt.xlabel('반복 횟수')
    plt.ylabel('막 전위 (mV)')
    plt.title('이미지별 반복 평가: 막 전위 및 활동전위 비교')
    plt.legend()
    plt.ylim(RESTING_POTENTIAL-5, ACTION_POTENTIAL+5)
    plt.grid(True)
    plt.show()

# ------------------------------
# GUI
# ------------------------------
root = tk.Tk()
root.title("두 이미지 비교 평가 (발달 vs 비발달)")
root.geometry("720x520")

top_frame = tk.Frame(root)
top_frame.pack(pady=8)

btn1 = tk.Button(top_frame, text="이미지1 선택")
btn1.grid(row=0, column=0, padx=6)
btn2 = tk.Button(top_frame, text="이미지2 선택")
btn2.grid(row=0, column=1, padx=6)

repeat_label = tk.Label(top_frame, text="반복 횟수:")
repeat_label.grid(row=0, column=2, padx=6)
repeat_entry = tk.Entry(top_frame, width=6)
repeat_entry.insert(0, "10")
repeat_entry.grid(row=0, column=3, padx=6)

eval_btn = tk.Button(top_frame, text="평가 실행")
eval_btn.grid(row=0, column=4, padx=10)

status_label = tk.Label(root, text="이미지를 선택하세요.", font=("맑은 고딕", 10))
status_label.pack()

images_frame = tk.Frame(root)
images_frame.pack(pady=6)

image1_label = tk.Label(images_frame)
image1_label.grid(row=0, column=0, padx=12)
image2_label = tk.Label(images_frame)
image2_label.grid(row=0, column=1, padx=12)

result_label = tk.Label(root, text="", font=("맑은 고딕", 12), justify="left")
result_label.pack(pady=10)

# 파일 경로 저장
img1_path = None
img2_path = None

def select_image1():
    global img1_path
    p = filedialog.askopenfilename(title="이미지1 선택", filetypes=[("이미지 파일", "*.jpg *.png *.jpeg")])
    if not p:
        return
    img1_path = p
    pil_img = Image.open(p).resize((220,220))
    tk_img = ImageTk.PhotoImage(pil_img)
    image1_label.config(image=tk_img)
    image1_label.image = tk_img
    status_label.config(text=f"이미지1 선택됨: {os.path.basename(p)}")

def select_image2():
    global img2_path
    p = filedialog.askopenfilename(title="이미지2 선택", filetypes=[("이미지 파일", "*.jpg *.png *.jpeg")])
    if not p:
        return
    img2_path = p
    pil_img = Image.open(p).resize((220,220))
    tk_img = ImageTk.PhotoImage(pil_img)
    image2_label.config(image=tk_img)
    image2_label.image = tk_img
    status_label.config(text=f"이미지2 선택됨: {os.path.basename(p)}")

def evaluate_two_images():
    global img1_path, img2_path
    if not img1_path or not img2_path:
        result_label.config(text="두 이미지 모두 선택해야 합니다.")
        return
    try:
        repeat = int(repeat_entry.get())
        if repeat <= 0:
            raise ValueError
    except:
        result_label.config(text="반복 횟수를 양의 정수로 입력하세요.")
        return

    eval_btn.config(state='disabled')
    btn1.config(state='disabled')
    btn2.config(state='disabled')
    status_label.config(text="평가 중... (잠시 기다려 주세요)")

    # 특징은 한 번만 추출해서 반복 시에는 노이즈만 추가
    try:
        features1 = extract_person_features(img1_path)
    except Exception as e:
        result_label.config(text=f"이미지1 처리 오류: {e}")
        eval_btn.config(state='normal'); btn1.config(state='normal'); btn2.config(state='normal')
        status_label.config(text="오류 발생")
        return
    try:
        features2 = extract_person_features(img2_path)
    except Exception as e:
        result_label.config(text=f"이미지2 처리 오류: {e}")
        eval_btn.config(state='normal'); btn1.config(state='normal'); btn2.config(state='normal')
        status_label.config(text="오류 발생")
        return

    potentials1_high, potentials1_low = [], []
    potentials2_high, potentials2_low = [], []
    ap1_high, ap1_low = [], []
    ap2_high, ap2_low = [], []

    for _ in range(repeat):
        # 노이즈 추가 (환경 변동 시뮬)
        noise1 = np.random.normal(0, 0.02, features1.shape)
        noise2 = np.random.normal(0, 0.02, features2.shape)
        f1 = np.clip(features1 + noise1, 0, 1)
        f2 = np.clip(features2 + noise2, 0, 1)

        # 그룹별 score
        h1 = preference_highdev(f1); l1 = preference_lowdev(f1)
        h2 = preference_highdev(f2); l2 = preference_lowdev(f2)

        # gain/offset
        h1_adj = apply_group_gain_offset(h1, '발달'); l1_adj = apply_group_gain_offset(l1, '비발달')
        h2_adj = apply_group_gain_offset(h2, '발달'); l2_adj = apply_group_gain_offset(l2, '비발달')

        # sigmoid -> membrane potential
        mem1_h = score_to_membrane_potential_sigmoid(h1_adj)
        mem1_l = score_to_membrane_potential_sigmoid(l1_adj)
        mem2_h = score_to_membrane_potential_sigmoid(h2_adj)
        mem2_l = score_to_membrane_potential_sigmoid(l2_adj)

        potentials1_high.append(mem1_h); potentials1_low.append(mem1_l)
        potentials2_high.append(mem2_h); potentials2_low.append(mem2_l)

        ap1_high.append(action_potential_yn(mem1_h)); ap1_low.append(action_potential_yn(mem1_l))
        ap2_high.append(action_potential_yn(mem2_h)); ap2_low.append(action_potential_yn(mem2_l))

    avg1_high = np.mean(potentials1_high); avg1_low = np.mean(potentials1_low)
    avg2_high = np.mean(potentials2_high); avg2_low = np.mean(potentials2_low)

    ap1_high_avg = 'O' if avg1_high >= THRESHOLD_POTENTIAL else 'X'
    ap1_low_avg  = 'O' if avg1_low  >= THRESHOLD_POTENTIAL else 'X'
    ap2_high_avg = 'O' if avg2_high >= THRESHOLD_POTENTIAL else 'X'
    ap2_low_avg  = 'O' if avg2_low  >= THRESHOLD_POTENTIAL else 'X'

    result_text = (
        f"파일1: {os.path.basename(img1_path)}\n"
        f"  발달그룹 평균 막 전위: {avg1_high:.1f} mV  활동전위: {ap1_high_avg}\n"
        f"  비발달그룹 평균 막 전위: {avg1_low:.1f} mV  활동전위: {ap1_low_avg}\n\n"
        f"파일2: {os.path.basename(img2_path)}\n"
        f"  발달그룹 평균 막 전위: {avg2_high:.1f} mV  활동전위: {ap2_high_avg}\n"
        f"  비발달그룹 평균 막 전위: {avg2_low:.1f} mV  활동전위: {ap2_low_avg}\n\n"
        f"(그래프에서 반복별 변동 확인)"
    )
    result_label.config(text=result_text)
    status_label.config(text="평가 완료")
    eval_btn.config(state='normal')
    btn1.config(state='normal')
    btn2.config(state='normal')

    # 그래프는 blocking 형태로 띄움
    graph_results(potentials1_high, potentials1_low, potentials2_high, potentials2_low,
                  ap1_high, ap1_low, ap2_high, ap2_low, repeat)

# 버튼 연결
btn1.config(command=select_image1)
btn2.config(command=select_image2)
eval_btn.config(command=evaluate_two_images)

root.mainloop()