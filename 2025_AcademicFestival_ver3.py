import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import matplotlib.pyplot as plt

# ✅ macOS 한글 폰트 설정
#matplotlib.rc('font', family='AppleGothic') # Mac용
matplotlib.rc('font', family='Malgun Gothic') # Windows용
matplotlib.rcParams['axes.unicode_minus'] = False

yolo_model = YOLO("yolov8n.pt")

# ---------------- Feature Extraction ----------------
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

# ---------------- Score Functions ----------------
def preference_highdev(features):
    weights = np.array([3.0, 3.0, 3.0, 3.2])
    score = np.dot(features, weights) / (np.sum(weights) + 1)
    score = np.clip(score, 0, 1)
    return score * 4 + 1

def preference_lowdev(features):
    weights = np.array([2.5, 0.3, 0.3, 0.3])
    score = np.dot(features, weights) / (np.sum(weights) + 1)
    score = np.clip(score, 0, 1)
    return score * 4 + 1

def motivation(score):
    x = (score - 3)/2
    moti = 1/(1 + np.exp(-x)) * 4 + 1
    return moti

def score_to_stars(score):
    score_int = int(round(score))
    filled = score_int
    empty = 5 - filled
    return "★" * filled + "☆" * empty

# ---------------- Evaluation ----------------
def evaluate_image(file_path, n_repeat=10):
    high_scores, low_scores, high_motis, low_motis = [], [], [], []
    base_features = extract_person_features(file_path)

    for _ in range(n_repeat):
        noise = np.random.normal(0, 0.02, base_features.shape)
        features = np.clip(base_features + noise, 0, 1)

        high_score = preference_highdev(features)
        low_score = preference_lowdev(features)
        high_moti = motivation(high_score)
        low_moti = motivation(low_score)

        high_scores.append(high_score)
        low_scores.append(low_score)
        high_motis.append(high_moti)
        low_motis.append(low_moti)

    return low_scores, low_motis, high_scores, high_motis

# ---------------- Main Function ----------------
def open_and_score():
    file_path = filedialog.askopenfilename(title="이미지 파일 선택", filetypes=[("이미지 파일", "*.jpg *.png *.jpeg")])
    if not file_path:
        label_high.config(text="발달 그룹\n(이미지 없음)")
        label_low.config(text="비발달 그룹\n(이미지 없음)")
        return

    # 🔹 선택한 이미지 표시
    pil_img = Image.open(file_path)
    pil_img = pil_img.resize((250, 250))
    tk_img = ImageTk.PhotoImage(pil_img)
    image_label.config(image=tk_img)
    image_label.image = tk_img

    # 🔹 반복 횟수 가져오기
    try:
        n_repeat = int(entry_repeat.get())
        if n_repeat <= 0:
            raise ValueError
    except ValueError:
        n_repeat = 10

    low_scores, low_motis, high_scores, high_motis = evaluate_image(file_path, n_repeat=n_repeat)

    # 평균값
    low_score, low_moti = np.mean(low_scores), np.mean(low_motis)
    high_score, high_moti = np.mean(high_scores), np.mean(high_motis)

    # 🔹 결과 업데이트
    label_low.config(
        text=f"[비발달 그룹] ({n_repeat}회 평균)\n"
             f"선호도: {low_score:.2f} {score_to_stars(low_score)}\n"
             f"동기값: {low_moti:.2f} {score_to_stars(low_moti)}"
    )

    label_high.config(
        text=f"[발달 그룹] ({n_repeat}회 평균)\n"
             f"선호도: {high_score:.2f} {score_to_stars(high_score)}\n"
             f"동기값: {high_moti:.2f} {score_to_stars(high_moti)}"
    )

    # 🔹 그래프 표시
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(range(1, n_repeat+1), low_scores, label="비발달 선호도", color="blue", marker="o")
    ax.plot(range(1, n_repeat+1), high_scores, label="발달 선호도", color="red", marker="o")
    ax.plot(range(1, n_repeat+1), low_motis, label="비발달 동기값", color="green", marker="x")
    ax.plot(range(1, n_repeat+1), high_motis, label="발달 동기값", color="orange", marker="x")

    # 평균선
    ax.axhline(low_score, color="blue", linestyle="--", alpha=0.6)
    ax.axhline(high_score, color="red", linestyle="--", alpha=0.6)
    ax.axhline(low_moti, color="green", linestyle="--", alpha=0.6)
    ax.axhline(high_moti, color="orange", linestyle="--", alpha=0.6)

    ax.set_title("반복 횟수에 따른 점수 변화")
    ax.set_xlabel("반복 횟수")
    ax.set_ylabel("점수")
    ax.set_ylim(1, 5)
    ax.legend()
    ax.grid(True)

    for widget in frame_graph.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=frame_graph)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# ---------------- GUI ----------------
root = tk.Tk()
root.title("사람 중심 선호도/동기 평가")
root.geometry("700x900")

# 🔹 반복 횟수 입력 + 버튼
frame_top = tk.Frame(root)
frame_top.pack(pady=5)

tk.Label(frame_top, text="반복 횟수:").pack(side="left", padx=5)
entry_repeat = tk.Entry(frame_top, width=5)
entry_repeat.insert(0, "10")
entry_repeat.pack(side="left")

btn = tk.Button(frame_top, text="이미지 선택 및 평가", command=open_and_score)
btn.pack(side="left", padx=10)

# 🔹 위쪽: 그림
frame_image = tk.Frame(root, bd=2, relief="solid")
frame_image.pack(pady=5, fill="both", expand=False, ipady=10)
image_label = tk.Label(frame_image, text="그림", font=("맑은 고딕", 14))
image_label.pack(pady=10)

# 🔹 중간: 결과 (좌=발달, 우=비발달)
frame_result = tk.Frame(root, bd=2, relief="solid")
frame_result.pack(pady=5, fill="x")

frame_left = tk.Frame(frame_result)
frame_left.pack(side="left", expand=True, fill="both", padx=20, pady=10)

frame_right = tk.Frame(frame_result)
frame_right.pack(side="right", expand=True, fill="both", padx=20, pady=10)

label_high = tk.Label(frame_left, text="발달 그룹", font=("맑은 고딕", 12), justify="left")
label_high.pack()

label_low = tk.Label(frame_right, text="비발달 그룹", font=("맑은 고딕", 12), justify="left")
label_low.pack()

# 🔹 아래: 그래프
frame_graph = tk.Frame(root, bd=2, relief="solid")
frame_graph.pack(pady=5, fill="both", expand=True)

root.mainloop()
