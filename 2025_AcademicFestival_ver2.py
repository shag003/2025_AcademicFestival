import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO  # pip install ultralytics

yolo_model = YOLO("yolov8n.pt")

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
    # 노이즈 추가
    noise = np.random.normal(0, 0.02, features.shape)
    features = features + noise
    features = np.clip(features, 0, 1)
    return features

def preference_highdev(features):
    # 밝기에도 충분히 반응하도록 가중치 부여
    weights = np.array([2.0, 2.0, 2.0, 2.2])  # [brightness, color_std, saturation, complexity]
    score = np.dot(features, weights) / (np.sum(weights) + 1)
    score = np.clip(score, 0, 1)
    return score * 4 + 1

def preference_lowdev(features):
    weights = np.array([2.5, 0.3, 0.3, 0.3])  # 밝기 중심, 나머지는 약하게
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

def open_and_score():
    file_path = filedialog.askopenfilename(title="이미지 파일 선택", filetypes=[("이미지 파일", "*.jpg *.png *.jpeg")])
    if not file_path:
        result_label.config(text="이미지를 선택하지 않았습니다.")
        image_label.config(image='')
        return

    pil_img = Image.open(file_path)
    pil_img = pil_img.resize((180, 180))
    tk_img = ImageTk.PhotoImage(pil_img)
    image_label.config(image=tk_img)
    image_label.image = tk_img

    features = extract_person_features(file_path)
    high_score = preference_highdev(features)
    low_score = preference_lowdev(features)
    high_moti = motivation(high_score)
    low_moti = motivation(low_score)

    result = (
        f"파일: {file_path.split('/')[-1]}\n"
        f"[비발달 그룹]\n"
        f"  선호도: {low_score:.2f}점  {score_to_stars(low_score)}\n"
        f"  동기값: {low_moti:.2f}점  {score_to_stars(low_moti)}\n"
        f"[발달 그룹]\n"
        f"  선호도: {high_score:.2f}점  {score_to_stars(high_score)}\n"
        f"  동기값: {high_moti:.2f}점  {score_to_stars(high_moti)}"
    )
    result_label.config(text=result)

root = tk.Tk()
root.title("사람 중심 선호도/동기 평가 (밝기+화려함 민감)")
root.geometry("480x370")

btn = tk.Button(root, text="이미지 선택 및 평가", command=open_and_score)
btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("맑은 고딕", 13), justify="left")
result_label.pack(pady=5)

root.mainloop()
