import os
import math
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import platform
import tkinter.font as tkfont

# optional imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# -----------------------------
# 환경/고정 파라미터
# -----------------------------
MODEL_PATH = "yolov8n.pt"         # optional YOLO 모델(로컬에 있으면 사용)
RESTING = -70                     # resting potential (mV)
THRESHOLD = -55                   # spike threshold (mV)
ACTION = 40                       # action potential peak (mV)

# 사용자가 고정한 값들
BASE_ALPHA = 0.1                  # 고정 색 민감도 (dev 전용)
CF_THRESHOLD = 0.25               # colorfulness 임계치
DEFAULT_NOISE_SIGMA = 0.008       # 노이즈 sigma
DEFAULT_REPEAT = 30               # 반복 횟수
P_EXP = 2.0
BETA = 6.0

# pref weights
PREF_HIGH_WEIGHTS = np.array([4.0, 4.0, 12.0, 4.0])
PREF_LOW_WEIGHTS  = np.array([1.0, 0.1, 0.1, 0.1])

RNG_SEED = 12345

# -----------------------------
# 폰트 설정(플랫폼별)
# -----------------------------
def setup_fonts():
    "간단한 tkinter/matplotlib 폰트 설정"
    try:
        default = tkfont.nametofont("TkDefaultFont")
        default.configure(size=10)
    except Exception:
        pass
    try:
        system = platform.system()
        if system == 'Windows':
            plt.rc('font', family='Malgun Gothic')
        elif system == 'Darwin':
            plt.rc('font', family='AppleGothic')
        else:
            plt.rc('font', family='DejaVu Sans')
    except Exception:
        pass
    plt.rcParams['axes.unicode_minus'] = False

setup_fonts()

# -----------------------------
# 모델 초기화
# -----------------------------
yolo = None
if YOLO_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        yolo = YOLO(MODEL_PATH)
    except Exception:
        yolo = None

# Haar cascade fallback for face detection
face_cascade = None
try:
    face_xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if os.path.exists(face_xml):
        face_cascade = cv2.CascadeClassifier(face_xml)
except Exception:
    face_cascade = None

# -----------------------------
# 기본 특징 추출 함수
# -----------------------------
def compute_basic_features(patch):
    "128x128 BGR 패치에서 [brightness,color_std,saturation,complexity] 반환"
    brightness = np.mean(patch) / 255.0
    color_std = np.std(patch) / 255.0
    saturation = np.mean(np.max(patch, axis=2) - np.min(patch, axis=2)) / 255.0
    complexity = cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() / 1000.0
    return np.array([brightness, color_std, saturation, complexity], dtype=np.float32)

def extract_features(path):
    "이미지에서 person ROI가 있으면 ROI 기반 특징(평균)과 boxes 반환, 없으면 전체 이미지 기반 반환"
    img = cv2.imread(path)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다.")
    h, w = img.shape[:2]
    boxes = []; feats = []
    if yolo is not None:
        try:
            res = yolo(img)
            try:
                xyxy = res[0].boxes.xyxy.cpu().numpy()
                cls = res[0].boxes.cls.cpu().numpy().astype(int)
            except Exception:
                data = getattr(res[0].boxes, "data", None)
                if data is not None:
                    arr = data.cpu().numpy(); xyxy = arr[:, :4]; cls = arr[:, 5].astype(int)
                else:
                    xyxy = np.empty((0,4)); cls = np.array([], dtype=int)
            person_idx = np.where(cls == 0)[0]
            for i in person_idx:
                x1, y1, x2, y2 = xyxy[i].astype(int)
                x1, y1 = max(0,x1), max(0,y1); x2, y2 = min(w,x2), min(h,y2)
                if x2 <= x1 or y2 <= y1: continue
                boxes.append((x1, y1, x2, y2))
                crop = cv2.resize(img[y1:y2, x1:x2], (128,128))
                feats.append(compute_basic_features(crop))
        except Exception:
            boxes = []; feats = []
    if not feats:
        patch = cv2.resize(img, (128,128))
        return compute_basic_features(patch), []
    return np.mean(feats, axis=0), boxes

# -----------------------------
# 자동 모드 감지: boxes 우선, 없으면 얼굴 검출
# -----------------------------
def auto_detect_by_boxes_or_face(path, boxes):
    "boxes가 있으면 'person', 없으면 얼굴 검출 시 'person', 아니면 'general' 반환"
    if boxes:
        return 'person'
    if face_cascade is not None:
        img = cv2.imread(path)
        if img is None:
            return 'general'
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
        if len(faces) > 0:
            return 'person'
    return 'general'

# -----------------------------
# composite colorfulness (ROI 우선)
# -----------------------------
def compute_colorfulness(path, boxes=None):
    "ROI 우선으로 Hasler 기반 colorfulness(0..1 근사) 계산"
    img = cv2.imread(path)
    if img is None: return 0.0
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float")
    def cf_for_arr(arr):
        R,G,B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        rg = np.abs(R-G); yb = np.abs(0.5*(R+G)-B)
        std_rg, std_yb = np.std(rg), np.std(yb)
        mean_rg, mean_yb = np.mean(rg), np.mean(yb)
        raw = math.sqrt(std_rg**2 + std_yb**2) + 0.3 * math.sqrt(mean_rg**2 + mean_yb**2)
        alpha = 0.02
        return 1.0 / (1.0 + math.exp(-alpha*(raw - 12.0)))
    if boxes:
        vals = []
        for (x1,y1,x2,y2) in boxes:
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0: continue
            vals.append(cf_for_arr(crop))
        return float(np.mean(vals)) if vals else cf_for_arr(rgb)
    else:
        return cf_for_arr(rgb)

# -----------------------------
# 선호도/부스트/매핑 (고정 alpha 사용)
# -----------------------------
def pref_high_base(f):
    "발달 그룹 기본 점수 (가중치 적용)"
    score = float(np.dot(f, PREF_HIGH_WEIGHTS) / (np.sum(PREF_HIGH_WEIGHTS) + 1.0))
    return np.clip(score, 0, 1)

def pref_low_base(f):
    "비발달 그룹 기본 점수 (가중치 적용)"
    score = float(np.dot(f, PREF_LOW_WEIGHTS) / (np.sum(PREF_LOW_WEIGHTS) + 1.0))
    return np.clip(score, 0, 1)

def dev_color_boost_conditional(score, colorfulness, base_alpha=BASE_ALPHA, p=P_EXP, beta=BETA, threshold=CF_THRESHOLD):
    "발달 전용의 조건부 비선형 부스트(포화 방지: (1-score) 사용)"
    if colorfulness < threshold:
        return score
    cf_pow = colorfulness ** p
    boost_factor = 1.0 - math.exp(-beta * cf_pow)
    boosted = score + (1.0 - score) * (base_alpha * boost_factor)
    return float(np.clip(boosted, 0, 1))

def apply_gain(s, group):
    "그룹별 gain/offset (원본 유지: 발달에 소폭 증폭)"
    if group == 'dev':
        return float(np.clip(1.1*s + 0.05, 0, 1))
    else:
        return float(np.clip(0.8*s - 0.05, 0, 1))

def score_to_mem_mode(s, mode='dev'):
    "모드별 sigmoid mapping (dev 더 민감)"
    if mode == 'dev':
        k, c = 10.0, 0.50
    else:
        k, c = 6.0, 0.55
    s_sig = 1.0 / (1.0 + math.exp(-k*(s - c)))
    return RESTING + s_sig * (ACTION - RESTING)

def is_spike(v):
    "막전위가 역치 이상인지 판정"
    return 1 if v >= THRESHOLD else 0

# -----------------------------
# 통계 유틸
# -----------------------------
def paired_ttest_and_cohend(a, b):
    "paired samples의 p-value와 Cohen's d (paired)를 반환"
    a = np.array(a); b = np.array(b)
    diff = a - b
    n = len(diff)
    mean_diff = float(np.mean(diff)) if n>0 else 0.0
    sd_diff = float(np.std(diff, ddof=1)) if n>1 else 0.0
    cohen_d = mean_diff / sd_diff if sd_diff > 0 else (float('inf') if mean_diff != 0 else 0.0)
    p_val = None
    if SCIPY_AVAILABLE and n>1:
        try:
            _, p_val = scipy_stats.ttest_rel(a, b)
        except Exception:
            p_val = None
    if p_val is None and n>1 and sd_diff>0:
        t_stat = mean_diff / (sd_diff / math.sqrt(n))
        p_val = 2.0 * (1.0 - 0.5*(1.0 + math.erf(abs(t_stat)/math.sqrt(2.0))))
    return p_val, cohen_d, mean_diff

# -----------------------------
# 평가: 자동 모드 감지(ROI 우선) -> 조건부 부스트 -> 통계 출력
# -----------------------------
def evaluate_pair_auto(path1, path2, repeat=DEFAULT_REPEAT, sigma=DEFAULT_NOISE_SIGMA):
    "두 이미지에 대해 자동 판정 및 분석(고정 파라미터) 수행"
    rng = np.random.default_rng(RNG_SEED)
    # features + boxes
    f1, boxes1 = extract_features(path1)
    f2, boxes2 = extract_features(path2)
    # auto detection
    detected1 = auto_detect_by_boxes_or_face(path1, boxes1)
    detected2 = auto_detect_by_boxes_or_face(path2, boxes2)
    # composite colorfulness (ROI 우선)
    cf1 = compute_colorfulness(path1, boxes1 if detected1=='person' else None)
    cf2 = compute_colorfulness(path2, boxes2 if detected2=='person' else None)
    p1_dev, p1_non, p2_dev, p2_non = [], [], [], []
    for _ in range(repeat):
        n1 = np.clip(f1 + rng.normal(0, sigma, f1.shape), 0, 1)
        n2 = np.clip(f2 + rng.normal(0, sigma, f2.shape), 0, 1)
        s1_dev = pref_high_base(n1); s1_non = pref_low_base(n1)
        s2_dev = pref_high_base(n2); s2_non = pref_low_base(n2)
        s1_dev = dev_color_boost_conditional(s1_dev, cf1, base_alpha=BASE_ALPHA, p=P_EXP, beta=BETA, threshold=CF_THRESHOLD)
        s2_dev = dev_color_boost_conditional(s2_dev, cf2, base_alpha=BASE_ALPHA, p=P_EXP, beta=BETA, threshold=CF_THRESHOLD)
        s1_dev = apply_gain(s1_dev, 'dev'); s1_non = apply_gain(s1_non, 'non')
        s2_dev = apply_gain(s2_dev, 'dev'); s2_non = apply_gain(s2_non, 'non')
        v1_dev = score_to_mem_mode(s1_dev, 'dev'); v1_non = score_to_mem_mode(s1_non, 'non')
        v2_dev = score_to_mem_mode(s2_dev, 'dev'); v2_non = score_to_mem_mode(s2_non, 'non')
        p1_dev.append(v1_dev); p1_non.append(v1_non); p2_dev.append(v2_dev); p2_non.append(v2_non)
    # stats
    p_val_dev, cohend_dev, mean_diff_dev = paired_ttest_and_cohend(p1_dev, p2_dev)
    p_val_non, cohend_non, mean_diff_non = paired_ttest_and_cohend(p1_non, p2_non)
    summary = {
        'mode1': detected1, 'mode2': detected2,
        'cf1': cf1, 'cf2': cf2,
        'dev': {'arr1': p1_dev, 'arr2': p2_dev, 'p_val': p_val_dev, 'cohen_d': cohend_dev, 'mean_diff': mean_diff_dev},
        'non': {'arr1': p1_non, 'arr2': p2_non, 'p_val': p_val_non, 'cohen_d': cohend_non, 'mean_diff': mean_diff_non},
    }
    return summary

# -----------------------------
# GUI 구성 (Grid UI 제거된 간단형)
# -----------------------------
root = tk.Tk()
root.title(" ")
root.geometry("920x700")

img1_path = None
img2_path = None

top = tk.Frame(root); top.pack(pady=8)
btn1 = tk.Button(top, text="이미지1 선택"); btn1.grid(row=0, column=0, padx=6)
btn2 = tk.Button(top, text="이미지2 선택"); btn2.grid(row=0, column=1, padx=6)
tk.Label(top, text="반복:").grid(row=0, column=2)
repeat_entry = tk.Entry(top, width=6); repeat_entry.insert(0, str(DEFAULT_REPEAT)); repeat_entry.grid(row=0, column=3)

# 표시 토글
mode = {'value': 'dev'}
def toggle_mode():
    "모드 토글(표시용)"
    if mode['value'] == 'dev':
        mode['value'] = 'non'; mode_btn.config(text="모드: 비발달", bg="#fff2d9")
    else:
        mode['value'] = 'dev'; mode_btn.config(text="모드: 발달", bg="#d9f2ff")
mode_btn = tk.Button(top, text="모드: 발달", command=toggle_mode, bg="#d9f2ff"); mode_btn.grid(row=0, column=4, padx=8)

run_btn = tk.Button(top, text="평가 실행"); run_btn.grid(row=0, column=5, padx=8)

status = tk.Label(root, text=f"BASE_ALPHA={BASE_ALPHA}, CF_THRESHOLD={CF_THRESHOLD}, sigma={DEFAULT_NOISE_SIGMA}")
status.pack(pady=6)

images_frame = tk.Frame(root); images_frame.pack(pady=6)
lbl1 = tk.Label(images_frame); lbl1.grid(row=0, column=0, padx=12)
lbl2 = tk.Label(images_frame); lbl2.grid(row=0, column=1, padx=12)

result_label = tk.Label(root, text="", justify="left", font=("TkDefaultFont", 11))
result_label.pack(pady=10)

# -----------------------------
# GUI 콜백
# -----------------------------
def choose_image1():
    "이미지1 선택하고 썸네일 표시"
    global img1_path
    p = filedialog.askopenfilename(filetypes=[("이미지","*.jpg *.png *.jpeg *.png")])
    if not p: return
    img1_path = p
    tkimg = ImageTk.PhotoImage(Image.open(p).resize((360,360)))
    lbl1.config(image=tkimg); lbl1.image = tkimg
    status.config(text=f"이미지1 선택: {os.path.basename(p)}")

def choose_image2():
    "이미지2 선택하고 썸네일 표시"
    global img2_path
    p = filedialog.askopenfilename(filetypes=[("이미지","*.jpg *.png *.jpeg *.png")])
    if not p: return
    img2_path = p
    tkimg = ImageTk.PhotoImage(Image.open(p).resize((360,360)))
    lbl2.config(image=tkimg); lbl2.image = tkimg
    status.config(text=f"이미지2 선택: {os.path.basename(p)}")

def run_evaluate():
    "GUI에서 호출되는 단순 평가 실행 및 결과 표시"
    global img1_path, img2_path
    if not img1_path or not img2_path:
        result_label.config(text="두 이미지를 모두 선택하세요."); return
    try:
        repeat = int(repeat_entry.get()); assert repeat>0
    except:
        result_label.config(text="반복은 양의 정수여야 합니다."); return

    status.config(text="처리 중...")
    root.update_idletasks()
    start = time.time()
    summary = evaluate_pair_auto(img1_path, img2_path, repeat=repeat, sigma=DEFAULT_NOISE_SIGMA)
    elapsed = time.time() - start
    cur = mode['value']
    d = summary['dev']; n = summary['non']
    if cur == 'dev':
        arr1, arr2 = d['arr1'], d['arr2']
    else:
        arr1, arr2 = n['arr1'], n['arr2']
    p_val = (d['p_val'] if cur=='dev' else n['p_val'])
    coh_text = (d['cohen_d'] if cur=='dev' else n['cohen_d'])
    p_text = f"{p_val:.4f}" if (p_val is not None and not math.isnan(p_val)) else "N/A"
    lines = []
    lines.append(f"자동 판정: img1_mode={summary['mode1']}, img2_mode={summary['mode2']}")
    lines.append(f"colorfulness: img1={summary['cf1']:.3f}, img2={summary['cf2']:.3f}")
    lines.append(f"사용 파라미터: BASE_ALPHA={BASE_ALPHA:.3f}, CF_THRESHOLD={CF_THRESHOLD:.3f}, sigma={DEFAULT_NOISE_SIGMA}")
    lines.append("")
    lines.append(f"=== 표시 모드: {'발달' if cur=='dev' else '비발달'} ===")
    lines.append(f"이미지1 평균 전위: {np.mean(arr1):.2f} mV    이미지2 평균 전위: {np.mean(arr2):.2f} mV")
    lines.append(f"Mean diff: {(d['mean_diff'] if cur=='dev' else n['mean_diff']):.3f}   p-value: {p_text}   Cohen's d: {coh_text:.3f}")
    result_label.config(text="\n".join(lines))
    status.config(text=f"완료 (소요 {elapsed:.1f}s)")
    # plot
    plot_compare(arr1, arr2, [is_spike(v) for v in arr1], [is_spike(v) for v in arr2], repeat, "발달" if cur=='dev' else "비발달")

btn1.config(command=choose_image1); btn2.config(command=choose_image2)
run_btn.config(command=run_evaluate)

# -----------------------------
# plotting helper
# -----------------------------
def plot_compare(arr1, arr2, ap1, ap2, repeat, mode_name):
    "반복별 막전위를 시각화"
    x = np.arange(1, repeat+1)
    plt.figure(figsize=(10,5))
    plt.plot(x, arr1, 'o-', label=f'이미지1 - {mode_name}')
    plt.plot(x, arr2, 's-', label=f'이미지2 - {mode_name}')
    plt.axhline(THRESHOLD, color='r', linestyle='--', label='역치')
    plt.fill_between(x, RESTING, ACTION, where=np.array(ap1)==1, color='blue', alpha=0.08)
    plt.fill_between(x, RESTING, ACTION, where=np.array(ap2)==1, color='green', alpha=0.06)
    plt.ylim(RESTING-5, ACTION+5); plt.xlabel('반복'); plt.ylabel('막 전위 (mV)')
    plt.title(f'두 이미지 반복 비교 ({mode_name} 모드)'); plt.legend(); plt.grid(True); plt.show()

# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":
    root.mainloop()