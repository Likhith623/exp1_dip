import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Setup Paths ---
base_img_path = '/Users/likhith./KobeBryant_fun/kobe_nba.png'
hair_png_path = '/Users/likhith./KobeBryant_fun/hair.png'
beard_png_path = '/Users/likhith./KobeBryant_fun/beard.png'

# --- Visualization Helper ---
def show_image(img, title, cmap=None, ax=None):
    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
    
    if len(img.shape) == 2:
        ax.imshow(img, cmap=cmap or 'gray')
    else:
        # Convert BGR (OpenCV) to RGB (Matplotlib)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    ax.set_title(title)
    ax.axis('off')

# ==========================================
# Task 1: Read, Verify & Metadata Annotation
# ==========================================
print("--- Task 1: IO & Annotation ---")
img_bgr = cv2.imread(base_img_path)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found: {base_img_path}")

h, w, c = img_bgr.shape
img_annotated = img_bgr.copy()
label = f"Res: {w}x{h} | Mode: RGB"
# Add a background strip for text readability
cv2.rectangle(img_annotated, (10, h-40), (250, h-10), (0,0,0), -1) 
cv2.putText(img_annotated, label, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

show_image(img_annotated, "Task 1: Original with Metadata")
plt.show()

# ==========================================
# Task 2: Channel Splitting (Color Analysis)
# ==========================================
print("--- Task 2: Channel Splitting ---")
B, G, R = cv2.split(img_bgr)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
show_image(R, "Red Channel", cmap='Reds', ax=axs[0])
show_image(G, "Green Channel", cmap='Greens', ax=axs[1])
show_image(B, "Blue Channel", cmap='Blues', ax=axs[2])
plt.show()

# ==========================================
# Task 3: Adaptive Thresholding
# ==========================================
# 
print("--- Task 3: Thresholding ---")
gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Using Adaptive Gaussian: Handles uneven lighting better than fixed threshold
binary_adaptive = cv2.adaptiveThreshold(
    gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

show_image(binary_adaptive, "Task 3: Adaptive Binary Map")
plt.show()

# ==========================================
# Task 4 & 5: Geometric Transforms
# ==========================================
print("--- Task 4 & 5: Geometry ---")
# Resize with Aspect Ratio preservation logic
scale_percent = 50 
width = int(img_bgr.shape[1] * scale_percent / 100)
height = int(img_bgr.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(img_bgr, dim, interpolation=cv2.INTER_AREA)

# Rotation with Border Replication (removes black corners)
M = cv2.getRotationMatrix2D((w//2, h//2), -15, 1) # -15 degrees
rotated = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
show_image(resized, f"Task 4: 50% Downsample ({width}x{height})", ax=axs[0])
show_image(rotated, "Task 5: -15 deg Rotation (Reflect Border)", ax=axs[1])
plt.show()

# ==========================================
# Task 6: Edge Detection (Canny)
# ==========================================
# 
print("--- Task 6: Canny Edge Detection ---")
# Detects structural edges using gradient intensity
edges = cv2.Canny(img_bgr, 100, 200) 
show_image(edges, "Task 6: Canny Edge Map", cmap='gray')
plt.show()

# ==========================================
# Task 7: Noise Reduction (Gaussian Blur)
# ==========================================
print("--- Task 7: Smoothing/Blurring ---")
# 15x15 kernel smooths out skin textures/noise
blurred = cv2.GaussianBlur(img_bgr, (15, 15), 0)
show_image(blurred, "Task 7: Gaussian Blur (15x15)")
plt.show()

# ==========================================
# ADVANCED TASK: Accurate AR Face Filter
# ==========================================
print("--- Advanced Task: Precision Face Overlay ---")

# 1. Load Assets
hair_png = cv2.imread(hair_png_path, cv2.IMREAD_UNCHANGED)
beard_png = cv2.imread(beard_png_path, cv2.IMREAD_UNCHANGED)

if hair_png is None or beard_png is None:
    print("❌ Error: Assets not found. Check paths.")
    exit()

# 2. Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_img, 1.1, 5, minSize=(50, 50))

if len(faces) == 0:
    print("❌ No face detected.")
    exit()


import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# ---------------- LOAD IMAGE ----------------
img_path = "/Users/likhith./KobeBryant_fun/kobe_nba.png"
frame = cv2.imread(img_path)
if frame is None:
    raise FileNotFoundError("Image not found")

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ---------------- LOAD FILTERS ----------------
sunglasses_img = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)
mustache_img   = cv2.imread("moustache.png",  cv2.IMREAD_UNCHANGED)

# ---------------- MEDIAPIPE ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

results = face_mesh.process(rgb)
if not results.multi_face_landmarks:
    raise RuntimeError("❌ No face detected")

face_landmarks = results.multi_face_landmarks[0]
h, w = frame.shape[:2]

def get_point(index):
    lm = face_landmarks.landmark[index]
    return int(lm.x * w), int(lm.y * h)

# ---------------- ALPHA OVERLAY ----------------
def overlay(bg, fg, x, y):
    h_fg, w_fg = fg.shape[:2]
    if x < 0 or y < 0 or x + w_fg > bg.shape[1] or y + h_fg > bg.shape[0]:
        return bg

    alpha = fg[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+h_fg, x:x+w_fg, c] = (
            alpha * fg[:, :, c] +
            (1 - alpha) * bg[y:y+h_fg, x:x+w_fg, c]
        )
    return bg

# =================================================
# APPLY SUNGLASSES (EXACT SAME LOGIC)
# =================================================
left_eye_outer  = get_point(33)
right_eye_outer = get_point(263)
left_eye_bottom = get_point(145)
right_eye_bottom = get_point(374)

eye_center_x = (left_eye_outer[0] + right_eye_outer[0]) // 2
eye_center_y = (left_eye_bottom[1] + right_eye_bottom[1]) // 2

eye_width = int(np.linalg.norm(
    np.array(left_eye_outer) - np.array(right_eye_outer)
))

sg_width  = int(eye_width * 2.0)
sg_height = int(sg_width * sunglasses_img.shape[0] / sunglasses_img.shape[1])

sunglasses_resized = cv2.resize(sunglasses_img, (sg_width, sg_height))

sg_x = eye_center_x - sg_width // 2
sg_y = eye_center_y - sg_height // 2 - 20

frame = overlay(frame, sunglasses_resized, sg_x, sg_y)

# =================================================
# APPLY MUSTACHE (EXACT SAME LOGIC)
# =================================================
nose_tip = get_point(4)
upper_lip = get_point(13)
left_temple = get_point(71)
right_temple = get_point(301)

face_width = int(np.linalg.norm(
    np.array(left_temple) - np.array(right_temple)
))

must_width = int(face_width * 0.9)
must_height = int(must_width * mustache_img.shape[0] / mustache_img.shape[1])

mustache_resized = cv2.resize(mustache_img, (must_width, must_height))

must_x = nose_tip[0] - must_width // 2 - 30
must_y = (nose_tip[1] + upper_lip[1]) // 2 - must_height // 2 + 10

frame = overlay(frame, mustache_resized, must_x, must_y)

# ---------------- DISPLAY & SAVE ----------------
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title("Static Image: Sunglasses + Mustache (Perfect MediaPipe)")
plt.axis("off")
plt.show()

cv2.imwrite("kobe_static_filters.jpg", frame)
print("✅ Saved as kobe_static_filters.jpg")








