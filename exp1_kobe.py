import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Setup Output Directory
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# 2. Helper Functions
def save_image(name, image):
    """Saves an image to the output directory, converting RGB to BGR for OpenCV."""
    if len(image.shape) == 3:  # If color image (RGB), convert to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{output_dir}/{name}.png", image)

def overlay_image_alpha(bg, overlay, x, y):
    """Overlays 'overlay' onto 'bg' at (x, y) handling transparency."""
    h, w = overlay.shape[:2]
    
    # Check if overlay fits within background boundaries
    if y < 0 or y + h > bg.shape[0] or x < 0 or x + w > bg.shape[1]:
        return bg

    roi = bg[y:y+h, x:x+w]

    # CASE 1: PNG with Alpha Channel
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(0, 3):
            roi[:, :, c] = (1.0 - alpha) * roi[:, :, c] + alpha * overlay[:, :, c]
            
    # CASE 2: JPG/PNG with solid background (Fallback)
    else:
        gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        bg_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        overlay_fg = cv2.bitwise_and(overlay, overlay, mask=mask)
        roi[:, :, :] = cv2.add(bg_bg, overlay_fg)

    bg[y:y+h, x:x+w] = roi
    return bg

# =========================================================
# PART 1: BASIC IMAGE PROCESSING
# =========================================================

# Load original image (Relative path fixed)
img = cv2.imread('kobe_nba.png') 

if img is None:
    print("Error: Could not load 'kobe_nba.png'. Please check the file name.")
    sys.exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Q1: Save image
cv2.imwrite('kobe_output.png', img)
save_image("Q1_Original_Image", img_rgb)

# Q2: Get image size
height, width, channels = img_rgb.shape

# Q3: RGB planes
red = img_rgb[:, :, 0]
green = img_rgb[:, :, 1]
blue = img_rgb[:, :, 2]
save_image("Q3_Red_Plane", red)
save_image("Q3_Green_Plane", green)
save_image("Q3_Blue_Plane", blue)

# Q4: Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
save_image("Q4_Grayscale_Image", gray)

# Q5: Binary
_, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
save_image("Q5_Binary_Image", binary)

# Q6: Resize
half = cv2.resize(img_rgb, (width//2, height//2))
quarter = cv2.resize(img_rgb, (width//4, height//4))
save_image("Q6_Half_Size_Image", half)
save_image("Q6_Quarter_Size_Image", quarter)

# Q7: Rotate
rot45 = cv2.warpAffine(img_rgb, cv2.getRotationMatrix2D((width//2, height//2), 45, 1), (width, height))
rot90 = cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
rot180 = cv2.rotate(img_rgb, cv2.ROTATE_180)
save_image("Q7_Rotate_45", rot45)
save_image("Q7_Rotate_90", rot90)
save_image("Q7_Rotate_180", rot180)

# Q8: Flip
flip = cv2.flip(img_rgb, 1)
save_image("Q8_Flipped_Image", flip)

# Q9: Blur
blur = cv2.GaussianBlur(img_rgb, (15, 15), 0)
save_image("Q9_Blurred_Image", blur)


# =========================================================
# PART 2: FACE FILTER (Q10) - CORRECTED
# =========================================================

# 1. Load base image (Using relative paths to avoid file errors)
face_img = cv2.imread('kobe_nba.png')
hair_img = cv2.imread('hair.png', cv2.IMREAD_UNCHANGED)
beard_img = cv2.imread('beard.png', cv2.IMREAD_UNCHANGED)

# Error Checking
if face_img is None:
    print("Error: Could not load 'kobe_nba.png'")
    sys.exit()
if hair_img is None or beard_img is None:
    print("Error: Could not load 'hair.png' or 'beard.png'")
    sys.exit()

# 2. Detect Face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

# FIXED: Changed scaleFactor from 1.3 to 1.1 (more sensitive)
# FIXED: Changed minNeighbors from 5 to 3 (less strict)
faces = face_cascade.detectMultiScale(gray_face, 1.1, 3)

# DEBUG PRINT: Check the terminal to see if this says "Found 1 faces"
print(f"DEBUG: Found {len(faces)} faces in the image.")

if len(faces) == 0:
    print("WARNING: No faces detected! The filter will not be applied.")
    print("Try using a close-up photo of a face (like 'kobe.jpg') instead of a full-body shot.")

for (fx, fy, fw, fh) in faces:
    # 1. Hair Logic
    hair_width = int(fw * 1.2)
    hair_height = int(hair_width * (hair_img.shape[0] / hair_img.shape[1]))
    hair_resized = cv2.resize(hair_img, (hair_width, hair_height))
    
    x_hair = (fx + fw // 2) - (hair_width // 2)
    y_hair = fy - int(hair_height * 0.5) # Adjusted to 0.5 to sit slightly lower on head
    
    face_img = overlay_image_alpha(face_img, hair_resized, x_hair, y_hair)

    # 2. Beard Logic
    beard_width = int(fw * 1.0)
    beard_height = int(beard_width * (beard_img.shape[0] / beard_img.shape[1]))
    beard_resized = cv2.resize(beard_img, (beard_width, beard_height))
    
    x_beard = (fx + fw // 2) - (beard_width // 2)
    y_beard = (fy + fh) - int(beard_height * 0.8) # Adjusted to fit chin better
    
    face_img = overlay_image_alpha(face_img, beard_resized, x_beard, y_beard)

# Convert final result to RGB
filtered_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

# *** Q10: SAVE THE FILTERED IMAGE ***
save_image("Q10_Hair_Beard_Filter", filtered_rgb)
print("Success: Filtered image saved.")


# =========================================================
# DISPLAY RESULTS
# =========================================================
plt.figure(figsize=(15, 12))
images = [
    (img_rgb, "Original"),
    (red, "Red"),
    (green, "Green"),
    (blue, "Blue"),
    (gray, "Grayscale"),
    (binary, "Binary"),
    (half, "Half Size"),
    (quarter, "Quarter Size"),
    (rot45, "Rotate 45"),
    (rot90, "Rotate 90"),
    (rot180, "Rotate 180"),
    (flip, "Flipped"),
    (blur, "Blurred"),
    (filtered_rgb, "Hair + Beard Filter")
]

for i, (image, title) in enumerate(images):
    plt.subplot(4, 4, i + 1)
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()