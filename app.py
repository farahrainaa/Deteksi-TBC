import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pickle

# Fungsi segmentasi lesi TBC
def segmentasi_lesi_tbc(img):
    h, w = img.shape
    roi_mask = np.zeros_like(img, dtype=np.uint8)
    cv2.ellipse(roi_mask, (int(w*0.35), int(h*0.5)), (int(w*0.18), int(h*0.38)), 0, 0, 360, 255, -1)
    cv2.ellipse(roi_mask, (int(w*0.65), int(h*0.5)), (int(w*0.18), int(h*0.38)), 0, 0, 360, 255, -1)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, otsu_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8((sobel / np.max(sobel)) * 255)
    combined = cv2.bitwise_and(otsu_mask, sobel)
    kernel = np.ones((2, 2), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    masked = cv2.bitwise_and(combined, roi_mask)
    _, final_mask = cv2.threshold(masked, 20, 255, cv2.THRESH_BINARY)
    return final_mask

# Fungsi ekstraksi fitur HOG sederhana
def compute_gradients(image):
    gx = np.zeros_like(image, dtype='float32')
    gy = np.zeros_like(image, dtype='float32')
    gx[:, :-1] = np.diff(image, axis=1)
    gy[:-1, :] = np.diff(image, axis=0)
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = (np.arctan2(gy, gx) * 180 / np.pi) % 180
    return magnitude, orientation

def extract_hog(image, cell_size=8, bins=9):
    magnitude, orientation = compute_gradients(image)
    h, w = image.shape
    hog_vector = []
    for i in range(0, h - cell_size + 1, cell_size):
        for j in range(0, w - cell_size + 1, cell_size):
            mag_cell = magnitude[i:i+cell_size, j:j+cell_size]
            ori_cell = orientation[i:i+cell_size, j:j+cell_size]
            hist = np.zeros(bins)
            for m in range(mag_cell.shape[0]):
                for n in range(mag_cell.shape[1]):
                    bin_idx = int(ori_cell[m, n] * bins / 180) % bins
                    hist[bin_idx] += mag_cell[m, n]
            hog_vector.extend(hist)
    return np.array(hog_vector)

# Fungsi highlight area lesi di citra grayscale
def highlight_lesi(img, mask):
    rgb = np.stack([img]*3, axis=-1)
    overlay = rgb.copy()
    overlay[mask > 0] = [0, 0, 255]
    return cv2.addWeighted(rgb, 0.5, overlay, 0.5, 0)

# Fungsi Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# KNN predict
def knn_predict(test_feat, train_feats, train_labels, k=3):
    distances = [euclidean_distance(test_feat, feat) for feat in train_feats]
    sorted_indices = np.argsort(distances)
    top_k = [train_labels[i] for i in sorted_indices[:k]]
    return max(set(top_k), key=top_k.count)

# Load data fitur dan label dari file pkl
@st.cache_data(show_spinner=True)
def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        features, labels = pickle.load(f)
    return features, labels

# Streamlit UI
st.title("Deteksi TBC dari Citra X-Ray")

uploaded_file = st.file_uploader("Upload gambar X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Load fitur & label pretrain offline dari pkl
train_features, train_labels = load_data('train_data.pkl')

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert('L').resize((128,128))
    img = np.array(img_pil) / 255.0
    img_uint8 = (img * 255).astype(np.uint8)

    st.image(img_pil, caption="Gambar Input", use_container_width=True)

    mask_lesi = segmentasi_lesi_tbc(img_uint8)
    hog_feat = extract_hog(mask_lesi / 255.0)

    prediksi = knn_predict(hog_feat, train_features, train_labels, k=3)
    st.subheader(f"Prediksi: **{prediksi}**")

    if prediksi == 'TBC':
        img_highlight = highlight_lesi(img_uint8, mask_lesi)
        st.image(img_highlight, caption="Highlight Area Lesi", use_container_width=True)
