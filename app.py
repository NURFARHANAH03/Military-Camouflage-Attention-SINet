import os
import base64
import io
import tempfile
import cv2
import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import time
import threading
import av
import textwrap

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from models.sinet_gra import SINet_GRA

def resize_image(path, size):
    if path:
        img = Image.open(path)
        img = img.resize(size)
        return img
    return None

# ==========================
# STANDARD IMAGE SIZES
# ==========================

TOP_LOGO_SIZE = (280, 180)      # picture4
SMALL_LOGO_SIZE = (180, 100)    # picture1, picture2
FOOTER_ICON_SIZE = (60, 60)     # picture3
HERO_IMAGE_SIZE = (650, 450)    # picture5
UPLOAD_ICON_SIZE = (130, 130)   # picture6
TIPS_ICON_SIZE = (60, 60)       # picture7
HOME_ICON_SIZE = (40, 40)       # picture8

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="CAMOU Vision",
    page_icon="🎯",
    layout="wide"
)

# =====================================================
# PATHS
# =====================================================
BASE_DIR = r"C:\Users\User\Documents\fyp_military"
IMAGE_DIR = os.path.join(BASE_DIR, "external_image")

# Model checkpoint path
# Put sinet_gra_best.pth inside: C:\Users\User\Documents\fyp_military\checkpoints
MODEL_PATH = os.path.join(
    BASE_DIR,
    "checkpoints_combined",
    "sinet_gra_best_seed42.pth"
)
IMG_SIZE = 320
MASK_THRESHOLD = 0.4

# Optional local mask folders used only when an uploaded image keeps
# its original dataset filename. External images simply have no match.
GT_MASK_DIRS = [
    os.path.join(BASE_DIR, "combined_dataset", "masks"),
    os.path.join(BASE_DIR, "mc_dataset_cropped", "masks"),
    os.path.join(BASE_DIR, "mc_dataset", "masks"),
]

def get_image(name):
    for ext in [".png", ".jpg", ".jpeg"]:
        path = os.path.join(IMAGE_DIR, name + ext)
        if os.path.exists(path):
            return path
    return None

def image_to_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def find_matching_ground_truth(image_name: str):
    """Return a matching local mask path when one exists, otherwise None."""
    base_name = os.path.splitext(os.path.basename(image_name))[0]
    extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    for mask_dir in GT_MASK_DIRS:
        if not os.path.isdir(mask_dir):
            continue
        for ext in extensions:
            candidate = os.path.join(mask_dir, base_name + ext)
            if os.path.exists(candidate):
                return candidate
    return None

PICTURE_1 = get_image("picture1")
PICTURE_2 = get_image("picture2")
PICTURE_3 = get_image("picture3")
PICTURE_4 = get_image("picture4")
PICTURE_5 = get_image("picture5")
PICTURE_6 = get_image("picture6")
PICTURE_7 = get_image("picture7")
PICTURE_8 = get_image("picture8")


# =====================================================
# SINet + GRA MODEL INFERENCE
# =====================================================
@st.cache_resource
def load_sinet_gra_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model checkpoint not found: {MODEL_PATH}")
        st.stop()

    # pretrained=False avoids downloading ResNet weights during app runtime.
    model = SINet_GRA(pretrained=False).to(device)

    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def preprocess_image_for_model(image: Image.Image):
    """Resize + normalize exactly like training ImageNet preprocessing."""
    image = image.convert("RGB")
    original_size = image.size  # (width, height)

    resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(resized).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std

    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, original_size


def run_sinet_gra_detection(image: Image.Image):
    model, device = load_sinet_gra_model()

    # Synchronise before timing CUDA operations
    if device.type == "cuda":
        torch.cuda.synchronize()

    total_start = time.perf_counter()

    # -------------------------
    # Preprocessing
    # -------------------------
    preprocess_start = time.perf_counter()

    input_tensor, original_size = preprocess_image_for_model(image)
    input_tensor = input_tensor.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    preprocess_time = time.perf_counter() - preprocess_start

    # -------------------------
    # Model inference
    # -------------------------
    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_start = time.perf_counter()

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_time = time.perf_counter() - inference_start

    # -------------------------
    # Postprocessing
    # -------------------------
    postprocess_start = time.perf_counter()

    probs = F.interpolate(
        probs,
        size=(original_size[1], original_size[0]),
        mode="bilinear",
        align_corners=False
    )

    prob_np = probs[0, 0].detach().cpu().numpy()
    mask_np = (prob_np > MASK_THRESHOLD).astype(np.uint8) * 255

    detected_pixels = prob_np[mask_np > 0]

    if detected_pixels.size > 0:
        foreground_probability = float(detected_pixels.mean() * 100)
    else:
        foreground_probability = float(prob_np.max() * 100)

    mask_area_ratio = float((mask_np > 0).mean())

    status = (
        "DETECTED"
        if mask_area_ratio > 0.001 and foreground_probability >= 50
        else "NOT DETECTED"
    )

    postprocess_time = time.perf_counter() - postprocess_start
    total_time = time.perf_counter() - total_start

    inference_fps = 1.0 / inference_time if inference_time > 0 else 0.0
    pipeline_fps = 1.0 / total_time if total_time > 0 else 0.0

    timing = {
        "preprocessing_ms": preprocess_time * 1000,
        "inference_ms": inference_time * 1000,
        "postprocessing_ms": postprocess_time * 1000,
        "total_ms": total_time * 1000,
        "inference_fps": inference_fps,
        "pipeline_fps": pipeline_fps,
        "device": str(device),
        "model_input": f"{IMG_SIZE} × {IMG_SIZE}",
        "original_resolution": (
            f"{original_size[0]} × {original_size[1]}"
        )
    }

    return (
        mask_np,
        foreground_probability,
        status,
        mask_area_ratio,
        prob_np,
        timing
    )

def process_video_every_2_seconds(video_bytes):
    """Extract one frame every 2 seconds and run SINet + GRA."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        os.remove(temp_video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        os.remove(temp_video_path)
        return []

    duration = frame_count / fps
    results = []
    current_time = 0.0

    while current_time <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)

        (
            mask_np,
            foreground_probability,
            status,
            mask_area_ratio,
            _,
            timing,
        ) = run_sinet_gra_detection(frame_image)

        overlay = create_prediction_overlay(frame_image, mask_np)

        results.append({
            "time": current_time,
            "frame": frame_image,
            "mask": mask_np,
            "overlay": overlay,
            "foreground_probability": foreground_probability,
            "status": status,
            "mask_area_ratio": mask_area_ratio,
            "timing": timing,
        })
        current_time += 2.0

    cap.release()
    os.remove(temp_video_path)
    return results

def create_prediction_overlay(
    image: Image.Image,
    prediction_mask: np.ndarray,
    alpha: float = 0.45
):
    image_np = np.array(image.convert("RGB"))

    if prediction_mask.shape[:2] != image_np.shape[:2]:
        prediction_mask = cv2.resize(
            prediction_mask,
            (image_np.shape[1], image_np.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    overlay = image_np.copy()
    region = prediction_mask > 0

    overlay[region] = (
        (1 - alpha) * overlay[region]
        + alpha * np.array([0, 255, 0])
    ).astype(np.uint8)

    return overlay

def prepare_ground_truth_mask(
    mask_source,
    target_size
):
    gt_image = Image.open(mask_source).convert("L")
    gt_image = gt_image.resize(
        target_size,
        Image.Resampling.NEAREST
    )

    gt_np = np.array(gt_image)
    gt_binary = (gt_np > 0).astype(np.uint8)

    return gt_binary

def calculate_sample_metrics(
    prediction_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    eps: float = 1e-7
):
    pred = (prediction_mask > 0).astype(np.uint8)
    gt = (ground_truth_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    union = np.logical_or(pred, gt).sum()

    dice = (
        (2.0 * intersection + eps)
        / (pred_sum + gt_sum + eps)
    )

    iou = (
        (intersection + eps)
        / (union + eps)
    )

    return float(dice), float(iou)

def create_comparison_overlay(
    image: Image.Image,
    prediction_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    alpha: float = 0.55
):
    image_np = np.array(image.convert("RGB"))

    pred = prediction_mask > 0
    gt = ground_truth_mask > 0

    true_positive = pred & gt
    false_positive = pred & (~gt)
    false_negative = (~pred) & gt

    overlay = image_np.copy().astype(np.float32)

    colours = {
        "tp": np.array([0, 255, 0]),     # green
        "fp": np.array([255, 0, 0]),     # red
        "fn": np.array([0, 100, 255])    # blue
    }

    overlay[true_positive] = (
        (1 - alpha) * overlay[true_positive]
        + alpha * colours["tp"]
    )

    overlay[false_positive] = (
        (1 - alpha) * overlay[false_positive]
        + alpha * colours["fp"]
    )

    overlay[false_negative] = (
        (1 - alpha) * overlay[false_negative]
        + alpha * colours["fn"]
    )

    return overlay.astype(np.uint8)
    # =====================================================
    # LIVE WEBCAM DETECTION PROCESSOR
    # =====================================================
class CamouflageLiveProcessor(VideoProcessorBase):
    """Continuous webcam preview with a refreshed mask approximately once per second."""

    def __init__(self, model, device, interval_seconds=1.0):
        self.model = model
        self.device = device
        self.interval_seconds = interval_seconds
        self.last_processed_time = 0.0
        self.latest_mask = None
        self.latest_foreground_probability = 0.0
        self.latest_status = "WAITING"
        self.latest_timing = None
        self.lock = threading.Lock()

    def recv(self, frame):
        frame_rgb = frame.to_ndarray(format="rgb24")
        current_time = time.monotonic()

        if current_time - self.last_processed_time >= self.interval_seconds:
            pil_image = Image.fromarray(frame_rgb)
            (
                mask_np,
                foreground_probability,
                status,
                _,
                _,
                timing,
            ) = run_sinet_gra_detection(pil_image)

            with self.lock:
                self.latest_mask = mask_np
                self.latest_foreground_probability = foreground_probability
                self.latest_status = status
                self.latest_timing = timing
                self.last_processed_time = current_time

        output_frame = frame_rgb.copy()
        with self.lock:
            mask = self.latest_mask
            foreground_probability = self.latest_foreground_probability
            status = self.latest_status
            timing = self.latest_timing

        if mask is not None:
            if mask.shape[:2] != output_frame.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (output_frame.shape[1], output_frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            detected_region = mask > 0
            output_frame[detected_region] = (
                0.55 * output_frame[detected_region]
                + 0.45 * np.array([0, 255, 0])
            ).astype(np.uint8)

        if status == "DETECTED":
            label_color = (0, 255, 0)
        elif status == "NOT DETECTED":
            label_color = (255, 0, 0)
        else:
            label_color = (255, 255, 0)

        cv2.putText(
            output_frame,
            f"Foreground probability: {foreground_probability:.2f}%",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            label_color,
            2,
        )
        cv2.putText(
            output_frame,
            f"Status: {status}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            label_color,
            2,
        )
        timing_text = "Model update interval: 1.0 s"
        if timing:
            timing_text += f" | Inference: {timing['inference_ms']:.1f} ms"
        cv2.putText(
            output_frame,
            timing_text,
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
        )
        return av.VideoFrame.from_ndarray(output_frame, format="rgb24")

# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>
.stApp {
    background-color: white;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

.topbar {
    background-color: #1f4328;
    color: white;
    font-size: 24px;
    font-weight: 500;
    height: 110px;
    display: flex;
    align-items: center;
    padding: 0 28px;
    gap: 18px;
}

.nav-sep {
    margin: 0 10px;
}

.brand-box {
    background-color: #6d7f2b;
    height: 110px;
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 34px;
    font-weight: 800;
    white-space: nowrap;
    min-width: 320px;
}

.blue-text {
    color: #143f8f;
}

.gold-text {
    color: #c89b3c;
}

.green-title {
    color: #1f4328;
    font-weight: 800;
}

.orange-title {
    color: #d99a24;
    font-weight: 800;
}

.hero-text {
    font-size: 26px;
    color: #65723f;
    line-height: 1.35;
}

.footer {
    background-color: #1f4328;
    color: white;
    padding: 22px;
    text-align: center;
    font-size: 24px;

    display:flex;
    justify-content:center;
    align-items:center;
}

.footer a:hover {
    text-decoration: underline !important;
}           

.upload-frame {
    border: 4px dashed #6d7f2b;
    padding: 65px 45px;
    min-height: 320px;
    text-align: center;
    border-radius: 6px;

    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.tips-box {
    background-color: #f4a21c;
    color: black;
    padding: 22px 35px;
    border-radius: 35px;
    font-size: 18px;
}

.upload-center {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

[data-testid="stFileUploader"] {
    width: 100% !important;
}

[data-testid="stFileUploader"] > div {
    width: 100% !important;
}

.nav-link {
    color: white !important;
    text-decoration: none !important;
}

.nav-link:hover {
    color: white !important;
    text-decoration: underline !important;
}

.nav-link:visited {
    color: white !important;
}  

.image-frame {
    background-color: #6d7f2b;
    padding: 28px;
    border-radius: 38px;
    text-align: center;
}
            
/* Green Live Detection button */
.st-key-live_detection_btn button {
    background-color: #2f7d32 !important;
    color: white !important;
    border: 2px solid #1f5c25 !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    white-space: nowrap !important;
}

.st-key-live_detection_btn button:hover {
    background-color: #236126 !important;
    color: white !important;
}

.result-card {
    background-color: #6d7f2b;
    color: white;
    padding: 38px;
    border-radius: 38px;
    font-size: 25px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
pages = ["Home", "Detection", "Results", "About"]

query_page = st.query_params.get("page", None)

if "page" not in st.session_state:
    st.session_state.page = query_page if query_page in pages else "Home"

if query_page in pages:
    st.session_state.page = query_page

page = st.sidebar.radio(
    "Navigation",
    pages,
    index=pages.index(st.session_state.page)
)

st.session_state.page = page

# =====================================================
# HEADER
# =====================================================
def top_navigation():
    col1, col2 = st.columns([2.7, 1.3], gap=None)

    with col1:
        icon_html = ""
        if PICTURE_8:
            ext = os.path.splitext(PICTURE_8)[1].lower().replace(".", "")
            icon_html = f'<img src="data:image/{ext};base64,{image_to_base64(PICTURE_8)}" width="40" style="vertical-align:middle; margin-right:12px;">'

        st.markdown(
            f"""
            <div class="topbar">
                {icon_html}
                <a href="?page=Home"
                    class="nav-link">
                    Home
                </a>
                <span class="nav-sep">|</span>
                <span>Universiti Teknologi PETRONAS</span>
                <span class="nav-sep">|</span>
                <span>STRIDE</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="brand-box">
                <span class="blue-text">CAMOU</span><span class="gold-text">Vision</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# =====================================================
# HOME PAGE
# =====================================================
if page == "Home":

    top_left, title_col, top_right = st.columns([1.2, 4, 2])

    with top_left:

        st.markdown(
        """
        <div style='margin-top:15px;'></div>
        """,
        unsafe_allow_html=True
        )

        if PICTURE_4:
            st.image(
                resize_image(PICTURE_4, TOP_LOGO_SIZE),
                use_container_width=False
            )

    with title_col:
        st.markdown(
            """
            <h1 style="font-size:72px; margin-top:10px;">
                <span style="color:#143f8f;">CAMOU</span><span style="color:#c89b3c;">Vision</span>
            </h1>
            <hr style="border:2px solid #143f8f; width:70%; margin-left:0;">
            """,
            unsafe_allow_html=True
        )

    with top_right:

        st.markdown("<br><br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if PICTURE_1:
                st.image(
                    resize_image(PICTURE_1, SMALL_LOGO_SIZE),
                    use_container_width=False
                )
        with c2:
            if PICTURE_2:
                st.image(
                    resize_image(PICTURE_2, SMALL_LOGO_SIZE),
                    use_container_width=False
                )

    st.write("")
    st.write("")

    left, right = st.columns([1.1, 1.2])

    with left:
        if PICTURE_5:
            st.image(
                resize_image(PICTURE_5, HERO_IMAGE_SIZE),
                use_container_width=False
            )
        else:
            st.info("Add picture5 inside external_image folder.")

    with right:
        st.markdown(
            """
            <h1 style="line-height:1.25;">
                <span class="orange-title">AI-Powered</span><br>
                <span class="green-title">Military Camouflage</span><br>
                <span class="orange-title">Detection System</span>
            </h1>
            <hr style="border:2px solid #d99a24; width:25%; margin-left:0;">
            <p class="hero-text">
                Detect camouflaged military personnel in complex environments
                using deep learning and computer vision.
            </p>
            """,
            unsafe_allow_html=True
        )

        if st.button("Start Detection  ➜"):
            st.session_state.page = "Detection"
            st.rerun()

    st.write("")
    st.write("")

    footer_icon = ""

    if PICTURE_3:
        ext = os.path.splitext(PICTURE_3)[1].lower().replace(".", "")
        img_base64 = image_to_base64(PICTURE_3)

        footer_icon = f'<img src="data:image/{ext};base64,{img_base64}" width="45" style="vertical-align:middle; margin-right:12px;">'

    st.markdown(
    f"""
    <div class="footer">
        <span>{footer_icon}</span>
        <span>
            <a href="?page=About" style="color:white; text-decoration:none;">
                About
            </a>
            &nbsp;&nbsp; | &nbsp;&nbsp;
            Universiti Teknologi PETRONAS
            &nbsp;&nbsp; | &nbsp;&nbsp;
            STRIDE
        </span>
    </div>
    """,
    unsafe_allow_html=True
    )

# =====================================================
# DETECTION PAGE
# =====================================================
elif page == "Detection":

    top_navigation()

    st.markdown(
        "<h1 style='text-align:center; color:#1f4328; font-size:38px;'>Upload Image, Video or Use Live Detection</h1>",
        unsafe_allow_html=True
    )

    # Center whole upload area
    left_space, upload_col, right_space = st.columns([1, 2.4, 1])

    with upload_col:

        upload_box = st.container(border=True)

        with upload_box:
            st.markdown(
                """
                <style>
                div[data-testid="stVerticalBlockBorderWrapper"] {
                    border: 3px dashed #6d7f2b !important;
                    border-radius: 4px !important;
                    min-height: 420px !important;
                    padding-top: 80px !important;
                    padding-left: 80px !important;
                    padding-right: 80px !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            center_left, center_mid, center_right = st.columns([1, 1, 1])

            with center_mid:

                st.markdown('<div class="upload-center">', unsafe_allow_html=True)

                if PICTURE_6:
                    st.image(
                        resize_image(PICTURE_6, UPLOAD_ICON_SIZE),
                        use_container_width=False
                    )

                # Default selected mode
                if "selected_input_mode" not in st.session_state:
                    st.session_state["selected_input_mode"] = "Image"

                btn1, btn2, btn3 = st.columns([1.1, 1, 1.6])

                with btn1:
                    if st.button("Image", use_container_width=True, key="image_mode_btn"):
                        st.session_state["selected_input_mode"] = "Image"

                with btn2:
                    if st.button("Video", use_container_width=True, key="video_mode_btn"):
                        st.session_state["selected_input_mode"] = "Video"

                with btn3:
                    if st.button(
                        "Live Detection",
                        use_container_width=True,
                        key="live_detection_btn"
                    ):
                        st.session_state["selected_input_mode"] = "Live Detection"

                input_mode = st.session_state["selected_input_mode"]

                uploaded_file = None

                if input_mode == "Image":
                    uploaded_file = st.file_uploader(
                        "Drag & Drop or browse image",
                        type=["jpg", "jpeg", "png"],
                        key="image_uploader"
                    )

                elif input_mode == "Video":
                    uploaded_file = st.file_uploader(
                        "Drag & Drop or browse video",
                        type=["mp4", "avi", "mov"],
                        key="video_uploader"
                    )

                elif input_mode == "Live Detection":
                    st.info(
                        "The live stream remains active while SINet + GRA updates "
                        "the camouflage mask every 1 second."
                    )

                    model, device = load_sinet_gra_model()

                    webrtc_streamer(
                        key="camouflage_live_detection",
                        video_processor_factory=lambda: CamouflageLiveProcessor(
                            model=model,
                            device=device,
                            interval_seconds=1.0
                        ),
                        media_stream_constraints={
                            "video": True,
                            "audio": False
                        },
                        async_processing=True
                    )

                st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()

            if input_mode == "Image":
                st.session_state["input_mode"] = "Image"
                st.session_state["uploaded_image_bytes"] = uploaded_bytes
                st.session_state["uploaded_image_name"] = uploaded_file.name

                st.success("Image uploaded successfully.")
                preview_image = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
                st.image(preview_image, caption="Uploaded Image Preview", width=300)

                matching_gt_path = find_matching_ground_truth(uploaded_file.name)
                if matching_gt_path:
                    st.info("Matching ground-truth mask found: internal evaluation mode.")
                    analysis_type = "Internal Evaluation"
                else:
                    st.info("No matching mask found: external prediction mode.")
                    analysis_type = "External Prediction"

                if st.button("Run Image Analysis", use_container_width=True):
                    with st.spinner("Running SINet + GRA analysis..."):
                        (
                            mask_np,
                            foreground_probability,
                            status,
                            mask_area_ratio,
                            _,
                            timing,
                        ) = run_sinet_gra_detection(preview_image)

                        prediction_overlay = create_prediction_overlay(preview_image, mask_np)

                        st.session_state["prediction_mask"] = mask_np
                        st.session_state["prediction_overlay"] = prediction_overlay
                        st.session_state["foreground_probability"] = foreground_probability
                        st.session_state["status"] = status
                        st.session_state["mask_area_ratio"] = mask_area_ratio
                        st.session_state["timing"] = timing
                        st.session_state["analysis_type"] = analysis_type

                        for key in [
                            "ground_truth_mask",
                            "comparison_overlay",
                            "sample_dice",
                            "sample_iou",
                            "ground_truth_path",
                        ]:
                            st.session_state.pop(key, None)

                        if matching_gt_path:
                            gt_binary = prepare_ground_truth_mask(
                                matching_gt_path, preview_image.size
                            )
                            sample_dice, sample_iou = calculate_sample_metrics(
                                mask_np, gt_binary
                            )
                            comparison_overlay = create_comparison_overlay(
                                preview_image, mask_np, gt_binary
                            )
                            st.session_state["ground_truth_mask"] = gt_binary * 255
                            st.session_state["comparison_overlay"] = comparison_overlay
                            st.session_state["sample_dice"] = sample_dice
                            st.session_state["sample_iou"] = sample_iou
                            st.session_state["ground_truth_path"] = matching_gt_path

                    st.session_state["detection_done"] = True
                    st.session_state.page = "Results"
                    st.rerun()

            else:
                st.session_state["input_mode"] = "Video"
                st.session_state["uploaded_video_bytes"] = uploaded_bytes
                st.session_state["uploaded_video_name"] = uploaded_file.name

                st.success("Video uploaded successfully.")
                st.video(uploaded_bytes)

                st.info("The system will process one frame every 2 seconds.")

                if st.button("Run Video Detection", use_container_width=True):
                    with st.spinner("Processing video frames every 2 seconds using SINet + GRA..."):
                        video_results = process_video_every_2_seconds(uploaded_bytes)

                    st.session_state["video_results"] = video_results
                    st.session_state["detection_done"] = True
                    st.session_state.page = "Results"
                    st.rerun()

    # Tips section
    tip_left, tip_mid, tip_right = st.columns([1, 2, 1])

    with tip_mid:
        tip_icon_html = ""

        if PICTURE_7:
            ext = os.path.splitext(PICTURE_7)[1].lower().replace(".", "")
            tip_icon_html = f"""
            <img src="data:image/{ext};base64,{image_to_base64(PICTURE_7)}"
            width="55"
            style="vertical-align:middle; margin-right:8px;">
            """

        st.markdown(
            f"""
            <h3 style="color:#1f4328;">
                {tip_icon_html} Tips for best results
            </h3>

            <div class="tips-box">
            <ul>
                <li>Upload a clear RGB image or video (.jpg, .jpeg, .png, .mp4, .avi, .mov).</li>
                <li>For video input, the system processes one frame every 2 seconds.</li>
                <li>Ensure the camouflaged soldier occupies a reasonable portion of the image.</li>
                <li>Use images with good lighting conditions.</li>
                <li>Keep the target within the main field of view.</li>
                <li>Images captured from eye-level, drone, or surveillance perspectives are supported.</li>
                <li>Recommended image resolution: 256×256 pixels or higher.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

# =====================================================
# RESULTS PAGE
# =====================================================
elif page == "Results":

    top_navigation()
    input_mode = st.session_state.get("input_mode", "Image")

    if input_mode == "Video":
        video_results = st.session_state.get("video_results", [])
        st.markdown(
            "<h1 style='text-align:center; color:#1f4328;'>Video Detection Results</h1>",
            unsafe_allow_html=True,
        )

        if not video_results:
            st.warning("No video frames were processed.")
            st.stop()

        summary_data = [
            {
                "Timestamp (s)": f"{r['time']:.0f}s",
                "Foreground Probability (%)": f"{r['foreground_probability']:.2f}",
                "Status": r["status"],
                "Total Time (ms)": f"{r['timing']['total_ms']:.1f}",
            }
            for r in video_results
        ]
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        detected_count = sum(r["status"] == "DETECTED" for r in video_results)
        total_count = len(video_results)
        avg_probability = sum(r["foreground_probability"] for r in video_results) / total_count
        avg_total_ms = sum(r["timing"]["total_ms"] for r in video_results) / total_count
        overall_status = "DETECTED" if detected_count > 0 else "NOT DETECTED"
        overall_color = "#62ff5f" if overall_status == "DETECTED" else "#ff3b3b"

        st.markdown(
            f"""
            <div class="result-card">
                <div style="display:flex; justify-content:space-between; gap:40px;">
                    <div>
                        Model: Adapted SINet + GRA<br>
                        Video sampling: every 2 seconds<br>
                        Processed frames: {total_count}<br>
                        Detected frames: {detected_count}<br>
                        Mean total time: {avg_total_ms:.1f} ms
                    </div>
                    <div>
                        Mean foreground probability:<br>
                        <span style="font-size:42px; color:{overall_color};">{avg_probability:.2f}%</span><br>
                        Overall status:<br>
                        <span style="font-size:42px; color:{overall_color};">{overall_status}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for result in video_results:
            st.markdown(
                f"<h2 style='color:#1f4328;'>Frame at {result['time']:.0f} seconds</h2>",
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("### Original Frame")
                st.image(result["frame"], use_container_width=True)
            with c2:
                st.markdown("### Prediction Mask")
                st.image(result["mask"], clamp=True, use_container_width=True)
            with c3:
                st.markdown("### Prediction Overlay")
                st.image(result["overlay"], use_container_width=True)

            color = "#62ff5f" if result["status"] == "DETECTED" else "#ff3b3b"
            st.markdown(
                f"""
                <div class="result-card">
                    Foreground probability: <span style="color:{color};">{result['foreground_probability']:.2f}%</span>
                    &nbsp;&nbsp; | &nbsp;&nbsp; Status: <span style="color:{color};">{result['status']}</span>
                    &nbsp;&nbsp; | &nbsp;&nbsp; Total processing: {result['timing']['total_ms']:.1f} ms
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.stop()

    uploaded_image_bytes = st.session_state.get("uploaded_image_bytes")
    prediction_mask = st.session_state.get("prediction_mask")
    prediction_overlay = st.session_state.get("prediction_overlay")
    ground_truth_mask = st.session_state.get("ground_truth_mask")
    comparison_overlay = st.session_state.get("comparison_overlay")
    analysis_type = st.session_state.get("analysis_type", "External Prediction")

    if uploaded_image_bytes is None or prediction_mask is None:
        st.warning("No completed image analysis was found.")
        st.stop()

    image = Image.open(io.BytesIO(uploaded_image_bytes)).convert("RGB")

    if ground_truth_mask is not None and comparison_overlay is not None:
        st.markdown(
            "<h1 style='text-align:center; color:#1f4328;'>Qualitative Segmentation Evaluation</h1>",
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("### Original")
            st.image(image, use_container_width=True)
        with c2:
            st.markdown("### Ground Truth")
            st.image(ground_truth_mask, clamp=True, use_container_width=True)
        with c3:
            st.markdown("### Prediction")
            st.image(prediction_mask, clamp=True, use_container_width=True)
        with c4:
            st.markdown("### Comparison Overlay")
            st.image(comparison_overlay, use_container_width=True)
        st.caption("Overlay: green = correct foreground, red = false positive, blue = missed ground-truth region.")
    else:
        st.markdown(
            "<h1 style='text-align:center; color:#1f4328;'>External Image Prediction</h1>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Original")
            st.image(image, use_container_width=True)
        with c2:
            st.markdown("### Prediction Mask")
            st.image(prediction_mask, clamp=True, use_container_width=True)
        with c3:
            st.markdown("### Prediction Overlay")
            st.image(prediction_overlay, use_container_width=True)
        st.caption("Green overlay = model-predicted camouflage region. No ground-truth mask was available for this external image.")

    foreground_probability = st.session_state.get("foreground_probability", 0.0)
    status = st.session_state.get("status", "NOT DETECTED")
    timing = st.session_state.get("timing", {})
    sample_dice = st.session_state.get("sample_dice")
    sample_iou = st.session_state.get("sample_iou")

    st.markdown(f"### Analysis Type: {analysis_type}")
    status_color = "#1b8f3a" if status == "DETECTED" else "#d62828"
    if sample_dice is not None and sample_iou is not None:
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(
                textwrap.dedent(
                    f"""
                    <div>
                        <div style="font-size:14px; color:#333333; margin-bottom:8px;">
                            Status
                        </div>
                        <div style="font-size:34px; font-weight:500; color:{status_color};">
                            {status}
                        </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True
            )

        with m2:
            st.metric("Sample Dice", f"{sample_dice:.4f}")

        with m3:
            st.metric("Sample IoU", f"{sample_iou:.4f}")

        with m4:
            st.metric(
                "Foreground Probability",
                f"{foreground_probability:.2f}%"
            )

    else:
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(
                textwrap.dedent(
                    f"""
                    <div>
                        <div style="font-size:14px; color:#333333; margin-bottom:8px;">
                            Status
                        </div>
                        <div style="font-size:34px; font-weight:500; color:{status_color};">
                            {status}
                        </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True
            )

        with m2:
            st.metric(
                "Foreground Probability",
                f"{foreground_probability:.2f}%"
            )

        with m3:
            st.metric(
                "Total Processing",
                f"{timing.get('total_ms', 0):.1f} ms"
            )

        with m4:
            st.metric(
                "Pipeline FPS",
                f"{timing.get('pipeline_fps', 0):.2f}"
            )
# =====================================================
# ABOUT PAGE
# =====================================================
elif page == "About":

    top_navigation()

    st.markdown(
        "<h1 style='color:#1f4328;'>About CAMO Vision</h1>",
        unsafe_allow_html=True
    )

    st.write("""
    CAMO Vision is an AI-powered military camouflage detection system
    developed to identify camouflaged military personnel in complex natural
    environments. The system uses a deep learning segmentation model to
    highlight potential hidden targets from uploaded images.
    """)

    st.subheader("Project Objectives")
    st.write("""
    - Detect camouflaged military personnel automatically.
    - Improve visibility of hidden targets in complex scenes.
    - Demonstrate AI-based camouflage segmentation for defence-related applications.
    - Provide a simple web-based interface for image analysis.
    """)

    st.markdown("---")

    st.markdown(
        "<h1 style='color:#00508c;'>Project Supervisor</h1>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1,4])

    with col1:
        st.markdown("""
        <div style="
            width:120px;
            height:120px;
            border-radius:50%;
            background:linear-gradient(135deg,#1f5f7a,#c89b3c);
            display:flex;
            align-items:center;
            justify-content:center;
            color:white;
            font-size:42px;
            font-weight:bold;
            text-align:center;">
            NY
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Dr Norashikin Bt Yahya")
        st.write("Faculty Supervisor")

        st.link_button(
            "LinkedIn Profile",
            "https://www.linkedin.com/in/norashikin-yahya-0a78a531"
        )

    st.markdown(
        "<h1 style='color:#00508c;'>Project Developer</h1>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1,4])

    with col1:
        st.markdown("""
        <div style="
            width:120px;
            height:120px;
            border-radius:50%;
            background:linear-gradient(135deg,#1f5f7a,#c89b3c);
            display:flex;
            align-items:center;
            justify-content:center;
            color:white;
            font-size:42px;
            font-weight:bold;
            text-align:center;">
            NA
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Nurfarhanah Amirah Binti Muhammad Nadzri")
        st.write("Final Year Project Developer")

        st.link_button(
            "LinkedIn Profile",
            "https://www.linkedin.com/in/nurfarhanah-amirah-61261b283"
        )

    st.markdown("---")

    st.subheader("Collaboration")
    st.write("""
    Developed under Universiti Teknologi PETRONAS with collaboration from
    Science and Technology Research Institute for Defence (STRIDE).
    """)