import os
import base64
import streamlit as st
from PIL import Image
import numpy as np

from PIL import Image

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
    page_title="STRIDE Vision",
    page_icon="🎯",
    layout="wide"
)

# =====================================================
# PATHS
# =====================================================
BASE_DIR = r"C:\Users\User\Documents\fyp_military"
IMAGE_DIR = os.path.join(BASE_DIR, "external_image")

def get_image(name):
    for ext in [".png", ".jpg", ".jpeg"]:
        path = os.path.join(IMAGE_DIR, name + ext)
        if os.path.exists(path):
            return path
    return None

def image_to_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

PICTURE_1 = get_image("picture1")
PICTURE_2 = get_image("picture2")
PICTURE_3 = get_image("picture3")
PICTURE_4 = get_image("picture4")
PICTURE_5 = get_image("picture5")
PICTURE_6 = get_image("picture6")
PICTURE_7 = get_image("picture7")
PICTURE_8 = get_image("picture8")

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
                <span class="blue-text">STRIDE</span><span class="gold-text">Vision</span>
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
                <span style="color:#143f8f;">STRIDE</span><span style="color:#c89b3c;">Vision</span>
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
        "<h1 style='text-align:center; color:#1f4328; font-size:38px;'>Upload Image</h1>",
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

                uploaded_file = st.file_uploader(
                    "Drag & Drop or browse",
                    type=["jpg", "jpeg", "png"]
                )

                st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            st.session_state["uploaded_file"] = uploaded_file
            st.success("Image uploaded successfully.")

            st.image(
                uploaded_file,
                caption="Uploaded Image Preview",
                width=300
            )

            if st.button("Run Detection", use_container_width=True):
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
                <li>Upload a clear RGB image (.jpg, .jpeg, .png).</li>
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

    uploaded_file = st.session_state.get("uploaded_file", None)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<h1 style='text-align:center; color:#1f4328;'>Original Image</h1>",
            unsafe_allow_html=True
        )
        st.markdown("<div class='image-frame'>", unsafe_allow_html=True)

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_container_width=True)
        elif PICTURE_5:
            st.image(PICTURE_5, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            "<h1 style='text-align:center; color:#1f4328;'>Prediction Mask</h1>",
            unsafe_allow_html=True
        )
        st.markdown("<div class='image-frame'>", unsafe_allow_html=True)

        if uploaded_file is not None:
            dummy_mask = np.zeros((320, 320), dtype=np.uint8)
            dummy_mask[90:240, 120:230] = 255
            st.image(dummy_mask, clamp=True, use_container_width=True)
        elif PICTURE_5:
            st.image(PICTURE_5, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    confidence = 88.7

    if confidence >= 50:
        status = "DETECTED"
        color = "#62ff5f"
    else:
        status = "NOT DETECTED"
        color = "#ff3b3b"

    st.markdown(
        f"""
        <div class="result-card">
            <div style="display:flex; justify-content:space-between; gap:40px;">
                <div>
                    Model: SINet + GRA<br>
                    Validation Dice Score:
                    <span style="color:#62ff5f;">0.8671</span><br>
                    Validation IoU Score:
                    <span style="color:#62ff5f;">0.7874</span>
                </div>
                <div>
                    Detection Confidence:<br>
                    <span style="font-size:42px; color:{color};">{confidence}%</span><br>
                    Status:<br>
                    <span style="font-size:42px; color:{color};">{status}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================================
# ABOUT PAGE
# =====================================================
elif page == "About":

    top_navigation()

    st.markdown(
        "<h1 style='color:#1f4328;'>About STRIDE Vision</h1>",
        unsafe_allow_html=True
    )

    st.write("""
    STRIDE Vision is an AI-powered military camouflage detection system
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