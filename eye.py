import os
import cv2
import torch
import timm
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from PIL import Image
from torchvision import transforms

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="RetinaCare",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# PROJECT SETTINGS
# ==================================================

APP_NAME_1 = "Retina"
APP_NAME_2 = "Care"

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "vit_b16_final.pth"
)s
CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

CLASS_NAMES_TA = [
    "பாதிப்பு இல்லை",
    "லேசான பாதிப்பு",
    "மிதமான பாதிப்பு",
    "கடுமையான பாதிப்பு",
    "தீவிர பாதிப்பு",
]

CLASS_COUNTS = [25810, 2443, 5292, 873, 708]
TOTAL_IMAGES = 35126

# Evaluation on the provided test CSV
ACCURACY = 87.33
MACRO_F1 = 0.789
QWK = 0.8309

MODEL_NAME = "ViT-B16"
IMG_SIZE = 256

DEVICE = torch.device("cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

GRADE_COLORS = [
    "#22c55e",
    "#4ade80",
    "#f5b942",
    "#f97373",
    "#a78bfa"
]

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])

# ==================================================
# LANGUAGE CONTENT
# ==================================================

TEXT = {
    "English": {
        "home": "Home",
        "analytics": "Dataset Analytics",
        "performance": "Model Performance",
        "about": "About DR",
        "prevention": "Prevention & Care",
        "contact": "Contact Us",
        "tagline": "AI-Based Diabetic Retinopathy Screening System",
        "motto": "Early Detection  •  Better Care  •  Clearer Vision",
        "total": "Total Images",
        "accuracy": "Accuracy",
        "classes": "Classes",
        "model": "Model",
        "upload": "Upload Retinal Image",
        "upload_help": "Upload a retinal fundus image (JPG / PNG)",
        "run_prediction": "Analyze Image",
        "uploaded_caption": "Uploaded Retinal Image",
        "analyzing": "Analyzing retinal image...",
        "predict": "Prediction Result",
        "confidence": "Confidence",
        "severity": "What this means",
        "probabilities": "Class Probabilities",
        "download": "Download Report",
        "no_image": "Upload an image to begin.",
        "no_result": "Result will appear here.",
        "invalid": "Unable to process this image.",
        "prediction_complete": "Prediction completed.",
        "grade_guidance": "Grade-wise Guidance",
        "grade": "Grade",
        "model_perf": "Model Performance",
        "dataset_snapshot": "Dataset Snapshot",
        "images": "Images",
        "about_dr": "About Diabetic Retinopathy",
        "what": "What is DR?",
        "causes": "Causes",
        "symptoms": "Symptoms",
        "prev": "Prevention",
        "know_more": "Diabetic Retinopathy - Know More",
        "languages": "Languages",
        "lang_avail": "Available languages",
        "gradcam_note": "The heatmap highlights image regions receiving stronger model attention. This is an explanation aid, not a diagnosis.",
        "original": "Original Image",
        "gradcam_title": "ViT Attention Heatmap",
        "gradcam_failed": "Attention heatmap could not be generated for this image.",
        "disclaimer": "AI output is not a medical diagnosis. Consult an ophthalmologist.",
        "perf_note": "The displayed accuracy is from the provided test CSV evaluation. It is not independently verified clinical performance.",
        "dataset_note": "Class counts are based on the prepared EyePACS dataset.",
        "class_col": "Class",
        "config": "Model Configuration",
        "contact_text": "For project queries, reach out using the details below.",
        "footer": "RetinaCare | Academic Deep Learning Project",
        "footer2": "Built using Python, PyTorch, OpenCV and Streamlit",
        "tiles": [
            ("🧬", "Risk Factors", "Long diabetes, high sugar"),
            ("🛡️", "Prevention Tips", "Sugar, BP, checkups"),
            ("📋", "Management", "Laser, injection"),
            ("🩺", "When to See Doctor", "Blurry or dark spots"),
        ],
        "guidance": [
            "Normal. Continue regular eye checkups.",
            "Maintain healthy diet, control blood sugar.",
            "Consult an ophthalmologist.",
            "Seek immediate medical attention.",
            "Proliferative DR. Get urgent treatment.",
        ],
        "severity_text": [
            "No clear signs of diabetic retinopathy were detected. Continue routine diabetic eye screening.",
            "Early retinal changes may be present. Consult an eye specialist for regular follow-up.",
            "Moderate retinal changes may be present. Arrange an ophthalmologist appointment.",
            "Significant retinal changes may be present. Seek prompt specialist evaluation.",
            "Advanced retinal changes may be present. Seek urgent ophthalmologist evaluation.",
        ],
        "about_tabs": {
            "what": "Diabetic Retinopathy is an eye disease caused by diabetes. It damages the blood vessels in the retina and may lead to vision loss.",
            "causes": "Long-term high blood sugar, high blood pressure, high cholesterol, smoking and a long duration of diabetes damage the tiny retinal vessels.",
            "symptoms": "Blurred vision, floating dark spots, poor night vision, faded colours and sudden vision loss. Early stages often have no symptoms.",
            "prev": "Control blood sugar and blood pressure, avoid smoking, eat healthy, stay active and get a dilated eye exam at least once a year.",
        },
        "prevention_list": [
            "Keep blood sugar within the target range set by your doctor.",
            "Monitor and control blood pressure and cholesterol.",
            "Get a dilated eye exam at least once every year.",
            "Eat a balanced diet and exercise regularly.",
            "Avoid smoking and limit alcohol.",
            "Visit a doctor quickly if your vision suddenly changes.",
        ],
    },

    "தமிழ்": {
        "home": "முகப்பு",
        "analytics": "தரவு விவரங்கள்",
        "performance": "மாடல் செயல்திறன்",
        "about": "DR பற்றி",
        "prevention": "தடுப்பு & பராமரிப்பு",
        "contact": "தொடர்புக்கு",
        "tagline": "AI அடிப்படையிலான நீரிழிவு விழித்திரை பரிசோதனை",
        "motto": "முன்கூட்டியே கண்டறிதல்  •  சிறந்த பராமரிப்பு  •  தெளிவான பார்வை",
        "total": "மொத்த படங்கள்",
        "accuracy": "துல்லியம்",
        "classes": "வகைகள்",
        "model": "மாடல்",
        "upload": "விழித்திரை படத்தை பதிவேற்றவும்",
        "upload_help": "விழித்திரை படத்தை பதிவேற்றவும் (JPG / PNG)",
        "run_prediction": "படத்தை ஆய்வு செய்",
        "uploaded_caption": "பதிவேற்றிய படம்",
        "analyzing": "படம் ஆய்வு செய்யப்படுகிறது...",
        "predict": "கணிப்பு முடிவு",
        "confidence": "நம்பகத்தன்மை",
        "severity": "இதன் பொருள்",
        "probabilities": "வகைவாரி வாய்ப்பு",
        "download": "அறிக்கையை பதிவிறக்கு",
        "no_image": "தொடங்க ஒரு படத்தை பதிவேற்றவும்.",
        "no_result": "முடிவு இங்கே தோன்றும்.",
        "invalid": "இந்த படத்தை செயலாக்க முடியவில்லை.",
        "prediction_complete": "கணிப்பு முடிந்தது.",
        "grade_guidance": "நிலைவாரி வழிகாட்டுதல்",
        "grade": "நிலை",
        "model_perf": "மாடல் செயல்திறன்",
        "dataset_snapshot": "தரவுத்தொகுப்பு சுருக்கம்",
        "images": "படங்கள்",
        "about_dr": "நீரிழிவு விழித்திரை பாதிப்பு பற்றி",
        "what": "DR என்றால்?",
        "causes": "காரணங்கள்",
        "symptoms": "அறிகுறிகள்",
        "prev": "தடுப்பு",
        "know_more": "நீரிழிவு விழித்திரை பாதிப்பு - மேலும் அறிக",
        "languages": "மொழிகள்",
        "lang_avail": "கிடைக்கும் மொழிகள்",
        "gradcam_note": "இந்த ஹீட்மேப் மாடல் அதிக கவனம் செலுத்திய படப் பகுதிகளைக் காட்டுகிறது. இது விளக்க உதவி மட்டுமே, நோயறிதல் அல்ல.",
        "original": "அசல் படம்",
        "gradcam_title": "ViT Attention ஹீட்மேப்",
        "gradcam_failed": "இந்த படத்திற்கு Attention ஹீட்மேப் உருவாக்க முடியவில்லை.",
        "disclaimer": "AI கணிப்பு மருத்துவ நோயறிதல் அல்ல. கண் மருத்துவரை அணுகவும்.",
        "perf_note": "காட்டப்படும் துல்லியம் வழங்கப்பட்ட test CSV மதிப்பீட்டின் அடிப்படையிலானது. இது சரிபார்க்கப்பட்ட மருத்துவ செயல்திறன் அல்ல.",
        "dataset_note": "தயாரிக்கப்பட்ட EyePACS தரவுத்தொகுப்பின் வகை எண்ணிக்கைகள்.",
        "class_col": "வகை",
        "config": "மாடல் அமைப்பு",
        "contact_text": "திட்ட கேள்விகளுக்கு கீழுள்ள விவரங்களைப் பயன்படுத்தவும்.",
        "footer": "RetinaCare | கல்வி சார்ந்த ஆழ் கற்றல் திட்டம்",
        "footer2": "Python, PyTorch, OpenCV, Streamlit மூலம் உருவாக்கப்பட்டது",
        "tiles": [
            ("🧬", "ஆபத்து காரணிகள்", "நீண்டகால நீரிழிவு"),
            ("🛡️", "தடுப்பு குறிப்புகள்", "சர்க்கரை, BP, பரிசோதனை"),
            ("📋", "சிகிச்சை முறைகள்", "லேசர், ஊசி, அறுவை"),
            ("🩺", "மருத்துவரை எப்போது", "மங்கல், கருப்பு புள்ளி"),
        ],
        "guidance": [
            "இயல்பு. வழக்கமான கண் பரிசோதனை தொடரவும்.",
            "ஆரோக்கிய உணவு, சர்க்கரை கட்டுப்பாடு.",
            "கண் மருத்துவரை அணுகவும்.",
            "உடனடி மருத்துவ உதவி பெறவும்.",
            "தீவிர நிலை. அவசர சிகிச்சை தேவை.",
        ],
        "severity_text": [
            "படத்தில் நீரிழிவு விழித்திரை பாதிப்புக்கான தெளிவான அறிகுறிகள் தெரியவில்லை. வழக்கமான கண் பரிசோதனையைத் தொடரவும்.",
            "ஆரம்ப நிலை மாற்றங்கள் இருக்கலாம். கண் மருத்துவரை அணுகி தொடர்ந்து பரிசோதனை செய்யவும்.",
            "மிதமான பாதிப்பு இருக்கலாம். விரைவில் கண் மருத்துவரை அணுகவும்.",
            "கடுமையான மாற்றங்கள் இருக்கலாம். தாமதிக்காமல் கண் நிபுணரை அணுகவும்.",
            "மிகவும் முன்னேறிய பாதிப்பு இருக்கலாம். உடனடியாக கண் நிபுணரை அணுகவும்.",
        ],
        "about_tabs": {
            "what": "நீரிழிவு விழித்திரை பாதிப்பு என்பது நீரிழிவால் ஏற்படும் கண் நோய். இது விழித்திரையின் இரத்தக் குழாய்களை சேதப்படுத்தி பார்வை இழப்புக்கு வழிவகுக்கலாம்.",
            "causes": "நீண்டகால அதிக சர்க்கரை, உயர் இரத்த அழுத்தம், அதிக கொலஸ்ட்ரால், புகைப்பழக்கம் ஆகியவை விழித்திரை குழாய்களை பாதிக்கும்.",
            "symptoms": "மங்கலான பார்வை, கருப்பு புள்ளிகள், இரவு பார்வை குறைவு, திடீர் பார்வை இழப்பு. ஆரம்பத்தில் அறிகுறிகள் இல்லாமலும் இருக்கலாம்.",
            "prev": "சர்க்கரை மற்றும் இரத்த அழுத்தத்தை கட்டுப்படுத்தவும், புகையை தவிர்க்கவும், ஆண்டுக்கு ஒருமுறை கண் பரிசோதனை செய்யவும்.",
        },
        "prevention_list": [
            "மருத்துவர் கூறிய அளவில் சர்க்கரையை வைத்திருங்கள்.",
            "இரத்த அழுத்தம் மற்றும் கொலஸ்ட்ராலை கட்டுப்படுத்துங்கள்.",
            "ஆண்டுக்கு ஒருமுறையாவது கண் பரிசோதனை செய்யுங்கள்.",
            "சமச்சீர் உணவு மற்றும் தினசரி உடற்பயிற்சி செய்யுங்கள்.",
            "புகை மற்றும் மதுவை தவிர்க்கவும்.",
            "பார்வை திடீரென மாறினால் உடனே மருத்துவரை பார்க்கவும்.",
        ],
    },
}

# ==================================================
# CUSTOM CSS - ORIGINAL DESIGN
# ==================================================

st.markdown("""
<style>

.stApp {
    background: radial-gradient(1200px 600px at 80% -10%, #12356b 0%, #071426 55%, #050d1a 100%);
    color: #E8F0FC;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

[data-testid="stMarkdownContainer"] *,
.stButton button, .stDownloadButton button,
button[data-baseweb="tab"], label {
    word-break: keep-all !important;
    overflow-wrap: normal !important;
    hyphens: none !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1D35, #0a2347);
    border-right: 1px solid #1E3A5F;
    min-width: 270px !important;
    max-width: 270px !important;
}

[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    white-space: nowrap !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 9px 12px;
    border-radius: 10px;
    width: 100%;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: #1b64d8;
}

[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none;
}

.brand {
    text-align: center;
    padding: 6px 0 4px 0;
}

.brand-name {
    font-size: 34px;
    font-weight: 800;
    color: #fff;
    white-space: nowrap;
}

.brand-name span, .hero-title span {
    color: #22d3ee;
}

.brand-sub {
    color: #b9c9e0;
    font-size: 13px;
    line-height: 1.5;
}

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 18px;
    padding: 28px 36px;
    border: 1px solid #24517B;
    background:
        radial-gradient(circle at 92% 50%, #ff9a3c 0%, #d4561a 14%, #7a2410 24%, rgba(0,0,0,0) 25%),
        radial-gradient(circle at 92% 50%, rgba(255,140,60,.25) 0%, rgba(0,0,0,0) 40%),
        linear-gradient(120deg, #0a2a55, #07162d 70%);
}

.hero-title {
    font-size: 54px;
    font-weight: 800;
    color: #fff;
    line-height: 1.1;
    white-space: nowrap;
}

.hero-sub {
    font-size: 22px;
    color: #e2ecfb;
    margin-top: 4px;
    white-space: nowrap;
}

.hero-motto {
    font-size: 16px;
    color: #7fd6ff;
    font-style: italic;
    margin-top: 12px;
    white-space: nowrap;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(16, 35, 60, 0.82);
    border: 1px solid #27507a !important;
    border-radius: 16px !important;
}

.panel-title {
    font-size: 19px;
    font-weight: 700;
    color: #F1F6FF;
    white-space: nowrap;
    margin-bottom: 6px;
}

.metric-card {
    background: linear-gradient(135deg, #102B4A, #153B60);
    padding: 16px 20px;
    border-radius: 14px;
    border: 1px solid #24517B;
    height: 104px;
    box-sizing: border-box;
}

.metric-label {
    color: #A8BED8;
    font-size: 14px;
    white-space: nowrap;
}

.metric-value {
    color: #FFFFFF;
    font-size: 20px;
    font-weight: 800;
    margin-top: 8px;
    white-space: nowrap;
}

.grade-row {
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 9px 12px;
    margin-bottom: 8px;
}

.grade-badge {
    flex: 0 0 82px;
    text-align: center;
    color: #06231a;
    font-weight: 700;
    font-size: 14px;
    border-radius: 9px;
    padding: 5px 0;
    white-space: nowrap;
}

.grade-text {
    flex: 1 1 auto;
    color: #d6e3f5;
    font-size: 14px;
    line-height: 1.4;
}

.result-title {
    font-size: 26px;
    font-weight: 800;
    white-space: nowrap;
}

.result-conf {
    font-size: 15px;
    color: #A8BED8;
    margin-top: 6px;
    white-space: nowrap;
}

.info-text {
    color: #B7C9DF;
    line-height: 1.7;
    font-size: 15px;
}

.ring {
    width: 150px;
    height: 150px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 6px auto;
}

.ring-in {
    width: 116px;
    height: 116px;
    border-radius: 50%;
    background: #0c1f3a;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.ring-val {
    font-size: 28px;
    font-weight: 800;
    color: #fff;
    white-space: nowrap;
}

.ring-lab {
    font-size: 14px;
    color: #b9c9e0;
    white-space: nowrap;
}

.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    padding: 7px 0;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    font-size: 14px;
}

.stat-row .k {
    color: #A8BED8;
    white-space: nowrap;
}

.stat-row .v {
    color: #fff;
    font-weight: 700;
    white-space: nowrap;
}

.tile {
    background: linear-gradient(135deg, #102B4A, #153B60);
    border: 1px solid #24517B;
    border-radius: 14px;
    padding: 14px 10px;
    text-align: center;
    height: 118px;
    box-sizing: border-box;
}

.tile .ic {
    font-size: 26px;
}

.tile .tt {
    color: #fff;
    font-weight: 700;
    font-size: 14px;
    margin-top: 4px;
    white-space: nowrap;
}

.tile .ds {
    color: #A8BED8;
    font-size: 12px;
    margin-top: 3px;
    white-space: nowrap;
}

.chip {
    display: inline-block;
    background: #12365e;
    border: 1px solid #2b5c91;
    color: #fff;
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 14px;
    margin-right: 8px;
    white-space: nowrap;
}

.footer {
    color: #8299B5;
    text-align: center;
    padding: 22px;
    font-size: 12px;
}

.stButton button, .stDownloadButton button {
    background: #087DBB;
    color: white;
    border-radius: 9px;
    border: none;
    font-weight: 700;
    white-space: nowrap;
}

.stButton button:hover, .stDownloadButton button:hover {
    background: #079EDC;
    color: white;
}

header[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #cfe0f5 !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: #e8f0fc !important;
    font-size: 16px;
}

label[data-baseweb="radio"] > div:first-child {
    display: none !important;
}

[data-testid="stFileUploader"] section {
    background: #0f2a4a;
    border: 1px dashed #3a6ea5;
}

[data-testid="stFileUploader"] section * {
    color: #cfe0f5 !important;
}

[data-testid="stFileUploader"] button {
    background: #087DBB !important;
    color: #fff !important;
    border: none;
}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    color-scheme: dark;
}

.stApp, [data-testid="stAppViewContainer"] {
    background: #071426 !important;
}

.stApp p, .stApp span, .stApp li, .stApp h1, .stApp h2,
.stApp h3, .stApp h4, .stApp label,
.stApp div[data-testid="stMarkdownContainer"] {
    color: #E8F0FC;
}

[data-testid="stSidebar"] {
    background: #0B1D35 !important;
}

[data-testid="stSidebar"] * {
    color: #E8F0FC;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #ffffff !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] * {
    color: #0b1d35 !important;
}

[data-baseweb="popover"] ul,
[data-baseweb="menu"] {
    background: #ffffff !important;
}

[data-baseweb="popover"] * {
    color: #0b1d35 !important;
}

div[data-testid="stAlert"] {
    background: #0f2a4a !important;
    border: 1px solid #2b5c91;
    border-radius: 12px;
}

div[data-testid="stAlert"] * {
    color: #dbe9fb !important;
}

[data-testid="stDataFrame"] {
    background: #0f2a4a;
    border-radius: 10px;
}

button[data-baseweb="tab"] p {
    color: #a8bed8 !important;
    white-space: nowrap !important;
}

button[data-baseweb="tab"][aria-selected="true"] p {
    color: #22d3ee !important;
}

</style>
""", unsafe_allow_html=True)

# Flatten HTML blocks
_orig_markdown = st.markdown


def _flat_markdown(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html") and isinstance(body, str):
        body = " ".join(
            line.strip()
            for line in body.splitlines()
            if line.strip()
        )
    return _orig_markdown(body, *args, **kwargs)


st.markdown = _flat_markdown

# ==================================================
# MODEL LOADING - ViT-B16
# ==================================================

@st.cache_resource
def load_model():
    try:
        model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            img_size=256,
            num_classes=5
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )

        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()

        # Enable explicit attention calculation for rollout.
        # This disables fused attention so attention matrices
        # can be captured from each transformer block.
        for block in model.blocks:
            block.attn.fused_attn = False

        return model, None

    except Exception as e:
        return None, str(e)


# ==================================================
# BEN GRAHAM PREPROCESSING
# Exact Kaggle Cell 3 preprocessing
# ==================================================

def crop_image_from_gray(img_np, tol=10):

    if img_np.ndim == 2:
        mask = img_np > tol
        return img_np[
            np.ix_(mask.any(1), mask.any(0))
        ]

    gray = cv2.cvtColor(
        img_np,
        cv2.COLOR_RGB2GRAY
    )

    mask = gray > tol

    if not np.any(mask):
        return img_np

    row_mask = mask.any(axis=1)
    col_mask = mask.any(axis=0)

    rmin, rmax = np.where(row_mask)[0][[0, -1]]
    cmin, cmax = np.where(col_mask)[0][[0, -1]]

    if rmax > rmin and cmax > cmin:
        return img_np[
            rmin:rmax + 1,
            cmin:cmax + 1
        ]

    return img_np


def ben_graham_top1_pipeline(
    image_pil,
    target_size=256,
    sigmaX=10
):

    img = np.array(
        image_pil.convert("RGB")
    )

    img = crop_image_from_gray(
        img,
        tol=10
    )

    img = cv2.resize(
        img,
        (target_size, target_size)
    )

    enhanced = cv2.addWeighted(
        img,
        4,
        cv2.GaussianBlur(
            img,
            (0, 0),
            sigmaX
        ),
        -4,
        128
    )

    h, w, _ = enhanced.shape

    center_x = w // 2
    center_y = h // 2

    radius = int(
        min(center_x, center_y) * 0.95
    )

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    cv2.circle(
        mask,
        (center_x, center_y),
        radius,
        255,
        -1
    )

    enhanced[mask == 0] = 128

    return Image.fromarray(enhanced)


def prepare_image(image):

    processed = ben_graham_top1_pipeline(
        image,
        target_size=256,
        sigmaX=10
    )

    tensor = VAL_TRANSFORM(processed)

    return tensor.unsqueeze(0)


# ==================================================
# VIT ATTENTION ROLLOUT HEATMAP
# ==================================================

def make_attention_heatmap(
    model,
    batch,
    original_image
):
    """
    Captures attention matrices from ViT transformer blocks,
    performs attention rollout, and overlays the attention map
    on the original uploaded image.

    This is an attention visualization, not Grad-CAM.
    """

    attention_maps = []
    hooks = []

    def attention_hook(module, input_data, output_data):
        attention_maps.append(
            output_data.detach().cpu()
        )

    try:
        # Register a hook on each block's attention dropout.
        for block in model.blocks:
            hook = block.attn.attn_drop.register_forward_hook(
                attention_hook
            )
            hooks.append(hook)

        with torch.no_grad():
            model(batch.to(DEVICE))

        if len(attention_maps) == 0:
            raise RuntimeError(
                "No attention matrices were captured."
            )

        # Remove hooks after collecting attention.
        for hook in hooks:
            hook.remove()

        hooks.clear()

        # Attention rollout
        num_tokens = attention_maps[0].shape[-1]

        identity = torch.eye(
            num_tokens
        ).unsqueeze(0)

        joint_attention = None

        for attention in attention_maps:

            # Average attention across heads.
            attention = attention.mean(dim=1)

            # Add residual connection.
            attention = attention + identity

            # Normalize each row.
            attention = attention / attention.sum(
                dim=-1,
                keepdim=True
            )

            if joint_attention is None:
                joint_attention = attention
            else:
                joint_attention = torch.bmm(
                    attention,
                    joint_attention
                )

        # Class token attention to image patches.
        mask = joint_attention[0, 0, 1:]

        patch_count = mask.numel()
        grid_size = int(np.sqrt(patch_count))

        if grid_size * grid_size != patch_count:
            raise RuntimeError(
                "Unexpected ViT patch-token dimensions."
            )

        mask = mask.reshape(
            grid_size,
            grid_size
        ).numpy()

        # Normalize heatmap.
        mask = np.maximum(mask, 0)

        if mask.max() <= 1e-8:
            raise ValueError(
                "Attention heatmap is empty."
            )

        mask = mask / mask.max()

        # Resize attention map to original image dimensions.
        heatmap = cv2.resize(
            mask.astype(np.float32),
            original_image.size,
            interpolation=cv2.INTER_CUBIC
        )

        heatmap = np.clip(
            heatmap,
            0,
            1
        )

        # Convert to colour heatmap.
        heatmap_uint8 = (
            heatmap * 255
        ).astype(np.uint8)

        coloured_heatmap = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        coloured_heatmap = cv2.cvtColor(
            coloured_heatmap,
            cv2.COLOR_BGR2RGB
        )

        base = np.array(
            original_image.convert("RGB")
        )

        # Overlay heatmap on original image.
        overlay = cv2.addWeighted(
            base,
            0.60,
            coloured_heatmap,
            0.40,
            0
        )

        return Image.fromarray(overlay)

    finally:
        # Ensure hooks are removed even if an error occurs.
        for hook in hooks:
            hook.remove()


# ==================================================
# PREDICTION + TEST-TIME AUGMENTATION
# ==================================================

def analyze_image(image):

    model, error = load_model()

    if model is None:
        raise RuntimeError(error)

    processed = prepare_image(image)

    # Horizontal flip TTA
    flipped = torch.flip(
        processed,
        dims=[3]
    )

    with torch.no_grad():

        original_output = model(
            processed.to(DEVICE)
        )

        flipped_output = model(
            flipped.to(DEVICE)
        )

        original_prob = torch.softmax(
            original_output,
            dim=1
        )

        flipped_prob = torch.softmax(
            flipped_output,
            dim=1
        )

        probabilities = (
            original_prob + flipped_prob
        ) / 2

        probabilities = probabilities.cpu().numpy()[0]

    probabilities = np.nan_to_num(
        probabilities,
        nan=0.0
    )

    total = probabilities.sum()

    if total <= 0:
        raise ValueError(
            "Invalid prediction output."
        )

    probabilities = probabilities / total

    predicted_class = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_class]
    ) * 100

    # Generate attention heatmap using original image pass.
    overlay = None
    heatmap_error = None

    try:
        overlay = make_attention_heatmap(
            model,
            processed,
            image
        )

    except Exception as e:
        heatmap_error = str(e)

    return {
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities.tolist(),
        "overlay": overlay,
        "gradcam_error": heatmap_error,
    }


# ==================================================
# SMALL UI HELPERS
# ==================================================

def metric_card(label, value):

    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


def panel_title(icon, title):

    st.markdown(
        f'<div class="panel-title">{icon} {title}</div>',
        unsafe_allow_html=True
    )


def style_fig(fig, height=320):

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=height,
        margin=dict(
            l=10,
            r=10,
            t=40,
            b=10
        ),
    )

    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)

    return fig


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown(
        f"""
        <div class="brand">
            <svg width="120" height="64" viewBox="0 0 120 64" xmlns="http://www.w3.org/2000/svg">
              <path d="M4 32 C24 4, 96 4, 116 32 C96 60, 24 60, 4 32 Z" fill="none" stroke="#22d3ee" stroke-width="4"/>
              <circle cx="60" cy="32" r="17" fill="#1b64d8"/>
              <circle cx="60" cy="32" r="8" fill="#06182f"/>
              <circle cx="66" cy="26" r="3" fill="#fff"/>
            </svg>
            <div class="brand-name">{APP_NAME_1}<span>{APP_NAME_2}</span></div>
            <div class="brand-sub">AI-Based Diabetic<br>Retinopathy Screening</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    language = st.selectbox(
        "🌐 Language / மொழி",
        ["English", "தமிழ்"]
    )

    t = TEXT[language]

    st.markdown("---")

    NAV_ICONS = {
        "home": "🏠",
        "analytics": "📊",
        "performance": "📈",
        "about": "ℹ️",
        "prevention": "🛡️",
        "contact": "✉️",
    }

    page = st.radio(
        "Navigation",
        list(NAV_ICONS.keys()),
        format_func=lambda k: (
            f"{NAV_ICONS[k]}  {t[k]}"
        ),
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.caption("ViT-B16  |  EyePACS  |  5-Class")
    st.caption("Academic Mini Project")


display_names = (
    CLASS_NAMES
    if language == "English"
    else CLASS_NAMES_TA
)


# ==================================================
# HERO - ALL PAGES
# ==================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">{APP_NAME_1}<span>{APP_NAME_2}</span></div>
        <div class="hero-sub">{t['tagline']}</div>
        <div class="hero-motto">{t['motto']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ==================================================
# HOME
# ==================================================

if page == "home":

    # ---- TOP METRICS ----

    cols = st.columns(4)

    metrics = [
        (t["total"], f"{TOTAL_IMAGES:,}"),
        (t["accuracy"], f"{ACCURACY}%"),
        (t["classes"], "05"),
        (t["model"], MODEL_NAME),
    ]

    for col, (label, value) in zip(cols, metrics):

        with col:
            st.markdown(
                metric_card(label, value),
                unsafe_allow_html=True
            )

    st.write("")

    # ---- ROW 1: UPLOAD + RESULT ----

    c_up, c_res = st.columns(2)

    with c_up:

        with st.container(border=True):

            panel_title("📤", t["upload"])

            st.caption(t["upload_help"])

            uploaded_file = st.file_uploader(
                t["upload"],
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )

            if uploaded_file:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                pred = st.session_state.get(
                    "prediction"
                )

                has_result = (
                    bool(pred)
                    and pred.get("filename") == uploaded_file.name
                )

                if has_result:

                    g1, g2 = st.columns(2)

                    with g1:

                        st.image(
                            image,
                            caption=t["original"],
                            use_container_width=True
                        )

                    with g2:

                        if pred.get("overlay") is not None:

                            st.image(
                                pred["overlay"],
                                caption=t["gradcam_title"],
                                use_container_width=True
                            )

                        else:

                            st.warning(
                                t["gradcam_failed"]
                            )

                            if pred.get("gradcam_error"):
                                st.caption(
                                    pred["gradcam_error"]
                                )

                    if pred.get("overlay") is not None:
                        st.caption(
                            t["gradcam_note"]
                        )

                else:

                    st.image(
                        image,
                        caption=t["uploaded_caption"],
                        use_container_width=True
                    )

                out = None

                if st.button(
                    t["run_prediction"],
                    use_container_width=True
                ):

                    with st.spinner(t["analyzing"]):

                        try:

                            out = analyze_image(image)

                            out["filename"] = uploaded_file.name

                            st.session_state["prediction"] = out

                        except Exception as e:

                            out = None

                            st.error(t["invalid"])
                            st.code(str(e))

                    if out is not None:
                        st.rerun()

            else:

                st.info(t["no_image"])

    with c_res:

        with st.container(border=True):

            panel_title("✅", t["predict"])

            if "prediction" in st.session_state:

                result = st.session_state["prediction"]

                cid = result["class"]

                color = GRADE_COLORS[cid]

                st.markdown(
                    f"""
                    <div class="result-title" style="color:{color};">
                        {t['grade']} {cid} - {display_names[cid]}
                    </div>
                    <div class="result-conf">
                        {t['confidence']}:
                        <b style="color:#fff;">
                            {result['confidence']:.2f}%
                        </b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write("")

                st.markdown(
                    f"**{t['severity']}**"
                )

                st.info(
                    t["severity_text"][cid]
                )

                prob_df = pd.DataFrame({
                    "Class": display_names,
                    "Probability": [
                        p * 100
                        for p in result["probabilities"]
                    ],
                })

                fig = px.bar(
                    prob_df,
                    x="Probability",
                    y="Class",
                    orientation="h",
                    text_auto=".1f",
                    color="Probability",
                    color_continuous_scale="Blues",
                )

                fig.update_layout(
                    coloraxis_showscale=False,
                    xaxis_title="%",
                    yaxis_title="",
                    yaxis=dict(
                        autorange="reversed"
                    ),
                )

                st.plotly_chart(
                    style_fig(fig, 250),
                    use_container_width=True
                )

                report_df = pd.DataFrame([{
                    "Image": result["filename"],
                    "Predicted Class": CLASS_NAMES[cid],
                    "Confidence (%)": round(
                        result["confidence"],
                        2
                    ),
                }])

                st.download_button(
                    label=f"⬇️ {t['download']}",
                    data=report_df.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="DR_Prediction_Report.csv",
                    mime="text/csv",
                )

            else:

                st.info(t["no_result"])

    st.write("")

    # ---- ROW 2: GRADE GUIDANCE ----

    with st.container(border=True):

        panel_title(
            "💡",
            t["grade_guidance"]
        )

        rows = ""

        for i in range(5):

            rows += f"""
            <div class="grade-row">
                <div class="grade-badge" style="background:{GRADE_COLORS[i]};">
                    {t['grade']} {i}
                </div>
                <div class="grade-text">
                    {t['guidance'][i]}
                </div>
            </div>
            """

        st.markdown(
            rows,
            unsafe_allow_html=True
        )

    st.write("")

    # ---- ROW 3: PERFORMANCE + DATASET ----

    c_perf, c_data = st.columns(2)

    with c_perf:

        with st.container(border=True):

            panel_title(
                "📈",
                t["model_perf"]
            )

            st.markdown(
                f"""
                <div class="ring" style="background:conic-gradient(#22d3ee {ACCURACY}%, #16304f 0);">
                    <div class="ring-in">
                        <div class="ring-val">{ACCURACY}%</div>
                        <div class="ring-lab">{t['accuracy']}</div>
                    </div>
                </div>

                <div class="stat-row">
                    <span class="k">{t['model']}</span>
                    <span class="v">ViT-B16</span>
                </div>

                <div class="stat-row">
                    <span class="k">Input</span>
                    <span class="v">256 x 256</span>
                </div>

                <div class="stat-row">
                    <span class="k">Macro F1</span>
                    <span class="v">0.789</span>
                </div>

                <div class="stat-row">
                    <span class="k">QWK</span>
                    <span class="v">0.8309</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with c_data:

        with st.container(border=True):

            panel_title(
                "🗂️",
                t["dataset_snapshot"]
            )

            rows = f"""
            <div class="stat-row">
                <span class="k">{t['total']}</span>
                <span class="v">{TOTAL_IMAGES:,}</span>
            </div>

            <div class="stat-row">
                <span class="k">Dataset</span>
                <span class="v">EyePACS</span>
            </div>
            """

            for i in range(5):

                rows += f"""
                <div class="stat-row">
                    <span class="k">{t['grade']} {i}</span>
                    <span class="v">{CLASS_COUNTS[i]:,}</span>
                </div>
                """

            st.markdown(
                rows,
                unsafe_allow_html=True
            )

    st.write("")

    # ---- ROW 4: ABOUT DR ----

    with st.container(border=True):

        panel_title(
            "ℹ️",
            t["about_dr"]
        )

        tabs = st.tabs([
            t["what"],
            t["causes"],
            t["symptoms"],
            t["prev"]
        ])

        for tab, key in zip(
            tabs,
            ["what", "causes", "symptoms", "prev"]
        ):

            with tab:

                st.markdown(
                    f'<div class="info-text">{t["about_tabs"][key]}</div>',
                    unsafe_allow_html=True,
                )

    st.write("")

    # ---- ROW 5: KNOW MORE ----

    with st.container(border=True):

        panel_title(
            "💙",
            t["know_more"]
        )

        tcols = st.columns(4)

        for col, (ic, tt, ds) in zip(
            tcols,
            t["tiles"]
        ):

            with col:

                st.markdown(
                    f"""
                    <div class="tile">
                        <div class="ic">{ic}</div>
                        <div class="tt">{tt}</div>
                        <div class="ds">{ds}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.write("")

    # ---- ROW 6: LANGUAGES ----

    with st.container(border=True):

        panel_title(
            "🌐",
            t["languages"]
        )

        st.caption(
            t["lang_avail"]
        )

        st.markdown(
            '<span class="chip">English</span><span class="chip">தமிழ்</span>',
            unsafe_allow_html=True,
        )

        st.write("")

    st.warning(
        f"⚕️ {t['disclaimer']}"
    )


# ==================================================
# DATASET ANALYTICS
# ==================================================

elif page == "analytics":

    a, b = st.columns(2)

    with a:

        st.markdown(
            metric_card(
                t["total"],
                f"{TOTAL_IMAGES:,}"
            ),
            unsafe_allow_html=True
        )

    with b:

        st.markdown(
            metric_card(
                t["classes"],
                "5"
            ),
            unsafe_allow_html=True
        )

    st.write("")

    df = pd.DataFrame({
        t["class_col"]: display_names,
        t["images"]: CLASS_COUNTS
    })

    g1, g2 = st.columns(2)

    with g1:

        with st.container(border=True):

            fig = px.bar(
                df,
                x=t["class_col"],
                y=t["images"],
                color=t["class_col"],
                text=t["images"],
            )

            fig.update_layout(
                showlegend=False
            )

            st.plotly_chart(
                style_fig(fig, 380),
                use_container_width=True
            )

    with g2:

        with st.container(border=True):

            fig2 = px.pie(
                df,
                names=t["class_col"],
                values=t["images"],
                hole=0.45
            )

            fig2.update_layout(
                legend=dict(
                    orientation="h",
                    y=-0.1
                )
            )

            st.plotly_chart(
                style_fig(fig2, 380),
                use_container_width=True
            )

    with st.container(border=True):

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        t["dataset_note"]
    )


# ==================================================
# MODEL PERFORMANCE
# ==================================================

elif page == "performance":

    c1, c2 = st.columns([1, 1.4])

    with c1:

        with st.container(border=True):

            panel_title(
                "📈",
                t["model_perf"]
            )

            st.markdown(
                f"""
                <div class="ring" style="background:conic-gradient(#22d3ee {ACCURACY}%, #16304f 0);">
                    <div class="ring-in">
                        <div class="ring-val">{ACCURACY}%</div>
                        <div class="ring-lab">{t['accuracy']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with c2:

        with st.container(border=True):

            panel_title(
                "⚙️",
                t["config"]
            )

            info = pd.DataFrame({
                "Parameter": [
                    "Model",
                    "Dataset",
                    "Input Shape",
                    "Output Classes",
                    "Task",
                    "Framework",
                    "Macro F1",
                    "Quadratic Weighted Kappa",
                    "Prediction Method"
                ],

                "Value": [
                    "ViT-B16",
                    "EyePACS",
                    "256 x 256 x 3",
                    "5",
                    "Image Classification",
                    "PyTorch",
                    "0.789",
                    "0.8309",
                    "Original + Horizontal Flip TTA"
                ],
            })

            st.dataframe(
                info,
                use_container_width=True,
                hide_index=True
            )

    st.write("")

    with st.container(border=True):

        panel_title(
            "📋",
            "Class-wise Evaluation"
        )

        class_metrics = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Precision": [0.91, 0.51, 0.85, 0.86, 0.91],
            "Recall": [0.95, 0.44, 0.73, 0.83, 0.93],
            "F1-Score": [0.93, 0.47, 0.78, 0.84, 0.92],
            "Support": [3673, 360, 788, 93, 104]
        })

        st.dataframe(
            class_metrics,
            use_container_width=True,
            hide_index=True
        )

        st.warning(
            "Mild DR recall is 44%. Some Mild DR cases may be missed by the model. "
            "Clinical decisions must always be made by an ophthalmologist."
        )

    st.info(
        t["perf_note"]
    )


# ==================================================
# ABOUT DR
# ==================================================

elif page == "about":

    for key, label in [
        ("what", t["what"]),
        ("causes", t["causes"]),
        ("symptoms", t["symptoms"]),
        ("prev", t["prev"])
    ]:

        with st.container(border=True):

            panel_title(
                "🔹",
                label
            )

            st.markdown(
                f'<div class="info-text">{t["about_tabs"][key]}</div>',
                unsafe_allow_html=True
            )

    st.warning(
        f"⚕️ {t['disclaimer']}"
    )


# ==================================================
# PREVENTION & CARE
# ==================================================

elif page == "prevention":

    with st.container(border=True):

        panel_title(
            "🛡️",
            t["prevention"]
        )

        items = "".join(
            f'<div class="grade-row"><div class="grade-text">✔️ {x}</div></div>'
            for x in t["prevention_list"]
        )

        st.markdown(
            items,
            unsafe_allow_html=True
        )

    st.warning(
        f"⚕️ {t['disclaimer']}"
    )


# ==================================================
# CONTACT
# ==================================================

elif page == "contact":

    with st.container(border=True):

        panel_title(
            "✉️",
            t["contact"]
        )

        st.markdown(
            f'<div class="info-text">{t["contact_text"]}</div>',
            unsafe_allow_html=True
        )

        st.write("")

        st.markdown(
            """
            <div class="stat-row">
                <span class="k">Email</span>
                <span class="v">your-email@example.com</span>
            </div>

            <div class="stat-row">
                <span class="k">College</span>
                <span class="v">Your College Name</span>
            </div>

            <div class="stat-row">
                <span class="k">Team</span>
                <span class="v">Your Team Name</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    f"""
    <div class="footer">
        {t['footer']}<br>{t['footer2']}
    </div>
    """,
    unsafe_allow_html=True,
)