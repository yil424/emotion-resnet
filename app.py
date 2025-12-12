import os
import time
from io import BytesIO
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import requests

from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen3-8B:nscale")

hf_client = None
if HF_TOKEN:
    try:
        hf_client = InferenceClient(
            model=HF_MODEL,
            token=HF_TOKEN,
            timeout=60,
        )
    except Exception as e:
        print(f"[WARN] Failed to init HF InferenceClient: {e}")
else:
    print("[WARN] HUGGINGFACEHUB_API_TOKEN is empty, will use template fallback.")

st.set_page_config(
    page_title="Emotion-Aware Chat Demo",
    layout="centered",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: transparent;
    }

    body {
        margin: 0;
        padding: 0;
        overflow-x: hidden;
        background: radial-gradient(circle at 0% 0%, #ff9a9e 0, #fad0c4 25%, #fbc2eb 50%, #a6c1ee 75%, #b9fffd 100%) !important;
        background-size: 200% 200% !important;
        animation: gradientBG 18s ease infinite !important;
    }

    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    #bg-canvas {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: 0;
        pointer-events: none;
    }

    .big-title {
        font-size: 46px;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        text-shadow: 0 4px 12px rgba(0,0,0,0.35);
        margin-top: 8vh;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #fefefe;
        text-align: center;
        margin-bottom: 40px;
    }
    .upload-label {
        font-size: 18px;
        font-weight: 600;
        color: #ffffff;
        text-align: center;
        margin-bottom: 10px;
    }
    .emotion-text {
        font-size: 32px;
        font-weight: 800;
        margin-top: 16px;
        margin-bottom: 8px;
        color: #ffffff;
        text-shadow: 0 4px 10px rgba(0,0,0,0.3);
        text-align: center;
    }
    .response-text {
        font-size: 22px;
        font-weight: 500;
        color: #ffffff;
        margin-top: 12px;
        text-shadow: 0 3px 8px rgba(0,0,0,0.35);
        text-align: center;
    }

    [data-testid="stFileUploader"] section {
        background: rgba(255, 255, 255, 0.22) !important;
        border-radius: 999px !important;
        border: 1px solid rgba(255, 255, 255, 0.9) !important;
        padding-top: 0.7rem !important;
        padding-bottom: 0.7rem !important;
    }
    [data-testid="stFileUploader"] section > div {
        color: #ffffff !important;
        text-align: center !important;
    }
    [data-testid="stFileUploader"] label {
        width: 100%;
    }
    [data-testid="stFileUploaderFileName"] {
        color: #ffffff !important;
    }

    [data-testid="stFileUploader"] button {
        background: rgba(255, 255, 255, 0.18) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.9) !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background: rgba(255, 255, 255, 0.32) !important;
    }

    .preview-wrapper img {
        border-radius: 24px;
        box-shadow: 0 18px 45px rgba(0,0,0,0.45);
    }

    .footer-text {
        margin-top: 24px;
        font-size: 13px;
        color: rgba(255,255,255,0.9);
        text-align: center;
    }
    div.stButton > button {
        background: rgba(255, 255, 255, 0.18) !important;
        color: #ffffff !important;
        border-radius: 999px !important;
        border: 1px solid rgba(255, 255, 255, 0.9) !important;
        font-weight: 600 !important;
        padding: 0.35rem 1.6rem !important;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    div.stButton > button:hover {
        background: rgba(255, 255, 255, 0.32) !important;
        box-shadow: 0 14px 32px rgba(0, 0, 0, 0.45);
    }
    </style>

    <canvas id="bg-canvas"></canvas>

    <script>
    const canvas = window.canvas || document.getElementById('bg-canvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        let particles = [];
        let w, h;

        function resize() {
            w = canvas.width = window.innerWidth;
            h = canvas.height = window.innerHeight;
        }
        window.addEventListener('resize', resize);
        resize();

        function initParticles(num = 120) {
            particles = [];
            for (let i = 0; i < num; i++) {
                particles.push({
                    x: Math.random() * w,
                    y: Math.random() * h,
                    vx: (Math.random() - 0.5) * 0.35,
                    vy: (Math.random() - 0.5) * 0.35,
                    r: 2.0 + Math.random() * 3.0,   
                    alpha: 0.4 + Math.random() * 0.4 
                });
            }
        }

        function draw() {
            ctx.clearRect(0, 0, w, h);
            for (const p of particles) {
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = "rgba(255, 255, 255," + p.alpha + ")";
                ctx.fill();

                p.x += p.vx;
                p.y += p.vy;

                if (p.x < -10) p.x = w + 10;
                if (p.x > w + 10) p.x = -10;
                if (p.y < -10) p.y = h + 10;
                if (p.y > h + 10) p.y = -10;
            }
            requestAnimationFrame(draw);
        }

        initParticles(120);
        draw();
    }
    </script>
    """,
    unsafe_allow_html=True,
)

if "show_result" not in st.session_state:
    st.session_state.show_result = False
if "result" not in st.session_state:
    st.session_state.result = None


def reset_to_upload():
    st.session_state.show_result = False
    st.session_state.result = None
    if "uploader_key" in st.session_state:
        st.session_state.uploader_key += 1
    else:
        st.session_state.uploader_key = 1


EMOTION_LABELS = ["Angry", "Fear", "Happy", "Sad", "Surprise"]


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)      # [B,1,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)    # [B,1,H,W]
        x_cat = torch.cat([avg_out, max_out], dim=1)      # [B,2,H,W]
        attn = self.sigmoid(self.conv(x_cat))             # [B,1,H,W]
        return x * attn

class ResNet18SE_SA(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = False):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.se = SEBlock(channels=512, reduction=16)
        self.sa = SpatialAttention(kernel_size=7)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.se(x)
        x = self.sa(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


@st.cache_resource
def load_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18SE_SA(num_classes=len(EMOTION_LABELS), pretrained=False).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model.load_state_dict(state_dict)
    model.eval()
    return model, device



transform_infer = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_emotion(model, device, pil_img: Image.Image):
    img = pil_img.convert("RGB")
    tensor = transform_infer(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return EMOTION_LABELS[idx], float(probs[idx])

def inject_emotion_background(emotion: str):
    if emotion == "Angry":
        bg = "radial-gradient(circle at 0% 0%, #ff5858 0, #f857a6 25%, #ff9966 60%, #ff5e62 100%)"
    elif emotion == "Fear":
        bg = "radial-gradient(circle at 0% 0%, #141e30 0, #243b55 40%, #4b79a1 80%, #283e51 100%)"
    elif emotion == "Happy":
        bg = "radial-gradient(circle at 0% 0%, #ffe259 0, #ffa751 30%, #fddb92 60%, #a1ffce 100%)"
    elif emotion == "Sad":
        bg = "radial-gradient(circle at 0% 0%, #2c3e50 0, #4ca1af 40%, #6dd5fa 80%, #2980b9 100%)"
    elif emotion == "Surprise":
        bg = "radial-gradient(circle at 0% 0%, #f5576c 0, #f093fb 25%, #fbd786 60%, #c6ffdd 100%)"
    else:
        # default
        bg = "radial-gradient(circle at 0% 0%, #ff9a9e 0, #fad0c4 25%, #fbc2eb 50%, #a6c1ee 75%, #b9fffd 100%)"

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: {bg};
            background-size: 200% 200%;
            animation: gradientBG 18s ease infinite;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def generate_emotion_response_llm(emotion: str) -> str:
    system_msg = (
        "You are a friendly assistant that reacts to a user's facial emotion. "
        "Be concise, empathetic, and natural. Do NOT mention that you were given an emotion label."
    )
    user_msg = (
        f"The detected emotion is: {emotion}.\n\n"
        "Write a short message directly to the user. "
        "If they seem sad or worried, be warm and encouraging and include a tiny light-hearted joke. "
        "If they seem happy or surprised, be playful and positive. "
        "Reply in at most 8 sentences."
    )

    if hf_client is None:
        print("[WARN] HF client is not available, using template response.")
        return generate_emotion_response_template(emotion)

    try:
        output = hf_client.chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=128,
            temperature=0.8,
        )
        text = output.choices[0].message.content.strip()
        if not text:
            raise RuntimeError("Empty response from HF Inference")
        return text
    except Exception as e:
        print(f"[WARN] HF LLM call failed, fallback to template. Error: {e}")
        return generate_emotion_response_template(emotion)


def generate_emotion_response_template(emotion: str) -> str:
    if emotion == "Sad":
        return (
            "It looks like you might be a bit down. "
            "Take a deep breath 💙 You’ve already survived 100% of your worst days so far. "
            "How about a tiny break and a silly meme after this?"
        )
    elif emotion == "Happy":
        return (
            "You look genuinely happy — that smile is absolutely contagious 😆 "
            "I hope today stays this bright for you!"
        )
    elif emotion == "Angry":
        return (
            "I can feel some frustration there 😡 "
            "Totally okay — everyone has those moments. "
            "Maybe grab a glass of water, take 3 slow breaths, and mentally punch a pillow, not a person."
        )
    elif emotion == "Fear":
        return (
            "You seem a bit worried or tense. "
            "Whatever it is, you don’t have to face it all at once 💫 "
            "Small steps still count as progress — and you’re not alone in feeling this way."
        )
    elif emotion == "Surprise":
        return (
            "That surprised face is priceless 🤯 "
            "I hope it’s because of something exciting… "
            "If not, well, plot twists keep life interesting, right?"
        )
    else:
        return (
            "I’m picking up some interesting vibes from your photo 🤔 "
            "Whatever you’re feeling, I hope you treat yourself gently today."
        )

ckpt_path = os.path.join(os.path.dirname(__file__), "resnet18_se_sa_best.pth")

if st.session_state.show_result and st.session_state.result is not None:
    res = st.session_state.result
    emotion = res["emotion"]

    inject_emotion_background(emotion)

    back_col, _ = st.columns([2, 10])
    with back_col:
        if st.button("← Back", use_container_width=True):
            reset_to_upload()
            st.rerun()

    img_bytes = res["image_bytes"]
    conf = res["conf"]
    reply = res["reply"]

    pil_img = Image.open(BytesIO(img_bytes))

    st.markdown('<div class="preview-wrapper">', unsafe_allow_html=True)
    st.image(pil_img, caption="Your photo", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="emotion-text">You look: {emotion.upper()} '
        f'({conf*100:.1f}% confidence)</div>',
        unsafe_allow_html=True,
    )

    placeholder = st.empty()
    displayed = ""
    for ch in reply:
        displayed += ch
        placeholder.markdown(
            f'<div class="response-text">{displayed}</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.02)

    st.markdown(
        '<div class="footer-text">Model: ResNet18-SE+SA (label smoothing trained) • Local LLM: qwen2.5:1.5B-Instruct</div>',
        unsafe_allow_html=True,
    )

else:
    inject_emotion_background("default")

    st.markdown(
        '<div class="big-title">Upload your selfie and I’ll guess your emotion 😄</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Then a friendly mini-assistant will send you a personalized message 💌</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="upload-label">Click to upload a selfie 👇</div>',
        unsafe_allow_html=True,
    )

    uploader_key = st.session_state.get("uploader_key", 0)
    uploaded_file = st.file_uploader(
        "",
        type=["png", "jpg", "jpeg"],
        key=f"uploader_{uploader_key}",
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        pil_img = Image.open(BytesIO(image_bytes))

        if not os.path.exists(ckpt_path):
            st.error("Cannot find resnet18_se_best.pth in the current folder.")
        else:
            model, device = load_model(ckpt_path)

            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Analyzing your photo...")
            progress_bar.progress(30)
            time.sleep(0.25)

            emotion, conf = predict_emotion(model, device, pil_img)
            progress_bar.progress(60)
            status_text.text("Asking a friendly LLM for a reply...")
            time.sleep(0.25)

            reply = generate_emotion_response_llm(emotion)
            progress_bar.progress(100)
            status_text.text("Done! ✨")
            time.sleep(0.4)

            st.session_state.result = {
                "image_bytes": image_bytes,
                "emotion": emotion,
                "conf": conf,
                "reply": reply,
            }
            st.session_state.show_result = True
            st.rerun()




