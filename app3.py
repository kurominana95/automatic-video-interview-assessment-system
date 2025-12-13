import streamlit as st
import os
import re
import ffmpeg
import whisper
import json
import math
import random
import time
import torch
from io import BytesIO

# =========================
# BATASI THREAD (WAJIB)
# =========================
torch.set_num_threads(1)

# =========================
# Pertanyaan
# =========================
QUESTIONS = {
    1: "Describe the challenges you faced during your TensorFlow certification.",
    2: "Explain your experience using transfer learning in TensorFlow.",
    3: "Describe a complex TensorFlow model you built and how you ensured both accuracy and efficiency.",
    4: "Explain how dropout is implemented in TensorFlow and its effects on model performance.",
    5: "Describe the process of building a CNN in TensorFlow for image classification."
}

# =========================
# RUBRIK
# =========================
RUBRIC = { 1: {
        "levels": {
            4: (
                "Comprehensive and Clear Response.\n"
                "Provides a detailed description of specific challenges encountered during the certification.\n"
                "Offers clear explanations of how each challenge was overcome.\n"
                "Demonstrates strong understanding of technical aspects and problem-solving skills.\n"
                "Response is very clear, well-organized, and reflects the learning process."
            ),
            3: (
                "Specific Challenge with Basic Solution.\n"
                "Describes at least one specific challenge related to building machine learning models with TensorFlow.\n"
                "Provides a basic explanation of how the challenge was overcome.\n"
                "Explanation may be brief or lack depth.\n"
                "Shows some understanding but lacks detailed insight."
            ),
            2: "General Challenge Mentioned without Details.",
            1: "Minimal or Vague Response.",
            0: "Unanswered"
        },
        "keywords": {
            4: ["challenge","overcome","solution","error","architecture","certification"],
            3: ["challenge","problem","difficult"],
            2: ["problem"],
            1: []
        }
    },

    2: {
        "levels": {
            4: (
                "Comprehensive and Very Clear Response.\n"
                "Offers a detailed description of personal experience using transfer learning in TensorFlow.\n"
                "Provides specific examples of projects where transfer learning was applied.\n"
                "Demonstrates strong understanding of transfer learning concepts and their practical application.\n"
                "Explains how transfer learning benefited those projects.\n"
                "Response is very clear, well-organized, and reflects deep engagement with the subject."
            ),
            3: (
                "Specific Experience with Basic Explanation.\n"
                "Describes personal experience with transfer learning in TensorFlow.\n"
                "Provides examples of projects where transfer learning was applied.\n"
                "Explains how transfer learning benefited those projects.\n"
                "Explanation may be brief or not fully comprehensive insight."
            ),
            2: (
                "General Response with Limited Details.\n"
                "Mentions transfer learning or TensorFlow in general terms.\n"
                "Provides minimal details about personal experience.\n"
                "Does not clearly explain how transfer learning benefited the projects.\n"
                "Shows basic understanding but lucks depth and specificity."
            ),
            1: "Minimal or Vague Response.",
            0: "Unanswered"
        },
        "keywords": {
            4: ["transfer learning","mobilenet","vgg","efficient","pretrained"],
            3: ["transfer learning"],
            2: ["tensorflow"],
            1: []
        }
    },

    3: {
        "levels": {
            4: (
                "Comprehensive and Very Clear Response.\n"
                "Offers a detailed description of a complex TensorFlow model built.\n"
                "Provides specific details about the model’s architecture, features, and purpose.\n"
                "Clearly explains the steps taken to ensure both accuracy and efficiency, such as preprocessing, model optimization techniques, regularization methods, and performance tuning.\n"
                "Demonstrates strong understanding of TensorFlow and machine learning concepts.\n"
                "Response is very clear, well-organized, and reflects deep engagement with the subject."
            ),
            3: (
                "Specific Model with Basic Explanation.\n"
                "Describes a complex TensorFlow model they have built.\n"
                "Provides some details about the model’s architecture or purpose.\n"
                "Explains steps taken to ensure accuracy and efficiency, but may lack depth.\n"
                "Explaination may be brief or not fully comprehensive."
            ),
            2: (
                "General Response with Limited Details.\n"
                "Mentions building a TensorFlow model in general terms.\n"
                "Provides minimal details about the model's complexity.\n"
                "Does not clearly explain the steps taken to ensure accuracy and efficiency.\n"
                "Shows basic understanding but lucks depth and specificity."
            ),
            1: "Minimal or Vague Response.",
            0: "Unanswered"
        },
        "keywords": {
            4: ["accuracy","optimization","regularization","preprocessing","tuning","callback","early stopping"],
            3: ["model","accuracy","optimization"],
            2: [],
            1: []
        }
    },

    4: {
        "levels": {
            4: (
                "Comprehensive and Very Clear Response.\n"
                "Provides a detailed explanation of how to implement dropout in a TensorFlow model, including code examples or specific functions (e.g., using tf.keras.layers.Dropout).\n"
                "Clearly explains the effect of dropout on training, such as how it helps prevent overfitting by randomly deactivating neurons during training.\n"
                "Discusses the impact on model performance, generalization, and possibly mentions considerations like dropout rates.\n"
                "Demonstrates strong understanding of TensorFlow and machine learning concepts.\n"
                "Response is very clear, well-organized, and reflects deep engagement with the subject."
            ),
            3: (
                "Specific Explanation with Basic Understanding.\n"
                "Explains how to implement dropout in a TensorFlow model with some specifics (e.g., mentions using Dropout layer).\n"
                "Describes the general effect of dropout on training, such as preventing overfitting.\n"
                "May omit some details about implementation or effects.\n"
                "Demonstrates a reasonable understanding but lacks comprehensive insight."
            ),
            2: "General Response with Limited Details.",
            1: "Minimal or Vague Response.",
            0: "Unanswered"
        },
        "keywords": {
            4: ["dropout", "rate", "overfitting", "random", "neuron", "impact","deactivating", "turning", "off"],
            3: ["dropout","prevent overfitting"],
            2: ["dropout"],
            1: []
        }
    },

    5: {
        "levels": {
            4: (
                "Comprehensive and Very Clear Response.\n"
                "Provides a detailed, step-by-step description of building a CNN in TensorFlow for image classification.\n"
                "Covers all key components, including data loading and preprocessing, defining the CNN layers (convolutional, pooling, activation functions), compiling the model with appropriate loss function and optimizer, training the model, and evaluating performance.\n"
                "May include code examples or specific TensorFlow functions used.\n"
                "Demonstrates strong understanding of CNNs and TensorFlow.\n"
                "Response is very clear, well-organized, and reflects deep engagement with the subject."
            ),
            3: (
                "Specific Explanation with Basic Understanding.\n"
                "Describes the process of building a CNN in TensorFlow with some specifics.\n"
                "Includes key steps such as data preprocessing, defining the CNN architecture, compiling the model, and training.\n"
                "May lack comprehensive detail or omit some important aspects.\n"
                "Demonstrates reasonable understanding but may not fully elaborate."
            ),
            2: "General Response with Limited Details.",
            1: "Minimal or Vague Response.",
            0: "Unanswered"
        },
        "keywords": {
            4: ["cnn","convolutional","pooling","activation","compile","optimizer","training"],
            3: ["cnn","architecture"],
            2: [],
            1: []
        }
    }}


# =========================
# Folder audio
# =========================
AUD_DIR = "audio"
os.makedirs(AUD_DIR, exist_ok=True)

# =========================
# Convert ke WAV
# =========================
def convert_to_wav(input_bytes, wav_path):
    with open("temp_video_file", "wb") as f:
        f.write(input_bytes.read())

    (
        ffmpeg
        .input("temp_video_file")
        .output(wav_path, ac=1, ar="16000")
        .overwrite_output()
        .run(quiet=True)
    )
    os.remove("temp_video_file")

# =========================
# Confidence helpers
# =========================
def seg_confidence_from_avglogprob(avg_logprob):
    try:
        p = math.exp(avg_logprob)
    except:
        p = 0.0
    return float(max(0.0, min(1.0, p)))

def overall_confidence_from_segments(segments):
    if not segments:
        return 0.0
    weights = [(s['end'] - s['start']) for s in segments]
    confs = [seg_confidence_from_avglogprob(s.get('avg_logprob', -10)) for s in segments]
    total_w = sum(weights)
    return float(sum(w * c for w, c in zip(weights, confs)) / total_w)

# =========================
# Normalisasi & scoring
# =========================
def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()

def pick_one_reason(reason_text: str) -> str:
    lines = [line.strip() for line in reason_text.split("\n") if line.strip()]
    return random.choice(lines) if lines else reason_text

def score_text_for_question(qid: int, text: str):
    norm = normalize_text(text)
    info = RUBRIC.get(qid)

    if not norm or len(norm) < 5:
        return 0, pick_one_reason(info["levels"][0]), []

    kw4 = info["keywords"].get(4, [])
    hits4 = [kw for kw in kw4 if kw in norm]
    if len(hits4) >= 4:
        return 4, pick_one_reason(info["levels"][4]), hits4

    kw3 = info["keywords"].get(3, [])
    hits3 = [kw for kw in kw3 if kw in norm]
    if hits3:
        return 3, pick_one_reason(info["levels"][3]), hits3

    kw2 = info["keywords"].get(2, [])
    hits2 = [kw for kw in kw2 if kw in norm]
    if hits2:
        return 2, pick_one_reason(info["levels"][2]), hits2

    wc = len(norm.split())
    if wc >= 20:
        return 2, pick_one_reason(info["levels"][2]), []
    if wc >= 5:
        return 1, pick_one_reason(info["levels"][1]), []

    return 0, pick_one_reason(info["levels"][0]), []

# =========================
# Load Whisper
# =========================
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

# =========================
# Streamlit UI
# =========================
st.title("Automatic Video Interview Assessment System")

if "processing" not in st.session_state:
    st.session_state.processing = False

uploaded_files = {}

MAX_MB = 40

for qid, qtext in QUESTIONS.items():
    f = st.file_uploader(
        f"Upload video untuk pertanyaan {qid}: {qtext}",
        type=["mp4", "webm", "mov"],
        key=f"upload_{qid}"
    )

    if f and f.size > MAX_MB * 1024 * 1024:
        st.error(f"Ukuran maksimal video {MAX_MB} MB")
        st.stop()

    uploaded_files[qid] = f

# =========================
# PROSES
# =========================
if st.button("Proses semua video", disabled=st.session_state.processing):
    st.session_state.processing = True
    model = load_whisper()

    final_payload = {
        "videos": [],
        "reviewChecklistResult": {"interview": {"scores": []}},
        "total_score": 0
    }

    start_all = time.time()

    for qid, f in uploaded_files.items():
        if f is None:
            continue

        wav_path = os.path.join(AUD_DIR, f"audio_{qid}.wav")
        convert_to_wav(BytesIO(f.getbuffer()), wav_path)

        t0 = time.time()
        res = model.transcribe(wav_path, language="en", verbose=False)
        t1 = time.time()

        os.remove(wav_path)  # 🔥 KRUSIAL

        transcript = " ".join([s["text"].strip() for s in res["segments"]])
        conf = overall_confidence_from_segments(res["segments"])
        score, reason, evidence = score_text_for_question(qid, transcript)

        final_payload["videos"].append({
            "question_id": qid,
            "transcript": transcript,
            "confidence": conf,
            "score": score,
            "reason": reason,
            "evidence": evidence,
            "time_sec": round(t1 - t0, 2)
        })

        final_payload["reviewChecklistResult"]["interview"]["scores"].append(score)
        final_payload["total_score"] += score

    final_payload["total_time_sec"] = round(time.time() - start_all, 2)
    st.session_state.processing = False

    st.success("Proses selesai")
    st.download_button(
        "Download JSON",
        json.dumps(final_payload, indent=2),
        "RESULT.json",
        "application/json"
    )
