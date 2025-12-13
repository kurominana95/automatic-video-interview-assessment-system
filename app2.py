import streamlit as st
import os
import re
import gdown
import ffmpeg
import whisper
import json
import math
import random
from rubrik import RUBRIC
import time
import torch
# =========================
# Direktori output
# =========================
VID_DIR = "videos"
AUD_DIR = "audio"
os.makedirs(VID_DIR, exist_ok=True)
os.makedirs(AUD_DIR, exist_ok=True)

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

st.title("Automatic Video Interview Assessment System")

# =========================
# Input link Google Drive
# =========================
drive_links = {}
for qid, qtext in QUESTIONS.items():
    st.write(f"**Pertanyaan {qid}: {qtext}**")
    links_input = st.text_area(
        f"Masukkan link Google Drive untuk pertanyaan {qid}:",
        height=100,
        key=f"links_{qid}"
    )
    drive_links[qid] = [line.strip() for line in links_input.splitlines() if line.strip()]

# =========================
# Fungsi ekstrak ID
# =========================
def extract_drive_id(url):
    patterns = [
        r"https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

# =========================
# Fungsi download
# =========================
def download_drive(id_, dest):
    url = f"https://drive.google.com/uc?id={id_}&export=download"
    gdown.download(url, dest, quiet=False)

# =========================
# Convert ke WAV
# =========================
def convert_to_wav(video_path, wav_path):
    (
        ffmpeg
        .input(video_path)
        .output(wav_path, ac=1, ar="16000")
        .overwrite_output()
        .run(quiet=True)
    )

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
# Load Whisper sekali
# =========================
@st.cache_resource
def load_whisper():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("medium").to(device)

# =========================
# Tombol proses semua video
# =========================
if st.button("Proses semua video"):
    model = load_whisper()
    final_payload = {
        "assessorProfile": {
            "id": 47,
            "name": "AutoSystem",
            "photoURL": "auto.png"
        },
        "videos": [],
        "videoCheck": [],
        "reviewChecklistResult": {
            "interview": {
                "minScore": 0,
                "maxScore": 4,
                "scores": []
            }
        },
        "overallNotes": "Interview responses appear consistent."
    }

    total_score = 0
    start_all = time.time()  # waktu mulai seluruh proses

    for qid, links in drive_links.items():
        for idx, link in enumerate(links, start=1):
            file_id = extract_drive_id(link)
            if not file_id:
                st.warning(f"Tidak bisa ekstrak ID dari link: {link}")
                continue

            # Download video
            with st.spinner(f"Mendownload video pertanyaan {qid} ({idx}/{len(links)})..."):
                video_path = os.path.join(VID_DIR, f"interview_question_{qid}_{idx}.webm")
                download_drive(file_id, video_path)

            # Convert ke WAV
            with st.spinner(f"Mengonversi video pertanyaan {qid} ke WAV..."):
                wav_path = os.path.join(AUD_DIR, f"audio_{qid}_{idx}.wav")
                convert_to_wav(video_path, wav_path)

            # Transcribe
            with st.spinner(f"Menjalankan transkripsi video pertanyaan {qid} ({idx}/{len(links)})..."):
                t0 = time.time()
                res = model.transcribe(wav_path, language="en", verbose=False, condition_on_previous_text=False)
                t1 = time.time()
                transcribe_time = t1 - t0

            # Bersihkan segmen
            cleaned_segments = []
            prev = ""
            for seg in res["segments"]:
                segt = seg["text"].strip()
                if segt != prev:
                    cleaned_segments.append(segt)
                    prev = segt
            transcript = " ".join(" ".join(cleaned_segments).split())
            conf = overall_confidence_from_segments(res["segments"])

            # Score rubrik
            score, reason, evidence = score_text_for_question(qid, transcript)
            total_score += score

            # Tambahkan ke final_payload["videos"]
            final_payload["videos"].append({
                "question_id": qid,
                "question": QUESTIONS[qid],
                "video_file": video_path,
                "transcript": transcript,
                "confidence": conf,
                "score": score,
                "reason": reason,
                "evidence": evidence,
                "transcription_time_sec": round(transcribe_time, 2)
            })

            # Tambahkan ke videoCheck
            final_payload["videoCheck"].append({
                "file_name": os.path.basename(video_path),
                "isExist": 1 if os.path.exists(video_path) else 0,
                "source_link": link
            })

            # Tambahkan skor ke checklist
            final_payload["reviewChecklistResult"]["interview"]["scores"].append(score)

    # Tambahkan summary dan keputusan
    end_all = time.time()
    final_payload["total_score"] = total_score
    final_payload["total_process_time_sec"] = round(end_all - start_all, 2)
    final_payload["decision"] = "Need Human"
    final_payload["scoresOverview"] = {
        "project": 100,
        "interview": total_score,
        "total": 94.3  # bisa disesuaikan jika ingin total dihitung
    }

    # Simpan JSON
    json_path = os.path.join("output", "RESULT.json")
    os.makedirs("output", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    st.success("Semua video diproses dan JSON siap diunduh!")

    # Tombol download JSON
    st.download_button(
        label="Download JSON",
        data=json.dumps(final_payload, indent=2),
        file_name="RESULT.json",
        mime="application/json"
    )
