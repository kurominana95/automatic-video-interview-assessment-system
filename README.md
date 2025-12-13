# automatic-video-interview-assessment-system
Automatic Video Interview Assessment System is a Streamlit-based web app that evaluates video interviews using speech-to-text. It uses OpenAI Whisper to extract and transcribe audio from interview videos, enabling automated, scalable, and objective interview response analysis.

# Automatic Video Interview Assessment System

This repository contains a **Streamlit-based application** for automatic assessment of video interview responses. The system uses **OpenAI Whisper** for speech-to-text transcription and a **rule-based rubric** to score candidate answers for predefined technical interview questions.

The project is designed for educational, research, and prototyping purposes, particularly in the context of automated interview analysis.

---

## Project Overview

The application performs the following steps:

1. Accepts video uploads for each interview question
2. Extracts and converts audio to WAV format
3. Transcribes speech using Whisper
4. Estimates transcription confidence
5. Scores responses using predefined rubrics and keyword heuristics
6. Produces a structured JSON output

---

## Project Structure

```
.
├── app3.py               # Main Streamlit application (upload video file)
├── app2.py               # Alternative Streamlit application (input via Google Drive link)
├── audio/                # Temporary audio files (created at runtime)
├── requirements.txt      # Python dependencies
├── packages.txt          # System-level dependencies (FFmpeg)
├── README.md             # Project documentation
```

---

## System Requirements

### Software

- Python 3.9–3.11 (recommended)
- FFmpeg (system-level installation required)
- CUDA-compatible GPU (optional; CPU execution supported)

### Hardware

- Minimum RAM: 8 GB
- Disk space: at least 1 GB free

---

## Dependencies

Core dependencies include:

- streamlit
- openai-whisper
- torch
- ffmpeg-python

Example `requirements.txt`:

```
streamlit
openai-whisper
torch
ffmpeg-python
```

Note: FFmpeg is not installed via pip and must be available in the system PATH.

---

## Installation and Setup

### 1. Clone the Repository

```
git clone https://github.com/your-username/automatic-video-interview-assessment.git
cd automatic-video-interview-assessment
```

### 2. Create and Activate a Virtual Environment (Recommended)

```
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Python Dependencies

```
pip install -r requirements.txt
```

### 4. Install FFmpeg

- Windows: Download from https://ffmpeg.org and add to PATH
- macOS:
  ```
  brew install ffmpeg
  ```
- Linux:
  ```
  sudo apt install ffmpeg
  ```

Verify installation:

```
ffmpeg -version
```

---

## Running the Application

### Local Execution

```
streamlit run app3.py
```

The application will be available at http://localhost:8501 by default.

### Deployed Version

The file `app3.py` has been deployed using Streamlit Cloud and is publicly accessible at:

https://avias-app.streamlit.app/

This deployed version corresponds to the video upload workflow (file-based input).

---

## Usage Instructions

1. Upload one video file for each interview question (supported formats: MP4, MOV, WEBM)
2. Maximum file size: 40 MB per video
3. Click the "Proses semua video" button
4. Wait until processing is completed
5. Download the generated RESULT.json file

---

## Output Format

The application produces a JSON file with the following structure:

```json
{
  "videos": [
    {
      "question_id": 1,
      "transcript": "...",
      "confidence": 0.87,
      "score": 3,
      "reason": "Specific Challenge with Basic Solution",
      "evidence": ["challenge", "problem"],
      "time_sec": 5.42
    }
  ],
  "reviewChecklistResult": {
    "interview": {
      "scores": [3, 4, 2, 3, 4]
    }
  },
  "total_score": 16,
  "total_time_sec": 28.31
}
```

---

## Scoring Methodology

- Transcribed text is normalized (lowercasing and punctuation removal)
- Each question is evaluated using an independent rubric
- Keyword matching is used to assign score levels (0–4)
- Word count thresholds are applied as fallback heuristics

This scoring approach is deterministic and rule-based, intended as a baseline evaluation method rather than a semantic understanding model.

---

## Reproducibility Notes

To reproduce results consistently:

- Use the same Whisper model variant (`base`)
- Limit Torch execution threads to one (`torch.set_num_threads(1)`)
- Maintain a fixed audio sampling rate (16 kHz, mono)
- Avoid modifying rubric definitions or keyword lists

---

## Known Limitations

- Scoring relies on keyword heuristics and does not capture semantic nuance
- Only English-language interviews are supported
- Long or low-quality videos may increase processing time or reduce transcription accuracy

---

## License

This project is provided for educational and research use. Users are free to adapt and extend the code for non-commercial or academic purposes.

---

## Potential Extensions

- Semantic scoring using sentence embeddings or large language models
- Multi-language transcription support
- Rubric management via an administrative interface
- Persistent storage using a database backend

