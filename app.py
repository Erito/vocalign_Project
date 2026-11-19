# Uji video asli pada CPU Streamlit Community Cloud
import os

# Batas thread ditetapkan sebelum framework diimpor.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

from dataclasses import replace
import hashlib
import io
from pathlib import Path
import resource
import tempfile
import threading
import time
import zipfile

import streamlit as st

st.set_page_config(page_title="VocalAlign — Uji Video", page_icon="🎵", layout="wide")
st.title("VocalAlign")
st.write("Uji analisis video dengan CREPE dan YOLO pada CPU cloud.")
st.caption("Gunakan video yang sama dengan pengujian Colab, sekitar 5–10 detik.")

uploaded = st.file_uploader("Upload video rekaman", type=["mov", "mp4", "webm"])
target = st.text_input("Nada target beserta oktaf", value="D#4").strip()


@st.cache_resource
def load_models():
    import torch
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    torch.set_num_threads(1)
    lip_path = Path(__file__).parent / "lip-pose.pt"
    if not lip_path.is_file():
        raise FileNotFoundError("Upload lip-pose.pt ke folder yang sama dengan app.py.")
    face_path = hf_hub_download(
        repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt"
    )
    face_model = YOLO(face_path)
    lip_model = YOLO(str(lip_path))
    if lip_model.task != "pose" or lip_model.names != {0: "mouth"}:
        raise ValueError("Bobot bibir harus berupa model pose dengan kelas mouth.")
    face_model.to("cpu")
    lip_model.to("cpu")
    return face_model, lip_model, threading.Lock()


def analyze_video(upload, note, progress):
    import cv2
    import numpy as np
    import tensorflow as tf
    import pipeline as p

    # Validasi nada sebelum pekerjaan model dimulai.
    p.note_to_hz_safe(note)
    started = time.perf_counter()
    progress.write("Memuat bobot wajah dan bibir...")
    face_model, lip_model, lock = load_models()

    # Model cache dipakai bersama, sehingga inferensi video diantrekan.
    progress.write("Menunggu giliran komputasi...")
    with lock, tempfile.TemporaryDirectory(prefix="vocalign_") as folder:
        folder = Path(folder)
        video = folder / ("input" + Path(upload.name).suffix.lower())
        video.write_bytes(upload.getbuffer())
        cfg = replace(p.CFG, WORK_DIR=str(folder))

        progress.write("Memisahkan audio dan menormalkan video ke 30 fps...")
        meta = p.prepare_input(str(video), cfg=cfg)
        meta["target_note"] = note

        progress.write("Mengukur pitch dengan CREPE small...")
        audio, sr = p.load_audio(meta)
        pitch, info = p.extract_pitch_vs_target(audio, sr, note, cfg=cfg)
        verdict = p.evaluate_note(pitch, info, cfg=cfg)

        progress.write("Mendeteksi wajah di seluruh frame...")
        faces = p.run_face_detection(meta, face_model, cfg=cfg)
        progress.write("Mengukur titik bibir dan MAR...")
        lips = p.run_lip_tracking(meta, faces, lip_model, cfg=cfg)
        synced = p.sync_pitch_mar(pitch, lips, fps=meta["fps"])

        progress.write("Menyiapkan hasil dan empat frame contoh...")
        valid_lips = lips.loc[np.isfinite(lips["mar"])]
        previews = []
        if not valid_lips.empty:
            positions = np.unique(np.linspace(0, len(valid_lips) - 1, 4).astype(int))
            colors = [(255, 0, 0), (0, 100, 255), (0, 255, 0), (255, 255, 0)]
            for position in positions:
                row = valid_lips.iloc[position]
                frame = p.get_frame(meta, int(row["frame"]))
                if frame is None:
                    continue
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for point, color in zip(p.LIP_ORDER, colors):
                    xy = (int(row[f"{point}_x"]), int(row[f"{point}_y"]))
                    cv2.circle(image, xy, 3, color, -1)
                previews.append((image, f"{row['t']:.2f} s · MAR {row['mar']:.3f}"))

        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w", zipfile.ZIP_DEFLATED) as archive:
            for filename, df in (
                ("pitch.csv", pitch), ("faces.csv", faces),
                ("mar.csv", lips), ("pitch_mar_sync.csv", synced),
            ):
                archive.writestr(filename, df.to_csv(index=False))

        summary = {
            "Target": note, "Acuan setelah penyesuaian oktaf": info["ref_note"],
            "Durasi video (s)": round(meta["video_duration"], 3),
            "Selisih durasi audio-video (s)": round(abs(meta["video_duration"] - meta["audio_duration"]), 3),
            "Frame video": len(lips), "Wajah terdeteksi": int(faces["detected"].sum()),
            "Bibir terdeteksi": int(lips["lip_detected"].sum()),
            "Frame MAR valid": len(valid_lips), "Sampel CREPE": len(pitch),
            "Sampel stabil": int(pitch["stable"].sum()),
            "Sampel stabil dan tepat": int((pitch["stable"] & pitch["status"].eq("in-tune")).sum()),
            "Sampel sinkron dengan MAR": int(synced["mar_smooth"].notna().sum()),
            "MAR median": round(float(valid_lips["mar"].median()), 3) if not valid_lips.empty else None,
            "Waktu total termasuk muat model dan antrean (s)": round(time.perf_counter() - started, 2),
            "Puncak RAM proses aplikasi sejak startup (MiB)": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "TensorFlow": tf.__version__,
        }
        return {"summary": summary, "verdict": verdict, "synced": synced,
                "previews": previews, "archive": archive_bytes.getvalue()}


# Hasil lama disembunyikan jika video atau target berubah.
input_key = None
if uploaded is not None:
    input_key = (hashlib.sha256(uploaded.getbuffer()).hexdigest(), target)
if st.session_state.get("result_key") != input_key:
    st.session_state.pop("video_result", None)

if st.button("Analisis video", disabled=uploaded is None):
    st.session_state.pop("video_result", None)
    try:
        with st.status("Menjalankan analisis...", expanded=True) as progress:
            result = analyze_video(uploaded, target, progress)
            st.session_state["video_result"] = result
            st.session_state["result_key"] = input_key
            progress.update(label="Analisis selesai", state="complete", expanded=False)
    except Exception as error:
        st.error("Analisis video belum berhasil.")
        st.exception(error)

if "video_result" in st.session_state:
    import numpy as np
    import pandas as pd

    result = st.session_state["video_result"]
    st.subheader("Ringkasan pengujian")
    st.dataframe(pd.DataFrame(
        [(key, str(value)) for key, value in result["summary"].items()],
        columns=["Pengukuran", "Hasil"],
    ), hide_index=True, use_container_width=True)
    st.write(result["verdict"])

    synced = result["synced"]
    st.subheader("Deviasi nada")
    chart = synced.set_index("t")
    st.line_chart(chart["cent"].where(chart["stable"]).rename("cent"))
    st.subheader("Bukaan mulut")
    st.line_chart(chart[["mar_interp", "mar_smooth"]])

    if result["previews"]:
        columns = st.columns(len(result["previews"]))
        for column, (image, caption) in zip(columns, result["previews"]):
            column.image(image, caption=caption, use_container_width=True)
    else:
        st.warning("Belum ada titik bibir dengan MAR valid untuk ditampilkan.")

    st.download_button("Download hasil CSV", data=result["archive"],
                       file_name="vocalign_hasil_uji.zip", mime="application/zip")
