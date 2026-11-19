# Uji video asli pada CPU Streamlit Community Cloud
# Alur aplikasi dimulai dari upload, pemanggilan pipeline, penyimpanan hasil sesi,
# lalu penyajian grafik, contoh titik bibir, dan arsip CSV di browser.

# --- 1. Lingkungan komputasi dan pustaka pendukung ---
import os

# Batas thread ditetapkan sebelum framework diimpor.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

from dataclasses import replace
import base64
import hashlib
import html
import io
from pathlib import Path
import tempfile
import threading
import time
import zipfile

# Pengukuran puncak RAM memakai modul bawaan Linux. Windows melewati pengukuran ini.
try:
    import resource
except ImportError:
    resource = None

import streamlit as st
import streamlit.components.v1 as components

# --- 2. Tampilan awal dan input pengguna ---
# Nada target diteruskan ke CREPE sebagai pembanding pitch rekaman.
st.set_page_config(page_title="VocalAlign — Uji Video", page_icon="🎵", layout="wide")
st.title("VocalAlign")
st.write("Uji analisis video dengan CREPE dan YOLO pada CPU cloud.")
st.caption("Gunakan video yang sama dengan pengujian Colab, sekitar 5–10 detik.")

uploaded = st.file_uploader("Upload video rekaman", type=["mov", "mp4", "webm"])
target = st.text_input("Nada target beserta oktaf", value="D#4").strip()


# --- 3. Pemuatan model yang dipakai bersama antar-sesi ---
# Cache mempertahankan model di memori saat Streamlit menjalankan ulang halaman.
# Lock yang dikembalikan mengatur giliran analisis untuk model yang sama.
@st.cache_resource
def load_models():
    import torch
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    torch.set_num_threads(1)
    lip_path = Path(__file__).parent / "lip-pose.pt"
    if not lip_path.is_file():
        raise FileNotFoundError("Upload lip-pose.pt ke folder yang sama dengan app.py.")
    # Detektor wajah diambil dari Hugging Face, bobot bibir berasal dari training tim.
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


# --- 4. Pengendali seluruh tahapan analisis untuk satu video ---
# Fungsi pengukuran berada di pipeline.py. Fungsi ini mengatur urutan eksekusinya
# dan mengembalikan data yang sudah siap ditampilkan oleh Streamlit.
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
        # Setiap analisis memakai folder sendiri agar file antar-upload terpisah.
        # Folder dibersihkan ketika blok selesai, sementara hasil disalin ke memori.
        folder = Path(folder)
        video = folder / ("input" + Path(upload.name).suffix.lower())
        video.write_bytes(upload.getbuffer())
        cfg = replace(p.CFG, WORK_DIR=str(folder))

        # Tahap 1 menghasilkan audio mono 16 kHz dan video dengan 30 fps konstan.
        progress.write("Memisahkan audio dan menormalkan video ke 30 fps...")
        meta = p.prepare_input(str(video), cfg=cfg)
        meta["target_note"] = note

        # Tahap 2 mengukur pitch, memilih oktaf acuan, dan memberi status nada.
        progress.write("Mengukur pitch dengan CREPE small...")
        audio, sr = p.load_audio(meta)
        pitch, info = p.extract_pitch_vs_target(audio, sr, note, cfg=cfg)
        verdict = p.evaluate_note(pitch, info, cfg=cfg)

        # Tahap 3 dan 4 mencari wajah, kemudian empat titik bibir untuk menghitung MAR.
        progress.write("Mendeteksi wajah di seluruh frame...")
        faces = p.run_face_detection(meta, face_model, cfg=cfg)
        progress.write("Mengukur titik bibir dan MAR...")
        lips = p.run_lip_tracking(meta, faces, lip_model, cfg=cfg)
        # Tahap 5 memasangkan MAR dengan sampel pitch berdasarkan timestamp.
        synced = p.sync_pitch_mar(pitch, lips, fps=meta["fps"])

        # Ambil hingga empat frame dengan MAR valid yang tersebar sepanjang video.
        # Titik digambar pada koordinat frame asli untuk pemeriksaan visual.
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

        # Simpan keempat tabel CSV dalam satu ZIP di memori sebelum folder dihapus.
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w", zipfile.ZIP_DEFLATED) as archive:
            for filename, df in (
                ("pitch.csv", pitch), ("faces.csv", faces),
                ("mar.csv", lips), ("pitch_mar_sync.csv", synced),
            ):
                archive.writestr(filename, df.to_csv(index=False))

        # Ringkasan menghitung cakupan deteksi dan sampel yang bisa dinilai.
        # Waktu mencakup muat model dan antrean. Puncak RAM berlaku sejak startup proses.
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
            "Puncak RAM proses aplikasi sejak startup (MiB)": (
                round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
                if resource is not None else "Tidak tersedia di Windows"
            ),
            "TensorFlow": tf.__version__,
        }
        return {"summary": summary, "verdict": verdict, "synced": synced,
                "previews": previews, "archive": archive_bytes.getvalue()}


# --- 5. Identitas input dan penyimpanan hasil selama sesi browser ---
# Hash isi video dan nama nada dipakai untuk mengenali perubahan input.
# Hasil lama disembunyikan jika video atau target berubah.
input_key = None
if uploaded is not None:
    input_key = (hashlib.sha256(uploaded.getbuffer()).hexdigest(), target)
if st.session_state.get("result_key") != input_key:
    st.session_state.pop("video_result", None)

# --- 6. Tombol eksekusi, progres, dan pelaporan kegagalan ---
# Analisis berjalan saat tombol ditekan. Hasil disimpan agar interaksi halaman
# berikutnya bisa menampilkan hasil dari sesi yang sama.
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

# --- 7. Penyajian hasil analisis yang tersimpan ---
# Sampel pitch bertanda stable ditampilkan di grafik nada. Grafik MAR menggunakan
# timestamp yang sama sehingga pengguna bisa membandingkan keduanya.
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

    # Tampilkan gambar RGB beserta caption yang di-escape sebelum dimasukkan ke HTML.
    if result["previews"]:
        from PIL import Image

        columns = st.columns(len(result["previews"]))
        for column, (image, caption) in zip(columns, result["previews"]):
            # Sertakan gambar di halaman agar tidak memerlukan URL /media terpisah.
            image_buffer = io.BytesIO()
            Image.fromarray(image).save(image_buffer, format="PNG")
            image_base64 = base64.b64encode(image_buffer.getvalue()).decode("ascii")
            safe_caption = html.escape(caption)
            column.markdown(
                f'<figure style="margin:0">'
                f'<img src="data:image/png;base64,{image_base64}" '
                f'alt="{safe_caption}" style="width:100%;height:auto;border-radius:8px">'
                f'<figcaption style="text-align:center;margin-top:8px">'
                f'{safe_caption}</figcaption></figure>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("Belum ada titik bibir dengan MAR valid untuk ditampilkan.")

    # --- 8. Unduhan hasil untuk evaluasi di luar website ---
    # Browser menerima arsip langsung, termasuk keempat CSV hasil pengukuran.
    archive_base64 = base64.b64encode(result["archive"]).decode("ascii")
    components.html(
        f'''<a href="data:application/zip;base64,{archive_base64}"
            download="vocalign_hasil_uji.zip"
            style="display:inline-block;padding:10px 16px;border:1px solid #777;
                   border-radius:8px;text-decoration:none;font:14px sans-serif;
                   color:white;background:#262730">
            Download hasil CSV (.zip)
        </a>''',
        height=65,
    )
