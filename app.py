# Uji pemuatan dan inferensi model bibir di CPU
from pathlib import Path
import platform
import shutil

import streamlit as st

st.set_page_config(page_title="VocalAlign", page_icon="🎵")
st.title("VocalAlign")
st.write(f"Versi Python: {platform.python_version()}")

if shutil.which("ffmpeg"):
    st.success("FFmpeg tersedia.")

model_path = Path(__file__).parent / "lip-pose.pt"

if not model_path.is_file():
    st.error("Upload lip-pose.pt ke folder yang sama dengan app.py.")
    st.stop()

if st.button("Uji model bibir"):
    try:
        with st.spinner("Memuat model dan menguji inferensi CPU..."):
            import numpy as np
            import torch
            import ultralytics
            from ultralytics import YOLO

            torch.set_num_threads(1)

            model = YOLO(str(model_path))
            model.to("cpu")

            if model.task != "pose" or model.names != {0: "mouth"}:
                raise ValueError(
                    f"Model yang dimuat memiliki task {model.task} "
                    f"dan kelas {model.names}."
                )

            # Gambar kosong dipakai untuk memastikan proses prediksi berjalan.
            gambar_uji = np.zeros((256, 256, 3), dtype=np.uint8)
            model.predict(
                gambar_uji,
                imgsz=256,
                device="cpu",
                verbose=False,
            )

        st.success("Model bibir berhasil dimuat dan inferensi CPU berjalan.")
        st.write("Kelas model:", model.names)
        st.write("PyTorch:", torch.__version__)
        st.write("Ultralytics:", ultralytics.__version__)

    except Exception as error:
        st.error("Uji model belum berhasil.")
        st.exception(error)



# Uji CREPE menggunakan nada sintetis A4 pada CPU
st.divider()
st.subheader("Uji pitch CREPE")

if st.button("Uji CREPE — A4 440 Hz"):
    try:
        with st.spinner("Memuat CREPE dan menganalisis nada uji..."):
            import os
            import time

            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
            os.environ["TF_NUM_INTEROP_THREADS"] = "1"

            import numpy as np
            import tensorflow as tf
            import crepe

            sr = 16000
            frekuensi_target = 440.0

            # Satu detik nada dengan frekuensi yang diketahui.
            t = np.arange(sr) / sr
            audio = (
                0.3 * np.sin(2 * np.pi * frekuensi_target * t)
            ).astype(np.float32)

            mulai = time.perf_counter()

            waktu, frekuensi, confidence, _ = crepe.predict(
                audio,
                sr,
                model_capacity="small",
                step_size=10,
                viterbi=True,
                verbose=0,
            )

            durasi_proses = time.perf_counter() - mulai

            # Ukur bagian tengah untuk menghindari tepi sinyal.
            valid = (
                (waktu >= 0.2)
                & (waktu <= 0.8)
                & (confidence >= 0.6)
                & np.isfinite(frekuensi)
                & (frekuensi > 0)
            )

            if not valid.any():
                raise ValueError(
                    "CREPE belum menghasilkan pitch valid pada nada uji."
                )

            pitch_median = float(np.median(frekuensi[valid]))
            selisih_cent = float(
                1200 * np.log2(pitch_median / frekuensi_target)
            )

        st.write(f"Target: {frekuensi_target:.2f} Hz")
        st.write(f"Pitch terukur: {pitch_median:.2f} Hz")
        st.write(f"Selisih: {selisih_cent:+.2f} cent")
        st.write(f"Waktu proses: {durasi_proses:.2f} detik")
        st.write(f"TensorFlow: {tf.__version__}")

        if abs(selisih_cent) <= 25:
            st.success("Uji CREPE lulus.")
        else:
            st.warning("CREPE berjalan, tetapi selisih melewati 25 cent.")

    except Exception as error:
        st.error("Uji CREPE belum berhasil.")
        st.exception(error)