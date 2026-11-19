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