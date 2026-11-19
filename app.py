import platform
import shutil
import streamlit as st

st.set_page_config(page_title="VocalAlign", page_icon="🎵")

st.title("VocalAlign")
st.write("Uji awal lingkungan komputasi untuk analisis vokal.")

st.success("Aplikasi Python berhasil berjalan.")
st.write(f"Versi Python: {platform.python_version()}")

if shutil.which("ffmpeg"):
    st.success("FFmpeg tersedia untuk memproses video.")
else:
    st.error("FFmpeg belum tersedia.")
