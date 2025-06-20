import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Wood Defect Image Info", layout="centered")
st.title("📷 Info about wood piece")

uploaded_file = st.file_uploader("Upload picture of wood (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Preview", use_container_width=True)

    width, height = image.size
    file_name = uploaded_file.name

    st.markdown("---")
    st.subheader("📋 Picture details:")
    st.write(f"**File name:** `{file_name}`")
    st.write(f"**Width:** {width} px")
    st.write(f"**Height:** {height} px")

else:
    st.info("Upload picture to see it's details.")
