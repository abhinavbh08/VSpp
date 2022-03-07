from email.mime import image
import io
import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image
from io import StringIO
import requests

cnt = 0
files = []

# Upload the files
uploaded_files = st.file_uploader("Upload the images here.", accept_multiple_files=True)


# Displaying the uploaded files.
if uploaded_files:
    if len(uploaded_files) > 6:
        st.error("Please upload less than or equal to 6 files.")
    else:
        for uploaded_file in uploaded_files:
            if cnt % 3 == 0:
                cols = st.columns(3)
                cnt = 0
            if uploaded_file is not None:
                bytes_data = uploaded_file.read()
                files.append(("myfile", ("image", bytes_data, "image/png")))
                cols[cnt].image(bytes_data, use_column_width=True)
                cnt += 1

cnt = 0

# Getting the text with which to match the images.
st.text_input("Upload a description here.", key="description")
if len(st.session_state.description) == 0:
    st.error("Please add some description")
else:
    if len(files) != 0:
        result = requests.post(
            "http://0.0.0.0:8081/uploadfile/",
            files=files,
            data={"desc": st.session_state.description},
        )
        st.write(st.session_state.description)
        indices = result.json()["indices"]

        # Display the results in the order of similarity.
        for idx in indices:
            if cnt % 3 == 0:
                cols = st.columns(3)
                cnt = 0
            cols[cnt].image(files[idx][1][1], use_column_width=True)
            cnt += 1
