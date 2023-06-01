import cv2
import numpy as np
import streamlit as st
from parti.streamlit_rec import reconstruction

uploaded_file = st.file_uploader("Choose a image file", type="jpg")


# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

col1, col2 = st.columns(2)

with col1:
    
    text_input = st.text_input(
        "Write your caption",
        "",
        key="placeholder",
    )

    if text_input:
        st.write("You entered: ", text_input)

        # img = reconstruction(img=opencv_image, checkpoint_path="output/models/vit_vq_step_270000.pt")



if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    img = reconstruction(img=opencv_image, checkpoint_path="output/models/vit_vq_step_270000.pt")

    # Now do something with the image! For example, let's display it:
    st.image(img, channels="BGR")