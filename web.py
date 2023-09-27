import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO


st.set_page_config(
    page_title="Pneumonia Detection",
    layout="wide"
)


st.title('Pneumonia Detection')
st.header('Upload the chest X-ray image')
file = st.file_uploader('', type=['png', 'jpg', 'jpeg'])


model = YOLO("./runs/classify/train2/weights/best.pt")

if file is not None:
    uploaded_image = Image.open(file)
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    image = np.array(uploaded_image)


    
    
    results = model(uploaded_image)

    
    res_dict = results[0].names
    res_probs = results[0].probs.data.tolist()
    res_max = res_dict[np.argmax(res_probs)]

    
    st.write(f'Predicted Class: {res_max}')


