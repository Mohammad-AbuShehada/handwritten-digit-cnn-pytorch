import streamlit as st
import torch
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
from predict_digit import preprocess_image, load_model, predict

st.title(" Handwritten Digit Recognizer")
st.write("Draw a digit or upload an image and get the prediction!")


with st.sidebar:
    st.header(" Drawing Settings")
    stroke_width = st.slider("Brush Size", 10, 40, 25)
    bg_color = st.color_picker("Background Color", "#FFFFFF")
    fg_color = st.color_picker("Pen Color", "#000000")

    if st.button(" Clear Canvas"):
        if "canvas_key" not in st.session_state:
            st.session_state.canvas_key = "canvas"
        st.session_state.canvas_key = np.random.randint(1000000)  


st.subheader(" Draw the digit here")
canvas_result = st_canvas(
    fill_color=bg_color,
    stroke_width=stroke_width,
    stroke_color=fg_color,
    background_color=bg_color,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=st.session_state.get("canvas_key", "canvas"),
)


st.subheader("Or upload an image 📸")
uploaded_file = st.file_uploader("Choose an image (PNG/JPG)", type=["png", "jpg", "jpeg"])


if st.button(" Predict", type="primary"):
    image_to_predict = None


    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
        img.save("temp_image.png")
        image_to_predict = "temp_image.png"
        st.image(img, caption="Processed Drawing (28x28)", width=200)


    elif uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        image.save("temp_image.png")
        image_to_predict = "temp_image.png"
        st.image(image, caption="Uploaded Image", width=200)

    else:
        st.warning("Please draw a digit or upload an image first!")
        st.stop()

    
    if image_to_predict:
        model, _ = load_model("mnist_model.pth")
        input_tensor = preprocess_image(image_to_predict)
        top3 = predict(model, input_tensor)

        predicted, confidence = top3[0]
        st.success(f"**Predicted Digit: {predicted}** (Confidence: {confidence:.2f}%)")

        st.write("**Top 3 Predictions:**")
        for digit, conf in top3:
            st.write(f"- **{digit}** → {conf:.2f}%")