import streamlit as st
import torch
from PIL import Image
from predict_digit import preprocess_image, load_model, predict  # from your file

st.title("🖐️ Handwritten Digit Recognizer")
st.write("Upload an image of a handwritten digit and get the prediction!")

uploaded_file = st.file_uploader("Choose an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Save temporary image
    image = Image.open(uploaded_file)
    image.save("temp_image.png")
   
    st.image(image, caption="Uploaded Image", use_column_width=True)
   
    if st.button("🔍 Predict"):
        model, _ = load_model("mnist_model.pth")
        input_tensor = preprocess_image("temp_image.png")
        top3 = predict(model, input_tensor)
       
        predicted, confidence = top3[0]
        st.success(f"**Predicted Digit: {predicted}** (Confidence: {confidence:.2f}%)")
       
        st.write("**Top 3 Predictions:**")
        for digit, conf in top3:
            st.write(f"- {digit} → {conf:.2f}%")