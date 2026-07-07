import streamlit as st
from model_helper import predict

st.title("Vehicle Damage Detection")

st.caption(
    "Upload a clear photo of a vehicle's front or rear. "
    "The model classifies it as: Front Breakage, Front Crushed, Front Normal, "
    "Rear Breakage, Rear Crushed, or Rear Normal."
)

uploaded_file = st.file_uploader("Upload the file", type=["jpg", "png"])

if uploaded_file:
    image_path = "temp_file.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        st.image(uploaded_file, caption="Uploaded File", use_container_width=True)
        prediction = predict(image_path)
        st.info(f"Predicted Class: {prediction}")
