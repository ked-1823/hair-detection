import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ======================
# Load Models
# ======================
@st.cache_resource
def load_models():
    age_gender_model = load_model("age_gender_model.h5")
    hair_model = load_model("hair_model_full.h5")
    return age_gender_model, hair_model

age_gender_model, hair_model = load_models()

# ======================
# Prediction Function
# ======================
def predict_gender_with_hair(img):
    """
    Input: RGB image as numpy array (any size)
    Returns: gender, age_group, hair
    """
    try:
        # Ensure image is valid
        if img is None or img.size == 0:
            return None, None, None

        # Preprocess input (resize all inputs to 224x224 → model expects fixed size)
        img_input = cv2.resize(img, (224, 224))
        img_input = img_input.astype("float32") / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        # Age & gender prediction
        pred_gender, pred_age = age_gender_model.predict(img_input, verbose=0)
        pred_gender_class = int(pred_gender[0][0] > 0.5)   # 0=female, 1=male
        pred_age_class = int(np.argmax(pred_age[0]))       # 0=<20, 1=20–30, 2=>30

        # Hair prediction
        hair_pred = hair_model.predict(img_input, verbose=0)
        hair_length = "long" if hair_pred[0][0] > 0.5 else "short"

        # Custom rule for age 20–30
        if pred_age_class == 1:
            if hair_length == "long":
                pred_gender_class = 0  # override → female
            elif hair_length == "short":
                pred_gender_class = 1  # override → male

        return pred_gender_class, pred_age_class, hair_length

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None, None

# ======================
# Streamlit App
# ======================
st.title("👩‍🦱 Long Hair Gender & Age Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict
    gender, age_group, hair = predict_gender_with_hair(img)

    # Mapping
    gender_map = {0: "Female", 1: "Male"}
    age_map = {0: "<20", 1: "20–30", 2: ">30"}

    # Show Results
    if gender is not None:
        st.subheader("🔎 Prediction")
        st.write(f"**Gender:** {gender_map[gender]}")
        st.write(f"**Age Group:** {age_map[age_group]}")
        st.write(f"**Hair Length:** {hair}")
    else:
        st.warning("Could not process the image. Please try another one.")
