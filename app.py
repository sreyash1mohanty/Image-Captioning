import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from PIL import Image
import pickle
import os
import sys

### Set Streamlit page config FIRST ###
st.set_page_config(page_title="🖼️ Image Captioning", layout="centered")

# Load resources with enhanced error handling
@st.cache_resource
def load_resources():
    try:
        with open('saved/word_to_idx.pkl', 'rb') as f:
            word_to_idx = pickle.load(f)
        with open('saved/idx_to_word.pkl', 'rb') as f:
            idx_to_word = pickle.load(f)
        with open('saved/max_len.pkl', 'rb') as f:
            max_len = pickle.load(f)
        return word_to_idx, idx_to_word, max_len
    except Exception as e:
        st.error(f"❌ Error loading resources: {e}")
        st.stop()

word_to_idx, idx_to_word, max_len = load_resources()

# Improved model loading with validation
@st.cache_resource
def load_captioning_model():
    model_path = 'model_weights/model_119.keras'
    try:
        # File validation
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # in MB
        st.sidebar.info(f"Model size: {file_size:.2f} MB")
        
        if file_size < 1:  # Assuming model should be >1MB
            raise ValueError(f"Model file is too small ({file_size:.2f} MB), likely corrupted")
        
        # Load model
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.error("Please check model file path and integrity")
        st.stop()

model = load_captioning_model()

@st.cache_resource
def load_feature_extractor():
    try:
        model_new = ResNet50(weights="imagenet", input_shape=(224,224,3))
        model_new = Model(model_new.input, model_new.layers[-2].output)
        return model_new
    except Exception as e:
        st.error(f"❌ Error loading feature extractor: {e}")
        st.stop()

model_new = load_feature_extractor()

def preprocess_img(img):
    # Convert to RGB if image has alpha channel (4 channels)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def encode_img(uploaded_file, model_new):
    img = Image.open(uploaded_file)
    img = preprocess_img(img)
    feature_vect = model_new.predict(img, verbose=0)
    return feature_vect.reshape((1, 2048))

def predict_caption(photo, model, word_to_idx, idx_to_word, max_len):
    in_text = "startseq"
    for _ in range(max_len):
        # Convert current text to sequence of indices
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        # Pad the sequence to fixed length
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')
        
        # Predict next word
        ypred = model.predict([photo, sequence], verbose=0)
        ypred = np.argmax(ypred)  # Get the index with highest probability
        
        word = idx_to_word.get(ypred, "")
        in_text += " " + word
        
        if word == "endseq" or word == "":
            break
            
    # Format final caption
    caption = in_text.split()[1:-1]  # Remove startseq and endseq
    return " ".join(caption).capitalize()

### ------------------ Streamlit App ------------------###

st.title("🧠 Image Captioning with Deep Learning")
st.markdown("""
Upload an image and let the model generate a caption using CNN + LSTM architecture.
""")

# Display TF version in sidebar
st.sidebar.subheader("Environment Info")
st.sidebar.text(f"TensorFlow Version: {tf.__version__}")
st.sidebar.text(f"Python Version: {sys.version.split()[0]}")

uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption='🖼️ Uploaded Image', use_column_width=True)
    
    with st.spinner('Generating caption... ⏳'):
        try:
            photo_2048 = encode_img(uploaded_file, model_new)
            caption = predict_caption(photo_2048, model, word_to_idx, idx_to_word, max_len)
            
            with col2:
                st.success("✅ Caption Generated!")
                st.markdown(f"**📝 Caption:** {caption}")
        except Exception as e:
            st.error(f"❌ Error during caption generation: {e}")
            st.exception(e)