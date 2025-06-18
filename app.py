import streamlit as st
import numpy as np
from keras.models import load_model, Model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.sequence import pad_sequences
import pickle
import os

### Set Streamlit page config FIRST ###
st.set_page_config(page_title="🖼️ Image Captioning", layout="centered")

# Load resources
@st.cache_resource
def load_resources():
    with open('saved/word_to_idx.pkl', 'rb') as f:
        word_to_idx = pickle.load(f)
    with open('saved/idx_to_word.pkl', 'rb') as f:
        idx_to_word = pickle.load(f)
    with open('saved/max_len.pkl', 'rb') as f:
        max_len = pickle.load(f)
    return word_to_idx, idx_to_word, max_len

word_to_idx, idx_to_word, max_len = load_resources()

@st.cache_resource
def load_captioning_model():
    return load_model('model_weights/model_119.keras')

model = load_captioning_model()

@st.cache_resource
def load_feature_extractor():
    model_new = ResNet50(weights="imagenet", input_shape=(224,224,3))
    model_new = Model(model_new.input, model_new.layers[-2].output)
    return model_new

model_new = load_feature_extractor()

def preprocess_img(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def encode_img(img_path, model_new):
    img = preprocess_img(img_path)
    feature_vect = model_new.predict(img)
    feature_vect = feature_vect.reshape((1,2048))
    return feature_vect

def predict_caption(photo, model, word_to_idx, idx_to_word, max_len):
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')
        ypred = model.predict([photo, sequence], verbose=0).argmax()
        word = idx_to_word[ypred]
        in_text += ' ' + word
        if word == "endseq":
            break
    return ' '.join(in_text.split()[1:-1])

### ------------------ Streamlit App ------------------###

st.title("🧠 Image Captioning with Deep Learning")
st.markdown("""
Upload an image and let the model generates a caption for it using a combination of CNN + LSTM.
""")

uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(temp_path, caption='🖼️ Uploaded Image', use_column_width=True)
    with st.spinner('Generating caption... ⏳'):
        photo_2048 = encode_img(temp_path, model_new)
        caption = predict_caption(photo_2048, model, word_to_idx, idx_to_word, max_len)
    st.success("✅ Caption Generated!")
    st.markdown(f"**📝 Caption:** `{caption}`")
    os.remove(temp_path)