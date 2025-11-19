# https://console.cloud.google.com/apis/api/customsearch.googleapis.com/quotas?project=precise-truck-453708-v1
# https://console.cloud.google.com/apis/credentials?project=precise-truck-453708-v1
# https://programmablesearchengine.google.com/controlpanel/all

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from gtts import gTTS
import streamlit as st
import os
import tempfile
import io
import spacy
import wikipediaapi
from googleapiclient.discovery import build
from collections import Counter

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Wikipedia API with custom user-agent
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='ImageCaptioningApp/1.0 (hymareddy8332@gmail.com)'
)

# Google Custom Search API Configuration
API_KEY = "AIzaSyBAmdoIK0n843ttb1N0Q20JjNFh8ytKS_0"
SEARCH_ENGINE_ID = "60f7f700474694ae5"

def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    caption_ids = model.generate(**inputs)
    caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
    return caption

def extract_keywords(text, top_n=5):
    doc = nlp(text)
    
    keywords = []

    for chunk in doc.noun_chunks:
        if not chunk.root.is_stop and chunk.root.is_alpha:
            keywords.append(chunk.text.lower())

    for token in doc:
        if token.pos_ in ("NOUN", "PROPN", "ADJ") and not token.is_stop and token.is_alpha:
            keywords.append(token.text.lower())
    # Count and return the most common keywords
    common_keywords = [item[0] for item in Counter(keywords).most_common(top_n)]
    return list(set(common_keywords))

def get_wikipedia_summary(keyword):
    page = wiki_wiki.page(keyword)
    return page.summary if page.exists() else "No information found."

def google_search(keyword):
    service = build("customsearch", "v1", developerKey=API_KEY)
    result = service.cse().list(q=keyword, cx=SEARCH_ENGINE_ID, searchType='image').execute()
    image_url = result['items'][0]['link'] if 'items' in result else None
    reference_url = result['items'][0]['image']['contextLink'] if 'items' in result else None
    return image_url, reference_url

def text_to_speech(text):
    tts = gTTS(text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# Streamlit UI
st.title("🖼️ Image Captioning App with Keyword Insights")
st.subheader("Upload an image, generate a caption, and explore related insights!")

upload_option = st.radio("Select Image Input Method:", ("Browse", "Use Camera"))

image = None
if upload_option == "Browse":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
elif upload_option == "Use Camera":
    captured_image = st.camera_input("Take a picture")
    if captured_image is not None:
        image = Image.open(captured_image).convert("RGB")

if image is not None:
    st.image(image, caption="Uploaded Image")

    if st.button("Generate Caption and Insights"):
        with st.spinner("Generating caption and extracting insights..."):
            caption = generate_caption(image)
            keywords = extract_keywords(caption, top_n=10)
        
        st.success("Caption and Insights Generated!")
        st.write("**Caption:**", caption)

        # Convert caption to speech and play it
        audio_file_path = text_to_speech(caption)
        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")
        os.remove(audio_file_path)

        # Display insights for each keyword
        st.subheader("Keyword Insights")
        for keyword in keywords:
            with st.expander(f"Keyword: {keyword}"):
                summary = get_wikipedia_summary(keyword)
                st.write("**Description:**", summary)

                image_url, reference_url = google_search(keyword)
                if image_url:
                    st.image(image_url, caption=f"Image related to {keyword}")
                if reference_url:
                    st.markdown(f"[More Info]({reference_url})")
                if not image_url and not reference_url:
                    st.write("No image or reference found.")
else:
    st.info("Please upload an image to generate a caption and insights.")




















