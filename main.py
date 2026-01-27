import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="IMDB Sentiment AI",
    page_icon="🎬",
    layout="centered"
)

# --- Resource Loading (Cached) ---
@st.cache_resource
def load_assets():
    """Loads the model and word index once and caches them for performance."""
    # Corrected method name: get_word_index()
    word_index = imdb.get_word_index()
    # Ensure 'simple_rnn_imdb.h5' is in your project folder
    model = load_model('simple_rnn_imdb.h5')
    return word_index, model

# Initialize assets
try:
    word_index, model = load_assets()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# --- Utility Functions ---
def preprocess_text(text):
    """Converts raw text into the padded numerical sequence the model expects."""
    words = text.lower().split()
    # IMDB indices are offset by 3: 0=PAD, 1=START, 2=UNK
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# --- UI Layout ---
st.title("🎬 Movie Review Sentiment AI")
st.markdown("""
    This app uses a **Simple Recurrent Neural Network (RNN)** to classify movie reviews.
    The model was trained on 50,000 highly polar reviews from the IMDB database.
""")

# Sidebar for metadata
with st.sidebar:
    st.header("Model Info")
    st.info("Architecture: Simple RNN")
    st.info("Input Length: 500 words")
    st.write("The score represents the probability of the review being positive.")

st.divider()

# Input Section
user_input = st.text_area(
    "Paste your movie review here:", 
    placeholder="The acting was superb and the plot kept me on the edge of my seat...",
    height=200
)

# Action Button
if st.button("Analyze Sentiment", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner('🤖 AI is processing the text...'):
            try:
                # 1. Preprocess
                preprocessed_input = preprocess_text(user_input)
                
                # 2. Prediction
                prediction = model.predict(preprocessed_input, verbose=0)
                raw_score = prediction[0][0]

                # 3. Robustness Check: Handle NaN and Clipping
                # This prevents st.progress from throwing errors
                if np.isnan(raw_score):
                    st.error("The model encountered a calculation error (NaN). This can happen with very short or repetitive text.")
                else:
                    # Clip value between 0 and 1 just in case of floating point overflow
                    score = float(np.clip(raw_score, 0.0, 1.0))
                    
                    # 4. Logic handling
                    is_positive = score > 0.5
                    sentiment_label = "POSITIVE" if is_positive else "NEGATIVE"
                    result_color = "green" if is_positive else "red"
                    
                    # 5. Display Results
                    st.subheader("Result")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric(label="Sentiment", value=sentiment_label)
                    
                    with col2:
                        confidence = score if is_positive else (1 - score)
                        st.write(f"**AI Confidence:** {confidence:.2%}")
                        st.progress(score)

                    # Final stylistic alert
                    if is_positive:
                        st.success(f"This review sounds like a Thumbs Up! 👍 (Score: {score:.4f})")
                    else:
                        st.error(f"This review sounds like a Thumbs Down. 👎 (Score: {score:.4f})")

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# Footer
st.markdown("---")
st.caption("Built with TensorFlow & Streamlit • Data: IMDB Movie Reviews")