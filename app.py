#!/usr/bin/env python3
"""
Multimodal Mental Health Emotion Recognition System
Interactive Web Interface using Streamlit
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import time
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.inference.realtime_inference import get_inference_engine
from utils.visualizations import (
    create_probability_chart,
    create_confidence_gauge,
    create_model_contribution_chart,
    format_prediction_result
)

# Page configuration
st.set_page_config(
    page_title="Multimodal Emotion Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .emotion-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .emotion-text {
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
    }
    .confidence-text {
        font-size: 1.5rem;
        margin-top: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []


@st.cache_resource
def load_engine():
    """Load inference engine (cached)."""
    return get_inference_engine()


def add_to_history(modality, input_desc, result):
    """Add prediction to history."""
    if 'error' not in result:
        st.session_state.prediction_history.append({
            'timestamp': time.strftime('%H:%M:%S'),
            'modality': modality,
            'input': input_desc,
            'prediction': result['prediction'],
            'confidence': result['confidence']
        })
        # Keep only last 10
        if len(st.session_state.prediction_history) > 10:
            st.session_state.prediction_history.pop(0)


def main():
    """Main application."""
    
    # Header
    st.markdown('<p class="main-header">🧠 Multimodal Emotion Recognition System</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered emotion detection from audio, images, and text</p>', 
                unsafe_allow_html=True)
    
    # Load engine
    if st.session_state.engine is None:
        with st.spinner("🔄 Loading AI models... (this may take 10-15 seconds)"):
            st.session_state.engine = load_engine()
        st.success("✅ Models loaded successfully!")
    
    engine = st.session_state.engine
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This system uses deep learning to recognize emotions from:
        - 🎵 **Audio**: Speech emotion recognition
        - 📸 **Images**: Facial expression analysis
        - 💬 **Text**: Sentiment analysis
        - 🔗 **Fusion**: Combined multimodal prediction
        """)
        
        st.divider()
        
        st.header("📊 Model Performance")
        st.write("""
        **Individual Models:**
        - Audio: 51.94% accuracy
        - Facial: 66.93% accuracy
        - Text: 78.32% accuracy
        
        **Fusion System:**
        - Combined: ~79% accuracy
        """)
        
        st.divider()
        
        st.header("🎯 Emotion Categories")
        emotions = {
            '😢': 'Very Negative',
            '😕': 'Negative',
            '😐': 'Neutral',
            '🙂': 'Positive',
            '😄': 'Very Positive'
        }
        for emoji, emotion in emotions.items():
            st.write(f"{emoji} **{emotion}**")
        
        st.divider()
        
        # Settings
        st.header("⚙️ Settings")
        show_details = st.checkbox("Show detailed analysis", value=True)
        detect_face = st.checkbox("Auto-detect face in images", value=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Text Analysis",
        "🎵 Audio Analysis", 
        "📸 Image Analysis",
        "🔗 Multimodal Fusion",
        "📜 History"
    ])
    
    # ============================================
    # TAB 1: Text Analysis
    # ============================================
    with tab1:
        st.header("💬 Text Sentiment Analysis")
        st.write("Enter any text to analyze its emotional sentiment.")
        
        # Example texts
        with st.expander("📝 Try these examples"):
            example_texts = {
                "Very Positive": "I am extremely happy and excited about this wonderful opportunity!",
                "Positive": "This is quite good, I'm pleased with the results.",
                "Neutral": "The meeting is scheduled for tomorrow at 3 PM.",
                "Negative": "I'm disappointed and frustrated with how things turned out.",
                "Very Negative": "This is absolutely terrible and completely unacceptable."
            }
            
            for emotion, text in example_texts.items():
                if st.button(f"Try: {emotion}", key=f"example_{emotion}"):
                    st.session_state.text_input = text
        
        # Text input
        text_input = st.text_area(
            "Enter your text:",
            value=st.session_state.get('text_input', ''),
            height=150,
            placeholder="Type or paste your text here..."
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            analyze_text = st.button("🚀 Analyze Text", type="primary", use_container_width=True)
        with col2:
            clear_text = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_text:
            st.session_state.text_input = ''
            st.rerun()
        
        if analyze_text and text_input.strip():
            with st.spinner("🔍 Analyzing text..."):
                result = engine.predict_text(text_input)
            
            if 'error' not in result:
                # Add to history
                add_to_history('Text', text_input[:50] + '...', result)
                
                # Display results
                st.divider()
                st.subheader("📊 Analysis Results")
                
                # Main prediction box
                emotion_emoji = {
                    'very_negative': '😢',
                    'negative': '😕',
                    'neutral': '😐',
                    'positive': '🙂',
                    'very_positive': '😄'
                }
                
                emoji = emotion_emoji.get(result['prediction'], '😐')
                emotion_display = result['prediction'].replace('_', ' ').title()
                
                st.markdown(f"""
                <div class="emotion-box">
                    <p class="emotion-text">{emoji} {emotion_display}</p>
                    <p class="confidence-text">Confidence: {result['confidence']*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(
                        create_probability_chart(result['probabilities']),
                        use_container_width=True
                    )
                
                with col2:
                    st.plotly_chart(
                        create_confidence_gauge(result['confidence']),
                        use_container_width=True
                    )
                
                # Details
                if show_details:
                    with st.expander("🔍 Detailed Analysis"):
                        st.write(f"**Processing time:** {result['processing_time']:.3f} seconds")
                        st.write(f"**Text length:** {result['text_length']} characters")
                        st.write(f"**Model used:** BiLSTM Text Sentiment Model")
                        
                        st.write("**Probability breakdown:**")
                        prob_df = pd.DataFrame([
                            {'Emotion': k.replace('_', ' ').title(), 'Probability': f"{v*100:.2f}%"}
                            for k, v in sorted(result['probabilities'].items(), 
                                             key=lambda x: x[1], reverse=True)
                        ])
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)
            else:
                st.error(f"❌ Error: {result['error']}")
    
    # ============================================
    # TAB 2: Audio Analysis
    # ============================================
    with tab2:
        st.header("🎵 Audio Emotion Recognition")
        st.write("Upload an audio file to detect emotional content in speech.")
        
        # File uploader
        audio_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'ogg', 'm4a'],
            help="Supported formats: WAV, MP3, OGG, M4A"
        )
        
        if audio_file is not None:
            # Display audio player
            st.audio(audio_file, format=f'audio/{audio_file.type.split("/")[1]}')
            
            # Analyze button
            if st.button("🚀 Analyze Audio", type="primary"):
                with st.spinner("🔍 Processing audio..."):
                    
                    result = engine.predict_audio(audio_file)
                
                if 'error' not in result:
                    # Add to history
                    add_to_history('Audio', audio_file.name, result)
                    
                    # Display results
                    st.divider()
                    st.subheader("📊 Analysis Results")
                    
                    # Main prediction
                    emotion_emoji = {
                        'very_negative': '😢',
                        'negative': '😕',
                        'neutral': '😐',
                        'positive': '🙂',
                        'very_positive': '😄'
                    }
                    
                    emoji = emotion_emoji.get(result['prediction'], '😐')
                    emotion_display = result['prediction'].replace('_', ' ').title()
                    
                    st.markdown(f"""
                    <div class="emotion-box">
                        <p class="emotion-text">{emoji} {emotion_display}</p>
                        <p class="confidence-text">Confidence: {result['confidence']*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(
                            create_probability_chart(result['probabilities']),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.plotly_chart(
                            create_confidence_gauge(result['confidence']),
                            use_container_width=True
                        )
                    
                    # Details
                    if show_details:
                        with st.expander("🔍 Detailed Analysis"):
                            st.write(f"**Processing time:** {result['processing_time']:.3f} seconds")
                            st.write(f"**Audio duration:** {result['audio_info'].get('duration', 'N/A'):.2f} seconds")
                            st.write(f"**Sample rate:** {result['audio_info'].get('sample_rate', 'N/A')} Hz")
                            st.write(f"**Model used:** ResNet50V2 Audio Emotion Model")
                else:
                    st.error(f"❌ Error: {result['error']}")
        else:
            st.info("👆 Upload an audio file to begin analysis")
    
    # ============================================
    # TAB 3: Image Analysis
    # ============================================
    with tab3:
        st.header("📸 Facial Emotion Recognition")
        st.write("Upload an image to detect emotions from facial expressions.")
        
        # File uploader
        image_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, PNG"
        )
        
        if image_file is not None:
            # Display image
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image_file, caption="Uploaded Image", use_container_width=True)
            
            # Analyze button
            if st.button("🚀 Analyze Image", type="primary"):
                with st.spinner("🔍 Processing image..."):
                
                    result = engine.predict_image(image_file, detect_face=detect_face)
                
                if 'error' not in result:
                    # Add to history
                    add_to_history('Image', image_file.name, result)
                    
                    # Display results
                    st.divider()
                    st.subheader("📊 Analysis Results")
                    
                    # Main prediction
                    emotion_emoji = {
                        'very_negative': '😢',
                        'negative': '😕',
                        'neutral': '😐',
                        'positive': '🙂',
                        'very_positive': '😄'
                    }
                    
                    emoji = emotion_emoji.get(result['prediction'], '😐')
                    emotion_display = result['prediction'].replace('_', ' ').title()
                    
                    st.markdown(f"""
                    <div class="emotion-box">
                        <p class="emotion-text">{emoji} {emotion_display}</p>
                        <p class="confidence-text">Confidence: {result['confidence']*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(
                            create_probability_chart(result['probabilities']),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.plotly_chart(
                            create_confidence_gauge(result['confidence']),
                            use_container_width=True
                        )
                    
                    # Details
                    if show_details:
                        with st.expander("🔍 Detailed Analysis"):
                            st.write(f"**Processing time:** {result['processing_time']:.3f} seconds")
                            st.write(f"**Image size:** {result['image_info'].get('width', 'N/A')} × {result['image_info'].get('height', 'N/A')} pixels")
                            st.write(f"**Face detected:** {'Yes' if detect_face else 'Not checked'}")
                            st.write(f"**Model used:** ResNet50V2 Facial Emotion Model")
                else:
                    st.error(f"❌ Error: {result['error']}")
        else:
            st.info("👆 Upload an image file to begin analysis")
    
    # ============================================
    # TAB 4: Multimodal Fusion
    # ============================================
    with tab4:
        st.header("🔗 Multimodal Fusion Analysis")
        st.write("Combine multiple inputs for more robust emotion prediction!")
        
        st.info("💡 **Tip:** The system will intelligently fuse predictions from all provided modalities, weighted by model accuracy.")
        
        # Input sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Text Input")
            fusion_text = st.text_area(
                "Enter text (optional):",
                height=100,
                key="fusion_text"
            )
        
        with col2:
            st.subheader("🎵 Audio Input")
            fusion_audio = st.file_uploader(
                "Upload audio (optional)",
                type=['wav', 'mp3', 'ogg', 'm4a'],
                key="fusion_audio"
            )
            if fusion_audio:
                st.audio(fusion_audio)
        
        st.subheader("📸 Image Input")
        fusion_image = st.file_uploader(
            "Upload image (optional)",
            type=['jpg', 'jpeg', 'png'],
            key="fusion_image"
        )
        if fusion_image:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(fusion_image, use_container_width=True)
        
        # Analyze button
        st.divider()
        
        # Check if at least one input provided
        inputs_provided = any([
            fusion_text and fusion_text.strip(),
            fusion_audio is not None,
            fusion_image is not None
        ])
        
        if inputs_provided:
            if st.button("🚀 Analyze All Inputs", type="primary", use_container_width=True):
                with st.spinner("🔍 Processing multimodal inputs..."):
                    
                    
                    result = engine.predict_multimodal(
                        audio=fusion_audio,
                        image=fusion_image,
                        text=fusion_text if fusion_text.strip() else None,
                        detect_face=detect_face
                    )
                
                if 'error' not in result:
                    # Add to history
                    modalities_str = ' + '.join(result['modalities_provided'])
                    add_to_history('Fusion', modalities_str, result)
                    
                    # Display results
                    st.divider()
                    st.subheader("📊 Fusion Analysis Results")
                    
                    # Main prediction
                    emotion_emoji = {
                        'very_negative': '😢',
                        'negative': '😕',
                        'neutral': '😐',
                        'positive': '🙂',
                        'very_positive': '😄'
                    }
                    
                    emoji = emotion_emoji.get(result['prediction'], '😐')
                    emotion_display = result['prediction'].replace('_', ' ').title()
                    
                    st.markdown(f"""
                    <div class="emotion-box">
                        <p class="emotion-text">{emoji} {emotion_display}</p>
                        <p class="confidence-text">Confidence: {result['confidence']*100:.1f}%</p>
                        <p style="font-size: 1rem; margin-top: 0.5rem;">
                            Modalities: {' + '.join(result['modalities_provided'])}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(
                            create_probability_chart(result['probabilities']),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.plotly_chart(
                            create_model_contribution_chart(
                                result['modalities_used'],
                                result['weights_used']
                            ),
                            use_container_width=True
                        )
                    
                    # Details
                    if show_details:
                        with st.expander("🔍 Detailed Analysis"):
                            st.write(f"**Processing time:** {result['processing_time']:.3f} seconds")
                            st.write(f"**Modalities used:** {', '.join(result['modalities_provided'])}")
                            st.write("**Model contributions:**")
                            for mod, weight in result['weights_used'].items():
                                st.write(f"  - {mod.title()}: {weight*100:.1f}%")
                else:
                    st.error(f"❌ Error: {result['error']}")
        else:
            st.warning("⚠️ Please provide at least one input (text, audio, or image)")
    
    # ============================================
    # TAB 5: History
    # ============================================
    with tab5:
        st.header("📜 Prediction History")
        
        if st.session_state.prediction_history:
            st.write(f"Last {len(st.session_state.prediction_history)} predictions:")
            
            # Create dataframe
            history_df = pd.DataFrame(st.session_state.prediction_history)
            history_df['confidence'] = history_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
            history_df.columns = ['Time', 'Modality', 'Input', 'Prediction', 'Confidence']
            
            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
        else:
            st.info("No predictions yet. Try analyzing some inputs!")


if __name__ == "__main__":
    main()
