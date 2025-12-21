# 🧠 Real-Time Multimodal AI Mental Health Assistant

**A Complete Guide to Building a Digital Therapist That Listens, Sees, and Understands You**

> Building AI that cares, one emotion at a time.

---

## 🏷️ Badges

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow&logoColor=white)  
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-brightgreen?logo=streamlit&logoColor=white)  
![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-red?logo=keras&logoColor=white)  
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue?logo=numpy&logoColor=white)  
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-teal?logo=pandas&logoColor=white)  
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-yellow)  
![Status](https://img.shields.io/badge/Status-Complete%20to%20Streamlit%20UI-brightgreen)

---

## 🧭 Table of Contents

1. Project Overview  
2. Project Status  
3. Project Roadmap  
4. System Architecture  
5. Completed Features  
6. Tech Stack and Libraries  
7. Project Structure  
8. Data Collection and Datasets  
9. Model Architectures  
10. Multimodal Fusion Strategy  
11. Inference and Real Time Processing  
12. Streamlit Web Interface  
13. How to Run the Project  
14. Ethical and Privacy Considerations  
15. Future Enhancements  
16. Author and Contact  

---

## 🌍 Project Overview

This project is a complete multimodal AI system designed to act as a supportive digital mental health assistant. It processes emotional signals from multiple sources including audio, facial expressions, and text to build a holistic understanding of a user's emotional state. The system combines specialized deep learning models for each modality, fuses their outputs using intelligent combination strategies, and presents insights through a friendly Streamlit web interface. Every component has been carefully developed from data collection through model training to inference and visualization, with a focus on clarity, empathy, and accessibility.

---

## 🧩 Project Status

The project has progressed through all major development stages and is now feature complete with a fully functional Streamlit interface ready for demonstration and further refinement.

- ✅ Phase 1: Environment setup and architecture design  
- ✅ Phase 2: Data collection from RAVDESS, FER2013, and text datasets  
- ✅ Phase 3: Data preprocessing and feature extraction for all modalities  
- ✅ Phase 4: Individual model development for audio, facial, and text  
- ✅ Phase 5: Multimodal fusion layer implementation  
- ✅ Phase 6: Real time inference pipeline and model evaluation  
- ✅ Phase 7: Streamlit based web interface for user interaction  
- ⏳ Phase 8: Advanced optimization, deployment, and extended testing  

---

## 🗺️ Project Roadmap

| Phase | Title                              | Description                                              | Status  |
|:------|:-----------------------------------|:---------------------------------------------------------|:-------:|
| 1     | Foundation and Setup              | Environment, folders, dependencies, configuration       | ✅      |
| 2     | Data Collection                   | Gathering audio, facial, and text emotion datasets      | ✅      |
| 3     | Data Preprocessing and Features   | Cleaning, normalizing, extracting meaningful features   | ✅      |
| 4     | Individual Model Development      | Building CNN LSTM audio, ResNet50V2 facial, LSTM text  | ✅      |
| 5     | Model Training and Evaluation     | Training all models, analyzing performance metrics      | ✅      |
| 6     | Multimodal Fusion                 | Combining outputs from all modalities intelligently     | ✅      |
| 7     | Real Time Inference               | Building inference pipelines for live processing        | ✅      |
| 8     | Streamlit Prototype UI            | Interactive web interface for user interaction         | ✅      |
| 9     | Optimization and Refinement       | Performance tuning, model compression, improvements     | ⏳      |
| 10    | Production Deployment             | Packaging, hosting, and real world testing             | ⏳      |

---

## ⚙️ System Architecture

The system follows a modular pipeline with clear separation between data collection, processing, modeling, and presentation:

```
User Input (Audio, Image, Text)
        ↓
Preprocessing Layer
  ├─ Audio Processing (MFCC extraction, normalization)
  ├─ Image Processing (face detection, alignment)
  └─ Text Processing (tokenization, normalization)
        ↓
Individual Models
  ├─ Audio Model (CNN LSTM Hybrid)
  ├─ Facial Model (ResNet50V2 Transfer Learning)
  └─ Text Model (LSTM based)
        ↓
Multimodal Fusion Layer
  (Weighted averaging and intelligent combination)
        ↓
Unified Emotional Profile
        ↓
Streamlit Interface
  (Visualization, feedback, recommendations)
        ↓
User Friendly Output
```

Each component is designed to be modular, allowing for easy updates and improvements without affecting the rest of the system.

---

## ✨ Completed Features

**Data Collection and Preprocessing**  
- Integrated RAVDESS dataset for audio emotion recognition  
- Integrated FER2013 dataset for facial emotion recognition  
- Collected and preprocessed text based sentiment data  
- Built robust preprocessing pipelines with validation checks  

**Individual Emotion Models**  
- Audio Emotion Recognition: CNN LSTM hybrid architecture for capturing temporal patterns in speech  
- Facial Emotion Recognition: ResNet50V2 transfer learning for robust facial expression analysis  
- Text Sentiment Analysis: LSTM based model for understanding emotional tone in written text  

**Multimodal Fusion**  
- Implemented intelligent fusion strategy combining predictions from all three modalities  
- Used weighted averaging and confidence scoring to balance contributions from each model  
- Added fallback mechanisms for cases where one or more modalities fail  

**Real Time Inference**  
- Built processors for converting live audio, video, and text inputs into model compatible formats  
- Implemented efficient batch processing for faster inference  
- Added error handling and graceful degradation when inputs are incomplete  

**Streamlit Web Application**  
- Clean and intuitive interface for uploading or recording inputs  
- Real time processing with clear status updates and progress feedback  
- Visual emotion dashboard showing results from each modality  
- Explanation of predictions in a gentle and supportive tone  
- Export functionality for results and emotional insights  

---

## 🛠️ Tech Stack and Libraries

This section lists the actual technologies and libraries used throughout the project, organized by function.

**Core Platform and Framework**  
- Python 3.8+: Primary programming language for all development  
- TensorFlow 2.x: Deep learning framework for building and training models  
- Keras: High level API for neural network development within TensorFlow  
- Streamlit: Web framework for building the interactive user interface  

**Data Handling and Processing**  
- Pandas: Loading, cleaning, and managing structured data  
- NumPy: Numerical computations and array operations  
- Librosa: Audio processing, feature extraction, and signal analysis  
- OpenCV: Image processing and computer vision tasks  
- Scikit learn: Data preprocessing, metrics, and utility functions  

**Model Development and Training**  
- Keras Sequential and Functional APIs: Building neural network architectures  
- ResNet50V2: Pre trained model for transfer learning in facial emotion recognition  
- Conv1D, LSTM, Dense layers: Building custom architectures for audio and text  
- Optimizers: Adam for training all models  
- Loss functions: Categorical crossentropy for multi class classification  

**Visualization and Reporting**  
- Matplotlib: Creating plots and charts within the Streamlit app  
- Seaborn: Statistical visualization for emotion distributions  
- Plotly: Interactive visualizations for better user engagement  

**Utilities and Support**  
- OS and Pathlib: File system operations and path management  
- Joblib: Saving and loading trained models  
- Pickle: Serializing Python objects for caching and persistence  
- Requests: Making HTTP requests if external APIs are involved  
- Python Logging: Tracking errors and system behavior  

**Environment and Dependency Management**  
- Virtual environment (venv): Isolating project dependencies  
- requirements.txt: Listing all project dependencies for reproducibility  

---

## 🗂️ Project Structure

The project follows a well organized structure that separates concerns and makes the codebase easy to navigate:

```
MULTIMODAL_MENTAL_HEALTH_AI/
│
├── .vscode/                        # VS Code configuration
├── checkpoints/                    # Model checkpoint files during training
├── data/
│   ├── raw/
│   │   ├── audio/                 # Raw audio files from RAVDESS
│   │   ├── facial/                # Raw image files from FER2013
│   │   └── text/                  # Raw text sentiment data
│   └── processed/                 # Cleaned and preprocessed data
│
├── deployment/                    # Deployment related files
├── docs/                          # Documentation and guides
├── logs/                          # Application logs
├── models/                        # Trained model files
│
├── notebooks/                     # Jupyter notebooks for experimentation
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py             # API endpoints if applicable
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py           # Configuration and constants
│   │
│   ├── data_collection/
│   │   ├── __init__.py
│   │   ├── audio_collector_ravdess.py
│   │   ├── facial_collector_fer2013.py
│   │   ├── text_collector.py
│   │   └── data_validator.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate_audio.py
│   │   ├── evaluate_facial.py
│   │   ├── evaluate_text.py
│   │
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── multimodal_fusion.py   # Main fusion logic
│   │   ├── evaluate_fusion.py
│   │   └── fusion_demo.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── audio_processor.py
│   │   ├── image_processor.py
│   │   └── realtime_inference.py
│   │
│   ├── models/                    # Saved trained models
│   ├── preprocessing/             # Data preprocessing utilities
│   ├── training/                  # Model training scripts
│   └── utils/
│       ├── __init__.py
│       └── visualizations.py      # Plotting and visualization helpers
│
├── streamlit_app/
│   └── app.py                     # Main Streamlit application
│
├── tensorboard_logs/
│   ├── audio/
│   ├── facial/
│   └── text/
│
├── tests/                         # Unit and integration tests
├── utils/                         # General utility functions
│   ├── __init__.py
│   └── visualizations.py
│
├── .env                           # Environment variables (not in version control)
├── .env.example                   # Example environment file
├── .gitignore
├── app.py                         # Main entry point
├── collect_all_data.py           # Script to collect all datasets
├── initialize_project.py          # Project initialization
├── main.py                        # Alternative main entry point
├── requirements.txt               # Python dependencies
├── test_wsgi_setup.py            # WSGI testing
├── verify_gpu.py                 # GPU availability check
└── README.md                      # This file
```

---

## 📊 Data Collection and Datasets

The project integrates three high quality emotion datasets to ensure robust model training:

**RAVDESS (Audio Emotion Recognition)**  
- Dataset: Remote Audio Visual Emotion Recognition Across Speakers  
- Size: 7,356 audio files covering 8 emotions  
- Emotions: Neutral, calm, happy, sad, angry, fearful, disgusted, surprised  
- Quality: High quality recordings from professional actors  
- Usage: Training the audio emotion recognition model  

**FER2013 (Facial Emotion Recognition)**  
- Dataset: Facial Expression Recognition 2013  
- Size: 35,887 grayscale images (48x48 pixels)  
- Emotions: Angry, disgust, fear, happy, neutral, sad, surprise  
- Quality: Real world facial expressions with natural variations  
- Usage: Training the facial emotion recognition model with transfer learning  

**Text Based Sentiment Data**  
- Combined from various sentiment analysis datasets  
- Emotions: Positive, negative, neutral sentiments  
- Quality: Real user generated text with emotional context  
- Usage: Training the text emotion model  

---

## 🧠 Model Architectures

Each modality uses a specialized architecture tailored to its unique characteristics:

**Audio Emotion Recognition (CNN LSTM Hybrid)**  
```
Input: MFCC features (Time steps × 13 MFCC coefficients)
       ↓
Conv1D Layer (32 filters, kernel size 5)
       ↓
MaxPooling1D
       ↓
LSTM Layer (64 units, return sequences)
       ↓
LSTM Layer (32 units)
       ↓
Dense Layer (128 units, ReLU)
       ↓
Dropout (0.5)
       ↓
Output Dense Layer (8 units, Softmax)
       ↓
Emotion Prediction (8 classes)
```

**Facial Emotion Recognition (ResNet50V2 Transfer Learning)**  
```
Input: Preprocessed facial image (224×224×3)
       ↓
ResNet50V2 (pre trained on ImageNet)
       ↓
Global Average Pooling
       ↓
Dense Layer (256 units, ReLU)
       ↓
Dropout (0.3)
       ↓
Output Dense Layer (7 units, Softmax)
       ↓
Emotion Prediction (7 classes)
```

**Text Emotion Recognition (LSTM Based)**  
```
Input: Tokenized and padded text sequences
       ↓
Embedding Layer (100 dimensional embeddings)
       ↓
LSTM Layer (64 units, return sequences)
       ↓
LSTM Layer (32 units)
       ↓
Dense Layer (64 units, ReLU)
       ↓
Dropout (0.4)
       ↓
Output Dense Layer (multiple units, Softmax)
       ↓
Sentiment/Emotion Prediction
```

---

## 🔗 Multimodal Fusion Strategy

The fusion layer combines predictions from all three modalities using an intelligent weighted averaging approach:

**Fusion Process**  
1. Each model outputs confidence scores for its predicted emotions  
2. Confidence scores are normalized to ensure comparability  
3. A weighted average is computed based on each model's reliability  
4. The fused scores are converted back to emotion probabilities  
5. The final emotion is determined by the highest probability  

**Advantages of This Approach**  
- Balanced contribution from all modalities  
- Robustness to individual model failures  
- Interpretable results showing contribution of each modality  
- Easy to adjust weights based on domain knowledge or performance metrics  

**Fallback Mechanisms**  
- If audio is unavailable, facial and text models contribute equally  
- If facial data is missing, audio and text predictions are averaged  
- If text is absent, audio and facial combine to form the prediction  
- If only one modality is available, its prediction is used directly  

---

## 🔄 Inference and Real Time Processing

The inference pipeline handles live or batch inputs efficiently:

**Audio Processor**  
- Records or loads audio from file  
- Converts to 16 kHz mono format  
- Extracts MFCC features with 13 coefficients  
- Normalizes features for model input  
- Returns emotion prediction and confidence scores  

**Image Processor**  
- Captures or loads image from camera or file  
- Detects faces using OpenCV  
- Aligns and crops facial region  
- Resizes to 224×224 pixels for ResNet50V2  
- Normalizes pixel values  
- Returns emotion prediction for each detected face  

**Text Processor**  
- Accepts text input from user  
- Cleans and tokenizes text  
- Pads sequences to fixed length  
- Processes through LSTM model  
- Returns emotion or sentiment prediction  

---

## 🌐 Streamlit Web Interface

The Streamlit application provides an intuitive and friendly user experience:

**Key Features**  

**Welcome Section**  
- Warm greeting and explanation of the assistant  
- Clear guidance on how to use the system  
- Links to documentation and support resources  

**Input Section**  
- File uploader for audio files (WAV, MP3)  
- Camera widget for real time facial capture  
- Text input box for sentiment analysis  
- Options to select which modalities to analyze  

**Processing and Feedback**  
- Animated progress bar during processing  
- Real time status messages  
- Gentle confirmation when analysis is complete  
- Clear display of any errors with helpful suggestions  

**Results and Visualization**  
- Individual emotion predictions from each modality  
- Bar charts showing confidence scores  
- Overall fused emotion with explanation  
- Timestamp and session information  

**Interpretation Guide**  
- Simple explanation of what each emotion means  
- Suggestions for self reflection based on results  
- Reminder that results are supportive, not diagnostic  
- Option to save or export results  

**Design Philosophy**  
- Soft, calming color palette  
- Large, readable text for accessibility  
- Ample white space to avoid overwhelming users  
- Empathetic language throughout  
- Clear visual hierarchy  

---

## ▶️ How to Run the Project

**Prerequisites**  
- Python 3.8 or higher installed  
- 8 GB RAM minimum (16 GB recommended)  
- Webcam and microphone for real time features  
- Internet connection for initial setup  

**Step 1: Clone the Repository**  
```bash
git clone <your-repo-link>
cd MULTIMODAL_MENTAL_HEALTH_AI
```

**Step 2: Create Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate          # On Linux/Mac
# or
venv\Scripts\activate              # On Windows
```

**Step 3: Install Dependencies**  
```bash
pip install -r requirements.txt
```

**Step 4: Verify GPU Support (Optional)**  
```bash
python verify_gpu.py
```

**Step 5: Initialize Project**  
```bash
python initialize_project.py
```

**Step 6: Run the Streamlit Application**  
```bash
cd streamlit_app
streamlit run app.py
```

**Step 7: Access the Application**  
- Open your browser and navigate to the URL shown in terminal (typically http://localhost:8501)  
- Start interacting with the interface  

**To Collect Data (Optional)**  
```bash
python collect_all_data.py
```

**To Train Models**  
```bash
python src/training/train_audio_model.py
python src/training/train_facial_model.py
python src/training/train_text_model.py
```

---

## 🔒 Ethical and Privacy Considerations

This project is built with a strong commitment to ethical AI and user privacy:

**Design Principles**  
- Support, not replacement: The system is designed to support emotional awareness, never to replace professional mental health care  
- No diagnosis or treatment: Results are informative only and should not be interpreted as medical advice  
- Transparency: Users are informed about how the system works and what data is processed  
- User consent: Only process data that users explicitly provide  

**Privacy Safeguards**  
- Data is processed locally whenever possible  
- No data is stored without explicit user consent  
- When data is collected, it is anonymized and secured  
- Clear privacy policy and data handling practices  
- Regular security audits and updates  

**Responsible Use**  
- Encourage users to seek professional help when needed  
- Provide crisis resources and support information  
- Avoid language that overstates the system's capabilities  
- Continuous monitoring for biases or harmful outputs  

---

## 🚀 Future Enhancements

**Model Improvements**  
- Implement attention mechanisms for better focus on relevant features  
- Explore ensemble methods combining multiple architectures  
- Add real time voice activity detection and speech emotion from prosody  
- Incorporate micro expression detection for subtle emotions  

**Interface Enhancements**  
- Add historical tracking and emotion trend visualization  
- Implement personalized recommendations based on patterns  
- Build journaling feature integrated with emotion tracking  
- Create guided meditation or breathing exercises triggered by stress detection  

**Deployment and Scaling**  
- Package application as Docker container for easy deployment  
- Optimize models for edge computing and mobile devices  
- Deploy on cloud platforms like AWS, Google Cloud, or Azure  
- Build REST API for integration with other applications  

**Extended Modalities**  
- Integrate wearable sensor data for physiological tracking  
- Add real time conversation analysis and response generation  
- Implement gesture recognition for non verbal cues  
- Support multi language emotion recognition  

---

## 👨‍💻 Author

**Tushar Sharma**  
AI and Deep Learning enthusiast passionate about building human centered AI systems that support mental health and emotional well being.

- GitHub: [@Tushar9422](https://github.com/Tushar9422)  
- Email: tusharsharma9422@gmail.com  
- LinkedIn: [Tushar Sharma](www.linkedin.com/in/tushar-squared)  

---

## 📞 Support and Contribution

If you find this project helpful, please consider:

- ⭐ Starring this repository on GitHub  
- 🐛 Reporting any issues or bugs  
- 💡 Suggesting features or improvements  
- 🤝 Contributing code or documentation  
- 💬 Providing feedback on the user experience  

For questions or suggestions, please open an issue on GitHub or reach out directly through email.

---

*Built with care and a deep commitment to mental health awareness and AI accessibility.*

*Remember: This assistant is here to support your emotional awareness and well being. If you are in crisis or need immediate help, please reach out to a mental health professional or crisis hotline.*
