# Multimodal AI Mental Health Companion 🧠

A Streamlit app that provides empathetic emotional support by analyzing facial expressions (using ResNet18) and text input (using RoBERTa), with personalized responses powered by the Groq API. The app features a calming interface, chat functionality, and a sidebar with user instructions.

**Disclaimer**: This tool is not a substitute for professional mental health support. Always consult a licensed professional for medical advice.

---

## Features

- **Image Analysis**: Detects emotions (e.g., Happy, Sad) from uploaded images or webcam captures using a ResNet18 model.
- **Text Analysis**: Identifies mental health states (e.g., Anxiety, Stress) from text input using a RoBERTa model.
- **Empathetic Chat**: Continues conversations with tailored, supportive responses via the Groq API.
- **User-Friendly Design**: Calming gradient UI (`#1E1E2F` to `#2A2A4A`), speech bubble chat, green buttons, and a "Clear Chat" option.
- **Sidebar Instructions**: Guides users on how to use the app effectively.
- **Deployment**: Hosted on Hugging Face Spaces using Docker for seamless access.

## Project Structure

```
multimodal-ai-mental-health-companion/
├── app.py                 # Main Streamlit app script
├── requirements.txt       # Python dependencies
├── notebooks/             # Jupyter notebooks for development/experiments
│   ├── notebook1.ipynb    # [Describe purpose, e.g., Data Preprocessing]
│   ├── notebook2.ipynb    # [Describe purpose, e.g., Model Training]
├── .gitignore             # Ignores virtual environment files
└── README.md              # Project documentation

```

**Note**: Model weights (`resnet_model/`, `roberta_model/`) and `.env` (with `GROQ_API_KEY`) are not included in the repository to keep it lightweight. Models are downloaded at runtime from Hugging Face Hub.

## Tech Stack

- **Frontend**: Streamlit
- **Models**: ResNet18 (image analysis), RoBERTa (text analysis), Groq API (chat)
- **Dependencies**: PyTorch, Transformers, OpenCV, Plotly, Pandas, Python-dotenv, Groq
- **Deployment**: Hugging Face Spaces (Docker)

## Installation

1. **Clone the Repository**:
   ```
   git clone https://github.com/<your-username>/multimodal-ai-mental-health-companion.git
   cd multimodal-ai-mental-health-companion
   ```
2. **Set Up a Virtual Environment:**
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. **Install Dependencies:**
```
pip install -r requirements.txt
```
4. **Set Up Environment Variables:**

    - Create a .env file in the root directory:

```
touch .env

```
- Add your Groq API key (get it from Groq Console):
```
GROQ_API_KEY=your_api_key_here
```
### Run the App:
```
streamlit run app.py
```
### Usage
- Image Analysis:
    - Go to the Image Analysis tab.

    - Upload a PNG/JPEG/JPG image or use your webcam.

    - Click Predict to see detected emotions, confidence scores, and a personalized response.

    - Continue the conversation in the chat interface or click Clear Chat to reset.

### Text Analysis:
- Go to the Text Analysis tab.

- Enter your feelings in the text box.

- Click Get Response to view detected mental states and a supportive response.

- Use the chat interface to continue or Clear Chat to start over.

- Sidebar Instructions:
    - Check the sidebar for detailed usage instructions and tips.

