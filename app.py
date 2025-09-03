import os
import logging
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

FACIAL_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
MENTAL_CLASSES = ['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Personality disorder', 'Stress', 'Suicidal']
RESNET_MODEL_PATH = "resnet_model/facial_expression_resnet.pth"
ROBERTA_MODEL_DIR = "roberta_model"

def safe_to_device(model, device):
    try:
        if any(p.is_meta for p in model.parameters()):
            logger.warning("Model contains meta tensors. Moving to device with to_empty.")
            model.to_empty(device=device)
            for p in model.parameters():
                if p.is_meta:
                    p.data = torch.zeros_like(p, device=device)
        else:
            model.to(device)
    except Exception as e:
        st.error(f"Error moving model to device: {str(e)}")
        st.stop()

def load_resnet18(model_path):
    try:
        if not os.path.exists(model_path):
            st.warning(f"ResNet18 model not found at {model_path}. Image analysis disabled.")
            return None
        resnet18 = models.resnet18(weights='IMAGENET1K_V1')
        resnet18.fc = torch.nn.Linear(resnet18.fc.in_features, len(FACIAL_CLASSES))
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        logger.info(f"State dict keys: {list(state_dict.keys())[:5]}")
        new_state_dict = {key.replace("resnet.", ""): value for key, value in state_dict.items()}
        missing_keys, unexpected_keys = resnet18.load_state_dict(new_state_dict, strict=False)
        if missing_keys or unexpected_keys:
            st.warning(f"State dict issues: Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
        if any(p.is_meta for p in resnet18.parameters()):
            st.error("ResNet18 model contains meta tensors after loading weights. Check the checkpoint file.")
            return None
        resnet18.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        safe_to_device(resnet18, device)
        test_image = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            test_output = resnet18(test_image)
            test_probs = torch.softmax(test_output, dim=1).tolist()[0]
            if max(test_probs) - min(test_probs) < 1e-4:
                st.warning(
                    f"ResNet18 model produces uniform probabilities ({test_probs[0]:.2f} for all classes). "
                    f"Model may be untrained. Fine-tune it for {len(FACIAL_CLASSES)} facial expression classes."
                )
        return resnet18
    except Exception as e:
        st.warning(f"Error loading ResNet18 model: {str(e)}. Image analysis disabled.")
        return None

def load_roberta(model_dir):
    try:
        required_files = ["config.json", "vocab.json", "merges.txt", "model.safetensors"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
        if missing_files:
            st.error(
                f"Missing files in {model_dir}: {', '.join(missing_files)}. "
                f"Please re-save the model using:\n"
                "from transformers import RobertaForSequenceClassification, RobertaTokenizer\n"
                "model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=7)\n"
                "tokenizer = RobertaTokenizer.from_pretrained('roberta-base')\n"
                "os.makedirs('roberta_model', exist_ok=True)\n"
                "model.save_pretrained('roberta_model')\n"
                "tokenizer.save_pretrained('roberta_model')"
            )
            st.stop()
        config = RobertaConfig.from_pretrained(model_dir)
        logger.info(f"RoBERTa config: {config}")
        if config.num_labels != len(MENTAL_CLASSES):
            st.error(
                f"RoBERTa model has {config.num_labels} classes, expected {len(MENTAL_CLASSES)}. "
                f"Please re-save the model with num_labels=7 as shown above."
            )
            st.stop()
        if hasattr(config, 'type_vocab_size') and config.type_vocab_size < 1:
            st.error(
                f"Invalid type_vocab_size in config.json: {config.type_vocab_size}. Expected >= 1. "
                f"Please re-save the model as shown above."
            )
            st.stop()
        tokenizer = RobertaTokenizer.from_pretrained(model_dir)
        roberta = RobertaForSequenceClassification.from_pretrained(model_dir, num_labels=len(MENTAL_CLASSES))
        if any(p.is_meta for p in roberta.parameters()):
            st.error("RoBERTa model contains meta tensors after loading. Check model files in 'roberta_model'.")
            st.stop()
        test_input = tokenizer("Test input", return_tensors="pt", truncation=True, padding=True, max_length=512)
        if 'token_type_ids' not in test_input:
            test_input['token_type_ids'] = torch.zeros_like(test_input['input_ids'], dtype=torch.long)
        test_input = {k: v.to(roberta.device) for k, v in test_input.items()}
        with torch.no_grad():
            test_output = roberta(**test_input)
            test_probs = torch.softmax(test_output.logits, dim=1).tolist()[0]
            if max(test_probs) - min(test_probs) < 1e-4:
                st.warning(
                    f"RoBERTa model produces uniform probabilities ({test_probs[0]:.2f} for all classes). "
                    f"Model may be untrained. Please fine-tune the model for {len(MENTAL_CLASSES)} classes:\n"
                    "1. Use a dataset with labeled mental health states (e.g., text labeled with Anxiety, Depression, etc.).\n"
                    "2. Fine-tune with: model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=7)\n"
                    "3. Save fine-tuned model: model.save_pretrained('roberta_model')"
                )
        roberta.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        safe_to_device(roberta, device)
        logger.info(f"RoBERTa loaded from {model_dir}. Files: {os.listdir(model_dir)}")
        return roberta, tokenizer
    except Exception as e:
        st.error(f"Error loading RoBERTa model: {str(e)}")
        st.stop()

def preprocess_image(image):
    try:
        if image is None:
            raise ValueError("No image provided")
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (224, 224))
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0)
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_facial_expression(image, model):
    try:
        if model is None:
            raise ValueError("ResNet18 model not loaded")
        with torch.no_grad():
            image = preprocess_image(image)
            if image is None:
                raise ValueError("Image preprocessing failed")
            device = next(model.parameters()).device
            output = model(image.to(device))
            output = torch.clamp(output, min=-10.0, max=10.0)
            output = torch.nan_to_num(output, nan=0.0, posinf=1e5, neginf=-1e5)
            probabilities = torch.softmax(output, dim=1)
            if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
                st.warning("Invalid probabilities in facial expression analysis. Model may be untrained.")
                return FACIAL_CLASSES[0], [1.0 / len(FACIAL_CLASSES)] * len(FACIAL_CLASSES)
            if max(probabilities.tolist()[0]) - min(probabilities.tolist()[0]) < 1e-4:
                st.warning("Uniform probabilities in facial expression analysis. Model may be untrained.")
            predicted_class = FACIAL_CLASSES[torch.argmax(probabilities, dim=1).item()]
            return predicted_class, probabilities.tolist()[0]
    except Exception as e:
        st.error(f"Error predicting facial expression: {str(e)}")
        return None, None

def predict_mental_health_state(text, model, tokenizer):
    try:
        if not text.strip():
            raise ValueError("Empty text input provided")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        if 'token_type_ids' not in inputs:
            inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'], dtype=torch.long)
        if inputs['token_type_ids'].max().item() >= model.config.type_vocab_size:
            raise ValueError(
                f"token_type_ids contains invalid indices (max: {inputs['token_type_ids'].max().item()}, "
                f"expected < {model.config.type_vocab_size}). Re-save the model with correct tokenizer."
            )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logger.info(f"RoBERTa output logits shape: {outputs.logits.shape}")
            if outputs.logits.shape[1] != len(MENTAL_CLASSES):
                st.error(
                    f"RoBERTa output has {outputs.logits.shape[1]} classes, expected {len(MENTAL_CLASSES)}. "
                    f"Please re-save the model with num_labels=7."
                )
                st.stop()
            probabilities = torch.softmax(outputs.logits, dim=1)
            if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
                st.warning("Invalid probabilities in text analysis. Model may be untrained.")
                return MENTAL_CLASSES[0], [1.0 / len(MENTAL_CLASSES)] * len(MENTAL_CLASSES)
            if max(probabilities.tolist()[0]) - min(probabilities.tolist()[0]) < 1e-4:
                st.warning("Uniform probabilities in text analysis. Model may be untrained.")
            predicted_class = MENTAL_CLASSES[torch.argmax(probabilities, dim=1).item()]
            return predicted_class, probabilities.tolist()[0]
    except Exception as e:
        st.error(f"Error predicting mental health state: {str(e)}")
        return None, None

def load_grok():
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        client = Groq(api_key=groq_api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing Grok client: {str(e)}")
        st.stop()

def generate_image_response(image, resnet_model, grok_client):
    try:
        if resnet_model is None:
            raise ValueError("ResNet18 model not available")
        facial_emotion, facial_probs = predict_facial_expression(image, resnet_model)
        if facial_emotion is None or facial_probs is None:
            raise ValueError("Facial expression prediction failed")
        confidence = max(facial_probs) if facial_probs else 0.0
        prompt = f"""
        You are an empathetic mental health companion. A user has provided an image with:
        - Facial expression: {facial_emotion or 'Not provided'}
        - Confidence: {confidence:.2f}
        Respond with a supportive, empathetic message, describing the detected emotional state and suggesting personalized coping strategies. Ensure responses are safe, encouraging, and avoid medical advice.
        """
        response = generate_grok_response(prompt, grok_client)
        return response, facial_emotion, facial_probs
    except Exception as e:
        st.error(f"Error generating image response: {str(e)}")
        return None, None, None

def generate_text_response(text, roberta_model, roberta_tokenizer, grok_client):
    try:
        mental_state, text_probs = predict_mental_health_state(text, roberta_model, roberta_tokenizer)
        if mental_state is None or text_probs is None:
            raise ValueError("Mental health state prediction failed")
        confidence = max(text_probs) if text_probs else 0.0
        prompt = f"""
        You are an empathetic mental health companion. A user has provided:
        - Text analysis: {mental_state or 'Not provided'}
        - Confidence: {confidence:.2f}
        - User input: "{text}"
        Respond with a supportive, empathetic message, describing the detected mental health state and suggesting personalized coping strategies. Ensure responses are safe, encouraging, and avoid medical advice.
        """
        response = generate_grok_response(prompt, grok_client)
        return response, mental_state, text_probs
    except Exception as e:
        st.error(f"Error generating text response: {str(e)}")
        return None, None, None

def generate_grok_response(prompt, client, conversation_history=None, model_name="meta-llama/llama-4-scout-17b-16e-instruct", max_tokens=1024):
    try:
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": prompt})
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=1,
            max_completion_tokens=max_tokens,
            top_p=1,
            stream=True
        )
        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""
        return response
    except Exception as e:
        st.error(f"Error generating Grok response: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Multimodal AI Mental Health Companion", page_icon="🧠", layout="wide")
    st.title("🌿 Multimodal AI Mental Health Companion")
    st.markdown("""
    Welcome to **Multimodal AI Mental Health Companion**, a supportive AI tool designed to help you reflect on your emotions through image and text analysis. 
    This app is here to listen and offer gentle, encouraging guidance. 
    **Please note**: This tool is not a substitute for professional mental health support.
    """)

    # User Instructions in Sidebar
    with st.sidebar.expander("📖 How to Use Multimodal AI Mental Health Companion", expanded=True):
        st.markdown("""
        **Multimodal AI Mental Health Companion** offers two ways to explore your emotions: **Image Analysis** and **Text Analysis**. Follow these steps to get started:

        ### Image Analysis
        1. **Upload an Image**: Use the file uploader to select a PNG, JPEG, or JPG image from your device.
        2. **Use Your Webcam**: Click the "Use Webcam" button to capture a live photo (ensure your webcam is enabled).
        3. **Predict Emotions**: Click the "Predict" button to analyze your facial expression. You'll see the detected emotion and a confidence score.
        4. **Receive Support**: Read the personalized, empathetic response in the "Personalized Support" section.
        5. **Continue the Conversation**: Use the chat box below to share more about how you're feeling and receive ongoing support.
        6. **Clear Chat**: Click "Clear Chat" to start a new conversation if needed.

        ### Text Analysis
        1. **Share Your Feelings**: Type how you're feeling in the text box provided.
        2. **Get a Response**: Click the "Get Response" button to analyze your text. You'll see the detected mental state and a confidence score.
        3. **Receive Support**: Read the personalized response in the "Personalized Support" section.
        4. **Continue the Conversation**: Use the chat box to keep talking and receive further guidance.
        5. **Clear Chat**: Click "Clear Chat" to start a new conversation if needed.

        ### Tips for Best Results
        - **Images**: Ensure the image is clear and well-lit for accurate facial expression analysis.
        - **Text**: Be open and descriptive to help the AI understand your emotions better.
        - **Chat**: Use the chat feature to dive deeper into your feelings or ask for more coping strategies.
        - **Privacy**: Your inputs are processed securely, but avoid sharing sensitive personal information.

        **Disclaimer**: Multimodal AI Mental Health Companion is designed to offer emotional support and encouragement. For professional help, please consult a licensed mental health professional.
        """)

    st.markdown("""
    <style>
    .stApp { 
        background: linear-gradient(to bottom right, #1E1E2F, #2A2A4A); 
        color: #E6E6FA; 
        font-family: 'Inter', sans-serif; 
    }
    .stButton>button { 
        background: linear-gradient(to right, #4CAF50, #45A049); 
        color: white; 
        border-radius: 10px; 
        padding: 12px 24px; 
        font-weight: 600; 
        border: none; 
        transition: transform 0.2s ease; 
    }
    .stButton>button:hover { 
        transform: scale(1.05); 
        background: linear-gradient(to right, #45A049, #3D8B40); 
    }
    .stTextArea textarea { 
        background-color: #2A2A4A; 
        color: #E6E6FA; 
        border: 2px solid #4B4B6F; 
        border-radius: 10px; 
        padding: 12px; 
        font-size: 16px; 
    }
    .stFileUploader label { 
        color: #E6E6FA; 
        font-weight: 500; 
    }
    .stFileUploader div[role='button'] { 
        background: linear-gradient(to right, #4CAF50, #45A049); 
        color: white; 
        border-radius: 10px; 
        padding: 10px; 
    }
    .support-response { 
        background-color: #2A2A4A; 
        padding: 20px; 
        border-radius: 10px; 
        border: 2px solid #4B4B6F; 
        color: #E6E6FA; 
        font-size: 16px; 
        line-height: 1.6; 
    }
    .chat-message-user { 
        background-color: #4B4B6F; 
        padding: 12px; 
        margin: 8px 0; 
        border-radius: 15px 15px 15px 5px; 
        color: #E6E6FA; 
        max-width: 70%; 
        margin-left: auto; 
        font-size: 15px; 
    }
    .chat-message-grok { 
        background-color: #3D3D5C; 
        padding: 12px; 
        margin: 8px 0; 
        border-radius: 15px 15px 5px 15px; 
        color: #E6E6FA; 
        max-width: 70%; 
        margin-right: auto; 
        font-size: 15px; 
    }
    .stExpander { 
        background-color: #2A2A4A; 
        border-radius: 10px; 
        border: 2px solid #4B4B6F; 
    }
    .stExpander div[role='button'] { 
        color: #E6E6FA; 
        font-weight: 600; 
    }
    h1, h2, h3 { 
        color: #D8BFD8; 
        font-weight: 700; 
    }
    .stSidebar .stExpander { 
        background-color: #2A2A4A; 
        border-radius: 10px; 
        border: 2px solid #4B4B6F; 
        color: #E6E6FA; 
    }
    .stSidebar .stExpander div[role='button'] { 
        color: #E6E6FA; 
        font-weight: 600; 
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state for conversation history
    if 'image_chat_history' not in st.session_state:
        st.session_state.image_chat_history = []
    if 'text_chat_history' not in st.session_state:
        st.session_state.text_chat_history = []

    resnet_model = None
    try:
        if os.path.exists(RESNET_MODEL_PATH):
            resnet_model = load_resnet18(RESNET_MODEL_PATH)
    except Exception as e:
        st.warning(f"Failed to load ResNet18 model: {str(e)}. Image analysis disabled.")

    try:
        roberta_model, roberta_tokenizer = load_roberta(ROBERTA_MODEL_DIR)
    except Exception as e:
        st.error(f"Failed to load RoBERTa model: {str(e)}")
        st.stop()

    try:
        grok_client = load_grok()
    except Exception as e:
        st.error(f"Failed to initialize Grok client: {str(e)}")
        st.stop()

    tab1, tab2 = st.tabs(["📸 Image Analysis", "✍️ Text Analysis"])

    with tab1:
        st.subheader("Analyze Your Facial Expression")
        uploaded_image = st.file_uploader("Upload an image (PNG, JPEG, JPG)", type=["png", "jpeg", "jpg"])
        image = None
        if uploaded_image:
            try:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

        if st.button("Use Webcam"):
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    raise ValueError("Webcam not accessible")
                ret, frame = cap.read()
                if ret:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    st.image(image, caption="Webcam Image", use_column_width=True)
                else:
                    st.error("Failed to capture webcam image")
                cap.release()
            except Exception as e:
                st.error(f"Webcam error: {str(e)}")

        if st.button("Predict", key="predict_image"):
            if image is None:
                st.error("Please upload an image or use the webcam.")
            elif resnet_model is None:
                st.warning("Image analysis disabled: ResNet18 model not found.")
            else:
                # Reset chat history for new prediction
                st.session_state.image_chat_history = []
                response, facial_emotion, facial_probs = generate_image_response(image, resnet_model, grok_client)
                if response and facial_emotion and facial_probs:
                    st.subheader("Facial Expression Results")
                    st.write(f"**Detected Emotion**: {facial_emotion} (Confidence: {max(facial_probs):.2f})")
                    fig = px.bar(x=FACIAL_CLASSES, y=facial_probs, title="Facial Emotion Probabilities",
                                 labels={'x': 'Emotion', 'y': 'Probability'}, color_discrete_sequence=['#4CAF50'])
                    st.plotly_chart(fig, use_container_width=True)
                    st.subheader("Personalized Support")
                    st.markdown(f"<div class='support-response'>{response}</div>", unsafe_allow_html=True)
                    # Add initial response to chat history
                    st.session_state.image_chat_history.append({"role": "assistant", "content": response})

        # Chat interface for image analysis
        if st.session_state.image_chat_history:
            st.subheader("💬 Continue the Conversation")
            # Display chat history
            for message in st.session_state.image_chat_history:
                if message["role"] == "user":
                    st.markdown(f"<div class='chat-message-user'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-message-grok'><strong>Companion:</strong> {message['content']}</div>", unsafe_allow_html=True)
            # Clear chat button
            if st.button("Clear Chat", key="clear_image_chat"):
                st.session_state.image_chat_history = []
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
            # Chat input
            chat_input = st.text_input("Your message:", key="image_chat_input", placeholder="Type your message here...")
            if st.button("Send", key="image_chat_send"):
                if chat_input.strip():
                    st.session_state.image_chat_history.append({"role": "user", "content": chat_input})
                    prompt = f"""
                    You are an empathetic mental health companion. The user has provided a follow-up message: "{chat_input}".
                    Continue the conversation based on the previous context, maintaining an empathetic and supportive tone.
                    Ensure responses are safe, encouraging, and avoid medical advice.
                    """
                    chat_response = generate_grok_response(prompt, grok_client, conversation_history=st.session_state.image_chat_history)
                    if chat_response:
                        st.session_state.image_chat_history.append({"role": "assistant", "content": chat_response})
                    else:
                        st.error("Failed to generate a response. Please try again.")
                    # Refresh to display new message
                    try:
                        st.rerun()
                    except AttributeError:
                        st.experimental_rerun()

    with tab2:
        st.subheader("Share Your Feelings")
        user_text = st.text_area("Describe how you're feeling:", placeholder="Type here...", height=150)
        if st.button("Get Response", key="get_response_text"):
            if not user_text.strip():
                st.error("Please enter some text.")
            else:
                # Reset chat history for new prediction
                st.session_state.text_chat_history = []
                response, mental_state, text_probs = generate_text_response(user_text, roberta_model, roberta_tokenizer, grok_client)
                if response and mental_state and text_probs:
                    st.subheader("Text Analysis Results")
                    st.write(f"**Detected Mental State**: {mental_state} (Confidence: {max(text_probs):.2f})")
                    fig = px.bar(x=MENTAL_CLASSES, y=text_probs, title="Mental Health State Probabilities",
                                 labels={'x': 'Mental State', 'y': 'Probability'}, color_discrete_sequence=['#4CAF50'])
                    st.plotly_chart(fig, use_container_width=True)
                    st.subheader("Personalized Support")
                    st.markdown(f"<div class='support-response'>{response}</div>", unsafe_allow_html=True)
                    # Add initial response to chat history
                    st.session_state.text_chat_history.append({"role": "assistant", "content": response})

        # Chat interface for text analysis
        if st.session_state.text_chat_history:
            st.subheader("💬 Continue the Conversation")
            # Display chat history
            for message in st.session_state.text_chat_history:
                if message["role"] == "user":
                    st.markdown(f"<div class='chat-message-user'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-message-grok'><strong>Companion:</strong> {message['content']}</div>", unsafe_allow_html=True)
            # Clear chat button
            if st.button("Clear Chat", key="clear_text_chat"):
                st.session_state.text_chat_history = []
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
            # Chat input
            chat_input = st.text_input("Your message:", key="text_chat_input", placeholder="Type your message here...")
            if st.button("Send", key="text_chat_send"):
                if chat_input.strip():
                    st.session_state.text_chat_history.append({"role": "user", "content": chat_input})
                    prompt = f"""
                    You are an empathetic mental health companion. The user has provided a follow-up message: "{chat_input}".
                    Continue the conversation based on the previous context, maintaining an empathetic and supportive tone.
                    Ensure responses are safe, encouraging, and avoid medical advice.
                    """
                    chat_response = generate_grok_response(prompt, grok_client, conversation_history=st.session_state.text_chat_history)
                    if chat_response:
                        st.session_state.text_chat_history.append({"role": "assistant", "content": chat_response})
                    else:
                        st.error("Failed to generate a response. Please try again.")
                    # Refresh to display new message
                    try:
                        st.rerun()
                    except AttributeError:
                        st.experimental_rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.write("Please try again or contact support.")