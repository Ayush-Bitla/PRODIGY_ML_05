import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from calorie_mapping import CALORIE_DICT
# Add Keras/TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
import os
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load Food-101 class names for MobileNet from info JSON
with open('models/food101_mobilenet_MOBILENET_info.json', 'r') as f:
    mobilenet_info = json.load(f)
    FOOD101_CLASS_NAMES = mobilenet_info['classes']

# Hardcoded macronutrient values per 100g (grams)
MACRO_DICT = {
    'apple_pie': {'protein': 2.1, 'fat': 14.0, 'carbs': 41.0},
    'beef_carpaccio': {'protein': 20.0, 'fat': 4.0, 'carbs': 0.0},
    'bibimbap': {'protein': 3.0, 'fat': 2.0, 'carbs': 22.0},
    'cup_cakes': {'protein': 3.6, 'fat': 12.0, 'carbs': 46.0},
    'foie_gras': {'protein': 7.0, 'fat': 43.0, 'carbs': 2.0},
    'french_fries': {'protein': 3.4, 'fat': 15.0, 'carbs': 41.0},
    'garlic_bread': {'protein': 7.0, 'fat': 17.0, 'carbs': 44.0},
    'pizza': {'protein': 11.0, 'fat': 10.0, 'carbs': 33.0},
    'spring_rolls': {'protein': 3.0, 'fat': 5.0, 'carbs': 24.0},
    'spaghetti_carbonara': {'protein': 13.0, 'fat': 17.0, 'carbs': 44.0},
    'strawberry_shortcake': {'protein': 3.0, 'fat': 10.0, 'carbs': 40.0},
    'omelette': {'protein': 10.0, 'fat': 12.0, 'carbs': 1.0},
}

# Model settings
MODEL_OPTIONS = {
    'MobileNet Food-101 (NEW)': {
        'path': 'models/MOBILENET_best_model_food101_mobilenet.pt',
        'class_names': FOOD101_CLASS_NAMES
    }
}

# Model definition (must match training)
class ImprovedResNet(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.5):
        super(ImprovedResNet, self).__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_classes)
        )
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# Ultimate model architecture for Food-101 (from train.py)
class UltimateFood101Model(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.5):
        super(UltimateFood101Model, self).__init__()
        self.backbone = models.efficientnet_b4(weights='IMAGENET1K_V1')
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(512, n_classes)
        )
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

# MobileNet model architecture for Food-101
class MobileNetFood101Model(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.3):
        super(MobileNetFood101Model, self).__init__()
        self.backbone = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        num_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Sequential(*list(self.backbone.classifier.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, n_classes)
        )
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

@st.cache_resource
def load_trained_model(model_path, n_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Special case: handle ULTIMATE model as UltimateFood101Model
    if 'ULTIMATE_best_model_test_mini_run.pt' in model_path:
        model = UltimateFood101Model(n_classes=n_classes, dropout_rate=0.5)
        loaded = torch.load(model_path, map_location=device)
        if isinstance(loaded, dict) and 'model_state_dict' in loaded:
            model.load_state_dict(loaded['model_state_dict'])
        else:
            model.load_state_dict(loaded)
        model = model.to(device)
        model.eval()
        return model, device
    # Special case: handle MobileNet model
    if 'MOBILENET_best_model_food101_mobilenet.pt' in model_path:
        model = MobileNetFood101Model(n_classes=n_classes, dropout_rate=0.3)
        loaded = torch.load(model_path, map_location=device)
        if isinstance(loaded, dict) and 'model_state_dict' in loaded:
            model.load_state_dict(loaded['model_state_dict'])
        else:
            model.load_state_dict(loaded)
        model = model.to(device)
        model.eval()
        return model, device
    # Default: load ImprovedResNet from state_dict
    model = ImprovedResNet(n_classes=n_classes, dropout_rate=0.5)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model, device

# Add a function to load Keras model
@st.cache_resource
def load_keras_model(model_path):
    return load_model(model_path)

# Add a function for Keras preprocessing

def preprocess_image_keras(img: Image.Image):
    img = img.resize((224, 224))
    arr = np.array(img)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]  # Remove alpha if present
    arr = arr.astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = inception_preprocess(arr)
    return arr
# Restore PyTorch preprocessing (must match training)
def preprocess_image(img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

st.title("Food Calorie Estimator 🍕🥗🍰")
st.write("Upload a food image. The app will predict the food type and estimate its calories per 100g.")

# Model selection
model_info = MODEL_OPTIONS['MobileNet Food-101 (NEW)']

uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Image', use_container_width=True)
        st.write("Classifying...")

        # Detect if Keras model
        if model_info['path'].endswith('.h5'):
            model = load_keras_model(model_info['path'])
            x = preprocess_image_keras(img)
            preds = model.predict(x)
            class_idx = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]))
        else:
            model, device = load_trained_model(model_info['path'], len(model_info['class_names']))
            x = preprocess_image(img)
            x = x.to(device)
            with torch.no_grad():
                outputs = model(x)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                class_idx = int(np.argmax(probs))
                confidence = float(probs[class_idx])

        class_name = model_info['class_names'][class_idx]
        calories = CALORIE_DICT.get(class_name, "Unknown")
        st.success(f"**Prediction:** {class_name.replace('_', ' ').title()}")
        st.info(f"**Estimated Calories (per 100g):** {calories}")
        st.write(f"**Confidence:** {confidence:.2%}")
        st.write("Note: Calorie values are approximate and based on standard serving sizes.")

        # Macronutrient pie chart
        macros = MACRO_DICT.get(class_name)
        if macros:
            macro_cals = {
                'Protein': macros['protein'] * 4,
                'Fat': macros['fat'] * 9,
                'Carbohydrates': macros['carbs'] * 4
            }
            labels = list(macro_cals.keys())
            values = list(macro_cals.values())
            colors = ['#4F81BD', '#C0504D', '#9BBB59']
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors), hole=0.3)])
            fig.update_traces(textinfo='percent+label', pull=[0.05, 0.05, 0.05])
            fig.update_layout(title_text='Calorie Split: Protein vs Fat vs Carbohydrates (per 100g)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning('Macronutrient breakdown not available for this food.')
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")
