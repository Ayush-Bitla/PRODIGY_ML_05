# Food Calorie Estimator 🍕🥗🍰

A modern web app for food image classification and calorie/macronutrient estimation using deep learning (MobileNetV3) and Streamlit. Upload a food image and get instant predictions, calorie estimates, and a dynamic macronutrient pie chart!

---

## 🎯 Project Overview

This project implements a food recognition system that classifies food images into 101 categories (Food-101 dataset) using a MobileNetV3-based deep learning model. The app provides calorie estimates and a dynamic breakdown of protein, fat, and carbohydrate calories per 100g serving. Built with PyTorch, TensorFlow/Keras, and Streamlit.

---

## 🎥 Demo Video

See the app in action:

![Food Calorie Estimator Demo](models/FoodCalorieEstimatorDemo.gif)

---

## 🖼️ Result

Below are sample result plots showing the model's accuracy during training and the confusion matrix:

![MobileNet Accuracy Curve](models/mobilenet_accuracy_curve.png)

![MobileNet Confusion Matrix](models/food101_mobilenet_MOBILENET_confusion_matrix.png)

---

## 🚀 Live Demo

Access the live Streamlit app here:  
👉 [Food Calorie Estimator App](https://food-calories-estimator.streamlit.app/)

---

## 🧠 Features

**Core ML Features**
- Deep Learning Model: MobileNetV3-Large, fine-tuned on Food-101 (101 classes)
- Calorie & Macronutrient Estimation: Per 100g for each food class
- Confidence Score: Shows model certainty for each prediction

**Application Features**
- Streamlit UI: Simple, interactive web interface
- Image Upload: Supports JPG, JPEG, PNG
- Dynamic Pie Chart: Interactive macronutrient breakdown (Plotly)
- Robust Error Handling: Handles missing models, unknown foods, and more

**Technical Features**
- Model Persistence: Save/load trained models (.pt, .h5)
- Cross-platform: Works on Windows, Mac, and Linux
- Visualization: Training curves, confusion matrix, and more

---

## 📦 Installation & Setup

**Prerequisites**
- Python 3.8+
- (Optional) CUDA GPU for faster inference

**Quick Start**

1. **Clone the project:**
   ```bash
   git clone https://github.com/Ayush-Bitla/PRODIGY_ML_05.git
   cd "Food Calorie Estimator"
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```
3. **Activate the virtual environment:**
   - Windows: `.venv\Scripts\activate`
   - Mac/Linux: `source .venv/bin/activate`
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
5. **Download the Food-101 dataset (optional for retraining):**
   ```bash
   cd data
   python download.py
   ```
6. **Run the app:**
   ```bash
   streamlit run app.py
   ```

---

## 🎮 How to Use

1. **Upload a food image** (JPG, JPEG, PNG)
2. **View prediction:**
   - Food class (e.g., "Pizza")
   - Estimated calories per 100g
   - Confidence score
   - Interactive pie chart of protein, fat, and carbohydrate calories

---

## 📊 Model Performance

- **Test Accuracy**: ~99.9% (on Food-101 test set)
- **Model**: MobileNetV3-Large, custom classifier head
- **Classes**: 101 food categories
- **Training Time**: ~8 hours (GPU)
- **Inference Time**: Instantaneous (on CPU/GPU)

**Model Architecture**
- Input Image (224x224x3)
    ↓
- MobileNetV3-Large backbone (pretrained on ImageNet)
    ↓
- Custom classifier head (Dropout, Dense layers)
    ↓
- Food class prediction

---

## 📁 Project Structure

```
Food Calorie Estimator/
├── app.py                      # Streamlit web app
├── model/
│   ├── train.py                # Model training script (PyTorch)
│   ├── train_keras_food101tiny.py # Keras training script (optional)
│   ├── predict.py              # Prediction utilities
│   ├── plot_mobilenet_accuracy.py # Plotting training curves
│   └── calorie_mapping.py      # Calorie/macronutrient data
├── models/
│   ├── MOBILENET_best_model_food101_mobilenet.pt  # Trained model (PyTorch)
│   ├── my_trained_food101tiny_model.h5            # Trained model (Keras)
│   ├── mobilenet_accuracy_curve.png               # Training accuracy curve
│   ├── food101_mobilenet_MOBILENET_confusion_matrix.png # Confusion matrix
│   ├── FoodCalorieEstimatorDemo.gif               # Demo GIF
│   └── food101_mobilenet_MOBILENET_info.json      # Class names, metrics
├── data/
│   ├── download.py             # Dataset download script
│   ├── prepare.py              # Data preparation
│   └── visualize.py            # Data visualization
├── requirements.txt            # Python dependencies
├── class_names.json            # Class names (legacy)
├── Food 101 dataset/           # Food-101 dataset (train/test)
├── img/                        # Sample images
├── README.md                   # This file
└── ...
```

---

## 🔧 Technical Details

**Dependencies**
- PyTorch: Deep learning framework (main model)
- TensorFlow/Keras: For .h5 model support (optional)
- Streamlit: Web app interface
- Plotly: Interactive charts
- OpenCV, Pillow: Image processing
- NumPy, scikit-learn, matplotlib, seaborn: Data processing & visualization

**Model Specifications**
- Input Size: 224x224 RGB images
- Classes: 101 food categories
- Regularization: Dropout, BatchNormalization
- Training: Adam optimizer, label smoothing, early stopping

---

## 📋 Calorie & Macronutrient Mapping

| Food Class         | Calories (per 100g) | Protein (g) | Fat (g) | Carbs (g) |
|--------------------|---------------------|-------------|---------|-----------|
| apple_pie          | 296                 | 2.1         | 14.0    | 41.0      |
| beef_carpaccio     | 120                 | 20.0        | 4.0     | 0.0       |
| bibimbap           | 112                 | 3.0         | 2.0     | 22.0      |
| cup_cakes          | 305                 | 3.6         | 12.0    | 46.0      |
| foie_gras          | 462                 | 7.0         | 43.0    | 2.0       |
| french_fries       | 312                 | 3.4         | 15.0    | 41.0      |
| garlic_bread       | 350                 | 7.0         | 17.0    | 44.0      |
| pizza              | 266                 | 11.0        | 10.0    | 33.0      |
| spring_rolls       | 154                 | 3.0         | 5.0     | 24.0      |
| spaghetti_carbonara| 380                 | 13.0        | 17.0    | 44.0      |
| strawberry_shortcake| 250                | 3.0         | 10.0    | 40.0      |
| omelette           | 154                 | 10.0        | 12.0    | 1.0       |
| ...                | ...                 | ...         | ...     | ...       |

---

## 🐛 Troubleshooting

**Common Issues**
- "No module named 'streamlit'":  
  Install with `pip install streamlit`
- Model loading error:  
  Ensure model files exist in the `models/` directory
- Low accuracy:  
  Retrain the model or check input image quality

**Performance Tips**
- Use a GPU for faster training/inference
- Ensure good lighting and clear images for best results

---

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving the model architecture
- Adding new food classes or calorie mappings

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- **Dataset**: [Food-101](https://www.kaggle.com/datasets/jayaprakashpondy/food-101-dataset)
- **Model Inspiration**: MobileNetV3, PyTorch community
- **UI**: Streamlit, Plotly 