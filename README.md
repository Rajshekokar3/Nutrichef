Project Title
# Food Classification Web Application

# Description
This project is a web-based application designed for classifying food images using a pre-trained ResNet50 model. It utilizes convolutional neural networks (CNNs) for image processing and prediction. The application is implemented with Streamlit, providing an interactive interface for users to upload food images and get classification results. This project is intended for a final year academic requirement.

# Features
ResNet50 Architecture: Leveraged for robust feature extraction and accurate food classification.
Real-Time Image Upload: Upload food images in .jpg, .jpeg, or .png formats.
Web Deployment: Accessible via a user-friendly web interface using Streamlit.
Multi-Class Prediction: Trained to classify food items into 101 different categories.


# How It Works
  Upload an Image: The user uploads a food image through the application.

# Preprocessing:
  The image is resized to (224x224) pixels.
  Normalized for the ResNet50 model input.
  
# Prediction:
  The image is fed into the trained ResNet50 model.
  The model outputs the class label of the food item.

# Result Display: The classified food item is displayed on the web interface.
  Installation Guide
  Prerequisites
  Python 3.8+
  Virtual environment (recommended)
