# Import necessary libraries
from skimage.io import imread
from skimage.color import rgb2gray
import os
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from skimage.transform import rotate
import random
import gradio as gr
from skimage.util import random_noise
from skimage.exposure import adjust_gamma
import logging
import matplotlib.pyplot as plt
import io
from PIL import Image

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Function to load images and convert to grayscale
def load_images_from_folder(folder, label_name):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                img = rgb2gray(imread(os.path.join(folder, filename)))
                if img is not None:
                    images.append(img)
                    labels.append(label_name)
            except Exception as e:
                logging.warning(f"Failed to load image {filename}: {e}")
    return images, labels

# Load images from 'Grass' and 'Woods' folders
grass_images, grass_labels = load_images_from_folder('Grass', 0)
wood_images, wood_labels = load_images_from_folder('Woods', 1)

# Log the number of images loaded
logging.info(f"Loaded {len(grass_images)} grass images and {len(wood_images)} wood images.")

# Combine images and labels
images = grass_images + wood_images
labels = grass_labels + wood_labels

# Convert labels to numpy array
labels = np.array(labels)  # Keep images as a list

# Function to extract LBP features
def extract_lbp_features(image, radius, n_points):
    image_uint8 = (image * 255).astype('uint8')
    lbp = local_binary_pattern(image_uint8, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype('float')
    if hist.sum() != 0:
        hist /= hist.sum()  # Normalize the histogram
    return hist

# Function to extract GLCM features
def extract_glcm_features(image, distances, angles, levels):
    image_uint8 = (image * (levels - 1)).astype('uint8')
    glcm = graycomatrix(image_uint8, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)
    # Extract statistical properties
    contrast = graycoprops(glcm, 'contrast').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    # Concatenate features
    features = np.hstack([contrast, correlation, energy, homogeneity])
    return features

# Data augmentation functions
def augment_image_LBP(image):
    if random.choice([True, False]):
        angle = random.choice([90, 180, 270])
        image = rotate(image, angle)
    if random.choice([True, False]):
        image = np.fliplr(image) if random.choice([True, False]) else np.flipud(image)
    return image

def augment_image_GLCM(image):
    # Adjust gamma
    if random.choice([True, False]):
        gamma = random.uniform(0.9, 1.1)
        image = adjust_gamma(image, gamma=gamma)
    # Add Gaussian noise
    if random.choice([True, False]):
        image = random_noise(image, mode='gaussian', var=0.005)
    return image

# Function to extract features and augment data
def extract_features_and_augment(images, labels, feature_type, feature_params):
    features = []
    augmented_labels = []
    for image, label in zip(images, labels):
        if feature_type == "LBP":
            feature = extract_lbp_features(image, **feature_params)
        elif feature_type == "GLCM":
            feature = extract_glcm_features(image, **feature_params)
        features.append(feature)
        augmented_labels.append(label)

        # Generate augmented versions
        for _ in range(2):
            if feature_type == "LBP":
                augmented_image = augment_image_LBP(image)
                feature = extract_lbp_features(augmented_image, **feature_params)
            elif feature_type == "GLCM":
                augmented_image = augment_image_GLCM(image)
                feature = extract_glcm_features(augmented_image, **feature_params)
            features.append(feature)
            augmented_labels.append(label)
    return np.array(features), np.array(augmented_labels)

# Function to train and evaluate the classifier with provided parameters
def train_and_evaluate(feature_type):
    # Use the best feature parameters obtained from experiments from previous version
    if feature_type == "LBP":
        best_params_feature = {'radius': 1, 'n_points': 24}
        best_params_svm = {'C': 10, 'kernel': 'rbf'}
    elif feature_type == "GLCM":
        best_params_feature = {
            'distances': [2],
            'angles': [0, np.pi / 4, np.pi / 2],
            'levels': 32
        }
        best_params_svm = {'C': 10, 'kernel': 'rbf'}

    # Extract features with the best feature parameters
    X, y_augmented = extract_features_and_augment(images, labels, feature_type, best_params_feature)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_augmented, test_size=0.3, random_state=42, stratify=y_augmented
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the SVM model with best parameters
    svm = SVC(**best_params_svm, random_state=42, probability=True)
    svm.fit(X_train_scaled, y_train)

    # Evaluate on test set
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    # Print best parameters and performance
    print(f"--- Best {feature_type} Model ---")
    print(f"Best Feature Parameters: {best_params_feature}")
    print(f"Best SVM Parameters: {best_params_svm}")
    print(f"Test Accuracy: {accuracy:.2f}")

    # Evaluate the model on the test set
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")

    return svm, scaler, best_params_feature, best_params_svm

# Train and evaluate models for LBP and GLCM
svm_model_lbp, scaler_lbp, best_params_feature_lbp, best_params_svm_lbp = train_and_evaluate("LBP")
svm_model_glcm, scaler_glcm, best_params_feature_glcm, best_params_svm_glcm = train_and_evaluate("GLCM")

# Function to classify texture and provide visual feedback
def classify_texture_gradio(image, feature_type):
    # Check if image is None
    if image is None:
        return None, gr.update(value="Error: Please upload an image.", visible=True)

    # Check if feature_type is valid
    if feature_type not in ["LBP", "GLCM"]:
        return None, gr.update(value="Error: Please select a feature extraction method.", visible=True)

    # Log the shape of the input image
    logging.info(f"Input image shape: {image.shape}")

    # Check the number of dimensions
    if image.ndim == 2:
        # Image is already in grayscale
        image_gray = image
    elif image.ndim == 3:
        # Image is RGB; convert to grayscale
        image_gray = rgb2gray(image)
    else:
        return None, gr.update(value="Error: Invalid image format.", visible=True)

    # Ensure the image has valid pixel values
    if not np.isfinite(image_gray).all():
        return None, gr.update(value="Error: Image contains invalid pixel values.", visible=True)

    try:
        # Proceed with feature extraction and classification
        if feature_type == "LBP":
            features = extract_lbp_features(image_gray, **best_params_feature_lbp).reshape(1, -1)
            features_scaled = scaler_lbp.transform(features)
            prediction = svm_model_lbp.predict(features_scaled)
            confidence = svm_model_lbp.decision_function(features_scaled)
        elif feature_type == "GLCM":
            features = extract_glcm_features(image_gray, **best_params_feature_glcm).reshape(1, -1)
            features_scaled = scaler_glcm.transform(features)
            prediction = svm_model_glcm.predict(features_scaled)
            confidence = svm_model_glcm.decision_function(features_scaled)
        else:
            return None, gr.update(value="Error: Invalid feature extraction method.", visible=True)
    except Exception as e:
        logging.error(f"Error during feature extraction or classification: {e}")
        return None, gr.update(value=f"An error occurred: {e}", visible=True)

    # Map prediction to label
    label = "Grass" if prediction[0] == 0 else "Wood"
    confidence_score = confidence[0]

    # Annotate image
    plt.figure(figsize=(5, 5))
    if image.ndim == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.title(f"Prediction: {label}\nConfidence Score: {confidence_score:.2f}")
    plt.axis('off')

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Open the buffer image with PIL and return
    result_image = Image.open(buf)
    return result_image, gr.update(value="", visible=False)  # Hide the message textbox when no error

# Update the Gradio interface
interface = gr.Interface(
    fn=classify_texture_gradio,
    inputs=[
        gr.Image(type="numpy", label="Upload Image"),
        gr.Radio(
            choices=["LBP", "GLCM"],
            label="Feature Extraction Method",
            value="LBP"
        )
    ],
    outputs=[
        gr.Image(type="pil", label="Classification Result"),
        gr.Textbox(label="Message", lines=2, visible=False)  # Initialize as hidden
    ],
    title="Texture Classification: Grass or Wood",
    description="Upload an image and select the feature extraction method to classify it.",
    allow_flagging='never'
)

# Launch Gradio interface with shareable link
interface.launch(share=True)