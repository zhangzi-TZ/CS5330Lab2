# Import necessary libraries
from skimage.io import imread
from skimage.color import rgb2gray
import os
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
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


# Load images and labels
def load_images_from_folder(folder, label_name):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                img = rgb2gray(imread(os.path.join(folder, filename)))
                if img is not None:
                    images.append(img)
                    labels.append(label_name)  # Use the provided label
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


# Function to extract features without augmentation
def extract_features(images, feature_type, feature_params):
    features = []
    for image in images:
        if feature_type == "LBP":
            feature = extract_lbp_features(image, **feature_params)
        elif feature_type == "GLCM":
            feature = extract_glcm_features(image, **feature_params)
        features.append(feature)
    return np.array(features)


# Function to augment images
def augment_images(images, feature_type):
    augmented_images = []
    for image in images:
        for _ in range(2):  # Number of augmentations per image
            if feature_type == "LBP":
                augmented_image = augment_image_LBP(image)
            elif feature_type == "GLCM":
                augmented_image = augment_image_GLCM(image)
            augmented_images.append(augmented_image)
    return augmented_images

# train with nested cross-validation
def train_and_evaluate(feature_type):
    # Define parameter grids for feature extraction
    if feature_type == "LBP":
        param_grid_feature = {
            'radius': [1, 2, 3],
            'n_points': [8, 16, 24]
        }
    elif feature_type == "GLCM":
        param_grid_feature = {
            'distances': [[1], [2], [1, 2]],
            'angles': [[0], [0, np.pi/4], [0, np.pi/4, np.pi/2]],
            'levels': [32, 64, 256]
        }

    # Define parameter grid for SVM
    param_grid_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }

    # Prepare to store the results
    outer_scores = []
    best_params_feature = None
    best_params_svm = None
    best_model = None
    best_scaler = None

    y_labels = labels

    # Prepare for outer cross-validation
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, test_index in outer_cv.split(images, y_labels):
        # Split data into training and test sets
        images_train = [images[i] for i in train_index]
        images_test = [images[i] for i in test_index]
        y_train = y_labels[train_index]
        y_test = y_labels[test_index]

        # Apply data augmentation to training images
        augmented_images = augment_images(images_train, feature_type)
        # Combine original and augmented images
        images_train_augmented = images_train + augmented_images
        y_train_augmented = np.concatenate([y_train, np.repeat(y_train, 2)])

        # Grid search over feature parameters
        from itertools import product

        keys_feature = param_grid_feature.keys()
        values_feature = param_grid_feature.values()
        best_inner_score = 0

        for feature_params_values in product(*values_feature):
            feature_params = dict(zip(keys_feature, feature_params_values))

            # Extract features with current feature_params for training data
            X_train = extract_features(images_train_augmented, feature_type, feature_params)
            X_test = extract_features(images_test, feature_type, feature_params)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Inner cross-validation for hyperparameter tuning
            svm = SVC(random_state=42)
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                svm, param_grid_svm, cv=inner_cv, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train_augmented)

            # Get best estimator from inner CV
            inner_best_model = grid_search.best_estimator_
            inner_best_params = grid_search.best_params_
            inner_score = grid_search.best_score_

            # Evaluate on the outer test set
            y_pred = inner_best_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            # Keep track of the best parameters and model
            if accuracy > best_inner_score:
                best_inner_score = accuracy
                best_params_feature_fold = feature_params
                best_params_svm_fold = inner_best_params
                best_model_fold = inner_best_model
                best_scaler_fold = scaler
                best_X_test_scaled_fold = X_test_scaled
                best_y_test_fold = y_test

        # Append the best score from this outer fold
        outer_scores.append(best_inner_score)

        # Update the best overall model if current fold is better
        if best_inner_score > np.mean(outer_scores):
            best_params_feature = best_params_feature_fold
            best_params_svm = best_params_svm_fold
            best_model = best_model_fold
            best_scaler = best_scaler_fold

    # After cross-validation, print the average performance
    print(f"--- Nested Cross-Validation Results for {feature_type} ---")
    print(f"Average Accuracy: {np.mean(outer_scores):.2f} ± {np.std(outer_scores):.2f}")
    print(f"Best Feature Parameters: {best_params_feature}")
    print(f"Best SVM Parameters: {best_params_svm}")

    # Retrain the best model on the entire dataset
    # Apply data augmentation to all images
    augmented_images = augment_images(images, feature_type)
    images_augmented = images + augmented_images
    y_labels_augmented = np.concatenate([y_labels, np.repeat(y_labels, 2)])

    # Extract features with best feature parameters
    X_all = extract_features(images_augmented, feature_type, best_params_feature)

    # Scale features
    best_scaler.fit(X_all)
    X_all_scaled = best_scaler.transform(X_all)

    # Retrain the best model on all data
    best_model.fit(X_all_scaled, y_labels_augmented)

    # Return the best model and parameters
    return best_model, best_scaler, best_params_feature, best_params_svm

# Train and evaluate models for LBP and GLCM
svm_model_lbp, scaler_lbp, best_params_feature_lbp, best_params_svm_lbp = train_and_evaluate("LBP")
svm_model_glcm, scaler_glcm, best_params_feature_glcm, best_params_svm_glcm = train_and_evaluate("GLCM")

# Function to classify texture and provide visual feedback
def classify_texture_gradio(image, feature_type):
    # Check if image is None
    if image is None:
        return "No image provided."

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
        return "Invalid image format."

    # Ensure the image has valid pixel values
    if not np.isfinite(image_gray).all():
        return "Image contains invalid pixel values."

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
            return "Invalid feature extraction method."
    except Exception as e:
        logging.error(f"Error during feature extraction or classification: {e}")
        return f"An error occurred: {e}"

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
    return result_image  # Return only the image

# Update the Gradio interface
interface = gr.Interface(
    fn=classify_texture_gradio,
    inputs=[
        gr.Image(type="numpy", label="Upload Image"),
        gr.Radio(["LBP", "GLCM"], label="Feature Extraction Method", value="LBP")
    ],
    outputs=gr.Image(type="pil", label="Classification Result"),
    title="Texture Classification: Grass or Wood",
    description="Upload an image and select the feature extraction method to classify it.",
    allow_flagging='never'
)

# Launch Gradio interface with shareable link
interface.launch(share=True)