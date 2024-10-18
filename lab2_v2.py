from skimage.io import imread
from skimage.color import rgb2gray
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
from sklearn.preprocessing import StandardScaler
from skimage.transform import rotate
import random
import gradio as gr
from skimage.util import random_noise
from skimage.exposure import adjust_gamma

# 读取图片并转换为灰度
def load_images_from_folder(folder, label_name):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = rgb2gray(imread(os.path.join(folder, filename)))
            if img is not None:
                images.append(img)
                labels.append(label_name)  # 使用传入的类别标签
    return images, labels

# 使用示例
grass_images, grass_labels = load_images_from_folder('Grass', 0)
wood_images, wood_labels = load_images_from_folder('Woods', 1)

# 合并图片和标签
images = grass_images + wood_images
labels = grass_labels + wood_labels

# 初始化 LabelEncoder 并拟合标签
le = LabelEncoder()
le.fit([0, 1])  # 拟合标签类别

# 将标签转换为 numpy 数组
y = np.array(labels)

# LBP 特征提取
def extract_lbp_features(image, radius=1, n_points=8):
    image_uint8 = (image * 255).astype('uint8')
    lbp = local_binary_pattern(image_uint8, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype('float')
    hist /= hist.sum()
    return hist

# GLCM 特征提取
# 将灰度级别减少到较少的级别
def extract_glcm_features(image):
    image_uint8 = (image * 255).astype('uint8')
    glcm = graycomatrix(image_uint8, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # 将结果转换为 NumPy 数组并返回
    return np.array([contrast, correlation, energy, homogeneity])

# 数据增强：旋转和翻转图像LBP
def augment_image_LBP(image):
    if random.choice([True, False]):
        angle = random.choice([90, 180, 270])
        image = rotate(image, angle)
    if random.choice([True, False]):
        if random.choice([True, False]):
            image = np.fliplr(image)
        else:
            image = np.flipud(image)

    return image

# 数据增强：旋转和翻转图像GLCM
def augment_image_GLCM(image):
    if random.choice([True, False]):
        gamma = random.uniform(0.9, 1.1)  # 限制在较小范围内
        image = adjust_gamma(image, gamma=gamma)

    # 添加噪声（减小方差）
    if random.choice([True, False]):
        image = random_noise(image, mode='gaussian', var=0.005)  # 更小的方差
    return image

# 提取特征并进行数据增强
def extract_features_and_augment(images, labels, feature_type):
    global feature
    features = []
    augmented_labels = []
    for image, label in zip(images, labels):
        if feature_type == "LBP":
            feature = extract_lbp_features(image)
        elif feature_type == "GLCM":
            feature = extract_glcm_features(image)
        features.append(feature)
        augmented_labels.append(label)

        # 每张图像生成两个增强版本
        for _ in range(2):
            if feature_type == "LBP":
                augmented_image = augment_image_LBP(image)
                feature = extract_lbp_features(augmented_image)
            elif feature_type == "GLCM":
                augmented_image = augment_image_GLCM(image)
                feature = extract_glcm_features(augmented_image)
            features.append(feature)
            augmented_labels.append(label)
    return np.array(features), np.array(augmented_labels)

def train_and_evaluate(feature_type):
    X, y_augmented = extract_features_and_augment(images, y, feature_type)
    X_train, X_test, y_train, y_test = train_test_split(X, y_augmented, test_size=0.3, random_state=42)

    # 为每种特征类型创建单独的 scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练 SVM
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # 预测并打印报告
    y_pred = svm_model.predict(X_test_scaled)
    print(f"--- {feature_type} Report ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return svm_model, scaler

# 为两种特征类型分别训练并返回模型和 scaler
svm_model_lbp, scaler_lbp = train_and_evaluate("LBP")
svm_model_glcm, scaler_glcm = train_and_evaluate("GLCM")

# 修改 Gradio 分类函数，根据选择的特征类型使用对应的模型和 scaler
def classify_texture(image, feature_type):
    global prediction
    image_gray = rgb2gray(image)

    # 根据用户选择的特征类型提取特征
    if feature_type == "LBP":
        features = extract_lbp_features(image_gray).reshape(1, -1)
        features_scaled = scaler_lbp.transform(features)
        prediction = svm_model_lbp.predict(features_scaled)
    elif feature_type == "GLCM":
        features = extract_glcm_features(image_gray).reshape(1, -1)
        features_scaled = scaler_glcm.transform(features)
        prediction = svm_model_glcm.predict(features_scaled)

    # 返回预测结果
    return "Grass" if prediction[0] == 0 else "Wood"

# 创建 Gradio 界面
interface = gr.Interface(
    fn=classify_texture,
    inputs=[gr.Image(type="numpy"), gr.Radio(["LBP", "GLCM"], label="Feature Extraction Method")],
    outputs="text",
    title="Texture Classification: Grass or Wood",
    description="Upload an image and select the feature extraction method to classify it."
)

# 启动 Gradio 界面
interface.launch()