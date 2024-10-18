from skimage.io import imread
from skimage.color import rgb2gray
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
from sklearn.ensemble import RandomForestClassifier
from skimage.transform import rotate
import random
from sklearn.preprocessing import StandardScaler
from skimage.util import random_noise
from skimage.exposure import adjust_gamma
import gradio as gr
from skimage.io import imread
from skimage.color import rgb2gray


# 读取图片并转换为灰度
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        # 跳过非图像文件
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            img = rgb2gray(imread(os.path.join(folder, filename)))
            if img is not None:
                images.append(img)
                # 假设文件夹名称为 'grass' 或 'wood' 来标记图片类别
                label = folder.split('/')[-1]
                labels.append(label)
    return images, labels


# 使用示例
grass_images, grass_labels = load_images_from_folder('Grass')
wood_images, wood_labels = load_images_from_folder('Woods')

# 合并草地和木材的图片和标签
images = grass_images + wood_images
labels = grass_labels + wood_labels

# 将标签转换为数字 (0: grass, 1: wood)
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)


# # 提取 GLCM 特征
# def extract_glcm_features(image):
#     glcm = greycomatrix((image * 255).astype('uint8'), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
#     contrast = greycoprops(glcm, 'contrast')[0, 0]
#     correlation = greycoprops(glcm, 'correlation')[0, 0]
#     energy = greycoprops(glcm, 'energy')[0, 0]
#     homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
#     return [contrast, correlation, energy, homogeneity]


# # 对所有图片提取 GLCM 特征
# glcm_features = [extract_glcm_features(image) for image in images]


# 提取 LBP 特征
def extract_lbp_features(image, radius=1, n_points=8):
    # 将图像转换为 uint8 类型，范围从 0 到 255
    image_uint8 = (image * 255).astype('uint8')
    # 计算 LBP，method='uniform' 表示统一模式
    lbp = local_binary_pattern(image_uint8, n_points, radius, method='uniform')
    # 计算 LBP 的直方图
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    # 归一化直方图，使得每个直方图的总和为1
    hist = hist.astype('float')
    hist /= hist.sum()
    return hist

# 数据增强：旋转和翻转图像
def augment_image(image):
    # 随机选择旋转角度或翻转
    if random.choice([True, False]):
        # 随机选择一个角度进行旋转
        angle = random.choice([90, 180, 270])
        image = rotate(image, angle)
    else:
        # 随机选择是否进行水平或垂直翻转
        if random.choice([True, False]):
            image = np.fliplr(image)
        else:
            image = np.flipud(image)
    return image

# 对所有图片提取 LBP 特征，同时进行数据增强
lbp_features = []
augmented_labels = []
for image, label in zip(images, labels_encoded):
    # 原始图像的 LBP 特征
    lbp_features.append(extract_lbp_features(image))
    augmented_labels.append(label)
    
    # 增强后的图像的 LBP 特征
    augmented_image = augment_image(image)
    lbp_features.append(extract_lbp_features(augmented_image))
    augmented_labels.append(label)

# 转换为 numpy 数组
X_lbp = np.array(lbp_features)
y = np.array(augmented_labels)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_lbp, y, test_size=0.3, random_state=42)

# 对特征进行标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建 SVM 模型
svm_model = SVC(kernel='linear', C=1.0, random_state=42)

# 训练模型
svm_model.fit(X_train_scaled, y_train)

# 在测试集上进行预测
y_pred = svm_model.predict(X_test_scaled)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")

print(f"Total number of samples: {len(lbp_features)}")
print(f"Number of grass samples: {sum(1 for label in augmented_labels if label == '0')}")
print(f"Number of wood samples: {sum(1 for label in augmented_labels if label == '1')}")


# 打印分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 打印混淆矩阵
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 定义分类函数
def classify_texture(image):
    # 将输入图像转换为灰度
    image_gray = rgb2gray(image)
    print("Grayscale image processed.")
    
    # 提取 LBP 特征
    features = extract_lbp_features(image_gray).reshape(1, -1)
    print(f"LBP Features: {features}")
    
    # 使用与训练集一致的标准化
    features_scaled = scaler.transform(features)
    print(f"Scaled Features: {features_scaled}")
    
    # 使用训练好的模型进行预测
    prediction = svm_model.predict(features_scaled)
    print(f"Prediction: {prediction}")
    
    # 解码预测结果（0: grass, 1: wood）
    label = le.inverse_transform(prediction)[0]
    print(f"Predicted Label: {label}")
    
    return label

# 创建 Gradio 界面
interface = gr.Interface(
    fn=classify_texture,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Texture Classification: Grass or Wood",
    description="Upload an image of grass or wood, and this model will classify it."
)

# 启动 Gradio 界面
interface.launch()