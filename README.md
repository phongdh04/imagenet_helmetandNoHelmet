# Dự Án Phát Hiện Mũ Bảo Hiểm (Helmet Detection)

## Mô Tả Dự Án

Dự án này sử dụng Deep Learning để phát hiện và phân loại người đội mũ bảo hiểm và không đội mũ bảo hiểm khi đi xe máy. Mô hình được xây dựng dựa trên kiến trúc MobileNetV2 với transfer learning từ ImageNet.

## Nguồn Dữ Liệu và Cách Sử Dụng

### **Các Nguồn Dữ Liệu Chính**

1. **Traffic Violation Dataset V3** từ Kaggle:
   - **Link**: https://www.kaggle.com/datasets/meliodassourav/traffic-violation-dataset-v3
   - **Mô tả**: Tập dữ liệu chứa các ảnh vi phạm giao thông từ camera giám sát
   - **Số lượng**: ~2,000+ ảnh vi phạm giao thông
   - **Định dạng**: JPG, PNG
   - **Độ phân giải**: Đa dạng (640x480, 1280x720, v.v.)
   - **Mục đích**: Cung cấp dữ liệu thực tế về các tình huống giao thông

2. **Helmet No Helmet Detection Dataset** từ Roboflow:
   - **Link**: https://universe.roboflow.com/programa-delfn/helmet-no-helmet-detection-hjdvvx/dataset/1
   - **Mô tả**: Tập dữ liệu chuyên biệt cho phát hiện mũ bảo hiểm
   - **Số lượng**: ~1,500+ ảnh được gán nhãn
   - **Định dạng**: JPG, PNG
   - **Độ phân giải**: Đa dạng
   - **Mục đích**: Cung cấp dữ liệu đã được gán nhãn chính xác

### **Quá Trình Xử Lý Dữ Liệu**

#### **Bước 1: Thu thập và Tổ chức**
```bash
# Cấu trúc thư mục gốc
dataset/
├── Clear_Helmet/         # Ảnh có mũ bảo hiểm
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── Clear_NoHelmet/       # Ảnh không có mũ bảo hiểm
    ├── image1.png
    ├── image2.png
    └── ...
```

#### **Bước 2: Tiền xử lý**
- **Resize**: Chuẩn hóa về 128x128 pixels
- **Padding**: Giữ nguyên aspect ratio với padding trắng
- **Chuẩn hóa**: Chuyển pixel values về [0,1]

#### **Bước 3: Tăng cường dữ liệu (Data Augmentation)**
- **Rotation**: Xoay ảnh ±35°
- **Translation**: Dịch chuyển ±20%
- **Shear**: Cắt xén ±20%
- **Zoom**: Thu phóng ±10%
- **Flip**: Lật ngang
- **Brightness**: Điều chỉnh độ sáng ±30%

### **Thống Kê Dataset Sau Xử Lý**

| Thông Số | Giá Trị |
|----------|---------|
| **Tổng số ảnh** | 1,305 |
| **Ảnh có mũ bảo hiểm** | 652 (49.9%) |
| **Ảnh không có mũ bảo hiểm** | 653 (50.1%) |
| **Kích thước ảnh** | 128×128 pixels |
| **Định dạng** | PNG |
| **Channels** | RGB (3) |
| **Cân bằng dataset** | ✅ Cân bằng tốt |

### **Cách Tải và Sử Dụng Dataset**

#### **Từ Kaggle:**
```bash
# Cài đặt kaggle CLI
pip install kaggle

# Tải dataset
kaggle datasets download -d meliodassourav/traffic-violation-dataset-v3
unzip traffic-violation-dataset-v3.zip
```

#### **Từ Roboflow:**
```python
# Sử dụng roboflow API
from roboflow import Roboflow
rf = Roboflow(api_key="your_api_key")
project = rf.workspace("programa-delfn").project("helmet-no-helmet-detection-hjdvx")
dataset = project.version(1).download("folder")
```

#### **Từ Google Drive (Colab):**
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

### **Sử Dụng Dataset Trong Code**

#### **Kiểm tra dataset:**
```python
# Chạy script đếm ảnh
python count_images.py

# Hoặc trong notebook
from count_images import count_images_in_directory
helmet_count = count_images_in_directory('./resized_dataset/Clear_Helmet')
no_helmet_count = count_images_in_directory('./resized_dataset/Clear_NoHelmet')
print(f"Helmet: {helmet_count}, No Helmet: {no_helmet_count}")
```

#### **Tiền xử lý dataset:**
```python
# Chạy script tiền xử lý
python resize_and_augment.py

# Hoặc trong notebook
from resize_and_augment import resize_images, augment_pictures
resize_images('./dataset', './resized_dataset', size=(128, 128))
augment_pictures('./resized_dataset', './augmented_dataset', num_augmented_images_per_image=10)
```

#### **Huấn luyện mô hình:**
```python
# Chạy script huấn luyện
python helmet_detection_model.py

# Hoặc trong notebook
from helmet_detection_model import HelmetDetectionModel
model = HelmetDetectionModel()
model.load_data('./resized_dataset')
model.build_model()
model.train_top_layers()
model.fine_tune_model()
```

## Sơ Đồ Luồng Mô Hình

### **Kiến Trúc Tổng Quan**

```
Input Image (128×128×3)
         ↓
   MobileNetV2 Base Model
   (Pre-trained on ImageNet)
         ↓
   Global Average Pooling
         ↓
   Dense Layer (128 units)
   + ReLU Activation
         ↓
   Dropout (0.5)
         ↓
   Dense Layer (1 unit)
   + Sigmoid Activation
         ↓
   Output: [0, 1]
   (0: Không có mũ, 1: Có mũ)
```

### **Quy Trình Huấn Luyện**

#### **Giai Đoạn 1: Transfer Learning**
```
1. Tải MobileNetV2 pre-trained
2. Đóng băng toàn bộ base model
3. Thêm top layers mới
4. Huấn luyện chỉ top layers
   - Learning Rate: 1e-3
   - Epochs: 20
   - Optimizer: Adam
```

#### **Giai Đoạn 2: Fine-tuning**
```
1. Unfreeze 20 layers cuối của base model
2. Huấn luyện toàn bộ mô hình
   - Learning Rate: 1e-5 (thấp hơn)
   - Epochs: 30
   - Optimizer: Adam
```

### **Pipeline Xử Lý Dữ Liệu**

```
Raw Images
    ↓
Resize to 128×128
    ↓
Data Augmentation
    ↓
Normalization (/255)
    ↓
Train/Validation Split
    ↓
Model Training
    ↓
Evaluation & Testing
```

## Cấu Trúc Dự Án

```
Proj/
├── CountImage.ipynb          # Đếm số lượng ảnh trong dataset
├── Resize.ipynb              # Tiền xử lý và tăng cường dữ liệu
├── Main.ipynb                # Mô hình chính và huấn luyện
├── count_images.py           # Script đếm ảnh (Python)
├── resize_and_augment.py     # Script tiền xử lý (Python)
├── helmet_detection_model.py # Script mô hình chính (Python)
├── create_visualizations.py  # Tạo ảnh minh họa
├── requirements.txt          # Thư viện cần thiết
├── README.md                 # Hướng dẫn chi tiết
├── images/                   # Thư mục chứa ảnh minh họa
└── resized_dataset/          # Dataset đã được xử lý
    ├── Clear_Helmet/         # 652 ảnh có mũ bảo hiểm
    └── Clear_NoHelmet/       # 653 ảnh không có mũ bảo hiểm
```

## Thống Kê Dataset

- **Tổng số ảnh**: 1,305 ảnh
- **Ảnh có mũ bảo hiểm**: 652 ảnh
- **Ảnh không có mũ bảo hiểm**: 653 ảnh
- **Kích thước ảnh**: 128x128 pixels
- **Định dạng**: PNG

## Các Bước Thực Hiện

### 1. Tiền Xử Lý Dữ Liệu (`Resize.ipynb`)
- Resize ảnh về kích thước 128x128 pixels
- Giữ nguyên tỷ lệ khung hình với padding
- Tăng cường dữ liệu với các kỹ thuật:
  - Xoay ảnh (rotation_range=35°)
  - Dịch chuyển (width_shift_range=0.2, height_shift_range=0.2)
  - Cắt xén (shear_range=0.2)
  - Thu phóng (zoom_range=0.1)
  - Lật ngang (horizontal_flip=True)
  - Điều chỉnh độ sáng (brightness_range=[0.7, 1.3])

### 2. Đếm Dữ Liệu (`CountImage.ipynb`)
- Đếm số lượng ảnh trong từng thư mục
- Kiểm tra định dạng ảnh hợp lệ
- Xác nhận cân bằng dữ liệu

### 3. Xây Dựng Mô Hình (`Main.ipynb`)
- **Kiến trúc**: MobileNetV2 (pre-trained trên ImageNet)
- **Transfer Learning**: Đóng băng base model, chỉ huấn luyện top layers
- **Fine-tuning**: Mở 20 lớp cuối cùng của base model
- **Regularization**: Dropout (0.5) và L2 regularization (0.001)
- **Optimizer**: Adam với learning rate khác nhau cho từng giai đoạn

## Kết Quả Huấn Luyện

### Phân Tích Quá Trình Điều Chỉnh Tham Số

Dự án đã trải qua **4 lần điều chỉnh tham số** để tối ưu hóa hiệu suất mô hình:

#### **Lần 1: Mô hình ban đầu (Overfitting nghiêm trọng)**
- **Kiến trúc**: MobileNetV2 với top layers đơn giản
- **Kết quả**: 
  - Training Accuracy: ~95% (epoch 25)
  - Validation Accuracy: ~80% (epoch 25)
  - **Vấn đề**: Overfitting rõ rệt với khoảng cách lớn giữa training và validation

#### **Lần 2: Thêm Regularization**
- **Cải tiến**: Thêm Dropout (0.5) và L2 Regularization (0.001)
- **Kết quả**:
  - Training Accuracy: ~85% (epoch 30)
  - Validation Accuracy: ~80% (epoch 30)
  - **Cải thiện**: Giảm overfitting nhưng vẫn còn khoảng cách

#### **Lần 3: Điều chỉnh Data Augmentation**
- **Cải tiến**: Tăng cường augmentation và điều chỉnh validation split
- **Kết quả**:
  - Training Accuracy: ~84% (epoch 30)
  - Validation Accuracy: ~77% (epoch 30)
  - **Vấn đề**: Validation accuracy giảm do augmentation quá mạnh

#### **Lần 4: Two-stage Training với Fine-tuning**
- **Chiến lược**: 
  - Giai đoạn 1: Huấn luyện top layers (20 epochs)
  - Giai đoạn 2: Fine-tuning 20 layers cuối (30 epochs)
- **Kết quả cuối cùng**:
  - **Top Layers Training**:
    - Training Accuracy: ~87% (epoch 18)
    - Validation Accuracy: ~82% (epoch 18)
  - **Fine-tuning**:
    - Training Accuracy: ~90% (epoch 22)
    - Validation Accuracy: ~83% (epoch 22)
    - **Validation Loss**: 0.52-0.53 (ổn định)

### **Kết Quả Tối Ưu Cuối Cùng**

- **Final Validation Accuracy**: **83%**
- **Final Validation Loss**: **0.52**
- **Training Accuracy**: **90%**
- **Overfitting Gap**: **7%** (cải thiện đáng kể từ 15% ban đầu)

### **Nhận Xét Kỹ Thuật**

1. **Two-stage Training hiệu quả**: Fine-tuning sau khi huấn luyện top layers giúp cải thiện generalization
2. **Regularization cân bằng**: Dropout + L2 giúp kiểm soát overfitting
3. **Learning Rate phù hợp**: 1e-3 cho top layers, 1e-5 cho fine-tuning
4. **Early Stopping**: Ngăn chặn overfitting kịp thời

## Công Nghệ Sử Dụng

- **Framework**: TensorFlow/Keras
- **Pre-trained Model**: MobileNetV2
- **Data Augmentation**: ImageDataGenerator
- **Regularization**: Dropout, L2 Regularization
- **Callbacks**: Early Stopping, Model Checkpoint

## Cách Sử Dụng

1. **Chuẩn bị môi trường**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Chạy các file Python theo thứ tự**:
   ```bash
   python count_images.py              # Đếm và kiểm tra dữ liệu
   python resize_and_augment.py        # Tiền xử lý dữ liệu
   python helmet_detection_model.py    # Huấn luyện mô hình
   ```

3. **Hoặc chạy các notebook theo thứ tự**:
   - `CountImage.ipynb`: Kiểm tra dữ liệu
   - `Resize.ipynb`: Tiền xử lý dữ liệu
   - `Main.ipynb`: Huấn luyện mô hình

4. **Kết quả**:
   - Mô hình được lưu dưới dạng `helmet_detection_final.keras`
   - Biểu đồ huấn luyện được hiển thị
   - Độ chính xác validation: **83%**

## Hiệu Suất và So Sánh

### **Kết Quả Cuối Cùng**
- **Validation Accuracy**: 83%
- **Training Accuracy**: 90%
- **Overfitting Gap**: 7%
- **Training Time**: ~45 phút (Google Colab GPU)

### **So Sánh với Baseline**
- **Mô hình ban đầu**: 80% validation accuracy (overfitting nghiêm trọng)
- **Sau 4 lần tối ưu**: 83% validation accuracy (cải thiện 3%)
- **Giảm overfitting**: Từ 15% xuống 7% gap

### **Đánh Giá Chất Lượng**
- ✅ **Cân bằng dataset**: 652 vs 653 ảnh
- ✅ **Regularization hiệu quả**: Dropout + L2
- ✅ **Transfer learning**: Tận dụng MobileNetV2
- ⚠️ **Dataset size**: Có thể tăng cường thêm
- ⚠️ **Test set**: Cần tách riêng từ đầu

## Đặc Điểm Kỹ Thuật

- **Transfer Learning**: Tận dụng kiến thức từ ImageNet
- **Data Augmentation**: Tăng cường dữ liệu để cải thiện generalization
- **Regularization**: Ngăn chặn overfitting
- **Two-stage Training**: Huấn luyện top layers trước, sau đó fine-tune
- **Early Stopping**: Dừng huấn luyện khi validation loss tăng

## Bài Học Rút Ra

### **Vấn Đề Gặp Phải**
1. **Overfitting nhẹ**: Khoảng cách 15% giữa training và validation accuracy
2. **Augmentation quá mạnh**: Làm giảm validation accuracy
3. **Learning rate không phù hợp**: Gây ra training không ổn định

### **Giải Pháp Hiệu Quả**
1. **Regularization cân bằng**: Dropout 0.5 + L2 0.001
2. **Two-stage training**: Tách biệt huấn luyện top layers và fine-tuning
3. **Learning rate scheduling**: 1e-3 → 1e-5
4. **Early stopping**: Patience = 10 epochs

### **Khuyến Nghị Cải Thiện**
1. **Tăng cường dataset**: Thu thập thêm dữ liệu đa dạng
2. **Cross-validation**: Sử dụng k-fold để đánh giá chính xác hơn
3. **Ensemble methods**: Kết hợp nhiều mô hình
4. **Test set riêng biệt**: Tách riêng test set từ đầu

## Tác Giả

Dự án được phát triển cho mục đích học tập và nghiên cứu về Computer Vision và Deep Learning.

