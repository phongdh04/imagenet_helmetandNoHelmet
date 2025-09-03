# =============================================================================
# HELMET_DETECTION_MODEL.PY - MÔ HÌNH PHÁT HIỆN MŨ BẢO HIỂM
# =============================================================================
# Mục đích: Xây dựng và huấn luyện mô hình CNN để phân loại có/không có mũ bảo hiểm
# Kiến trúc: MobileNetV2 với Transfer Learning và Fine-tuning
# Input: Dataset đã được tiền xử lý (128x128 pixels)
# Output: Mô hình đã huấn luyện và đánh giá hiệu suất
# Tác giả: Dự án phát hiện mũ bảo hiểm
# =============================================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os

class HelmetDetectionModel:
    """
    Lớp mô hình phát hiện mũ bảo hiểm sử dụng MobileNetV2
    """
    
    def __init__(self, input_shape=(128, 128, 3), num_classes=1):
        """
        Khởi tạo mô hình
        
        Args:
            input_shape (tuple): Kích thước input (height, width, channels)
            num_classes (int): Số lượng class (1 cho binary classification)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.fine_tune_history = None
        
    def load_data(self, data_dir, validation_split=0.2, batch_size=32):
        """
        Tải và chuẩn bị dữ liệu cho huấn luyện
        
        Args:
            data_dir (str): Đường dẫn đến thư mục dataset
            validation_split (float): Tỷ lệ validation (0.2 = 20%)
            batch_size (int): Kích thước batch
        """
        print("🔄 Đang tải dữ liệu...")
        
        # Cấu hình ImageDataGenerator cho training với augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,              # Chuẩn hóa pixel values về [0,1]
            validation_split=validation_split,
            fill_mode='nearest',         # Cách lấp đầy pixel mới
            width_shift_range=0.1,      # Dịch chuyển ngang ±10%
            height_shift_range=0.1,     # Dịch chuyển dọc ±10%
            rotation_range=30,           # Xoay ảnh ±30 độ
            zoom_range=0.2,              # Thu phóng ±20%
            shear_range=0.1,             # Cắt xén ±10%
            horizontal_flip=True         # Lật ngang
        )
        
        # Cấu hình ImageDataGenerator cho validation (chỉ rescale)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Tạo training dataset
        self.train_ds = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='binary',        # Binary classification
            subset='training',
            seed=123
        )
        
        # Tạo validation dataset
        self.val_ds = val_datagen.flow_from_directory(
            data_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='binary',
            subset='validation',
            seed=123
        )
        
        print(f"✅ Training samples: {self.train_ds.samples}")
        print(f"✅ Validation samples: {self.val_ds.samples}")
        print(f"✅ Classes: {self.train_ds.class_indices}")
        
    def build_model(self, dropout_rate=0.5, l2_reg=0.001):
        """
        Xây dựng mô hình MobileNetV2 với transfer learning
        
        Args:
            dropout_rate (float): Tỷ lệ dropout để ngăn overfitting
            l2_reg (float): Hệ số L2 regularization
        """
        print("🔄 Đang xây dựng mô hình...")
        
        # 1. Tải MobileNetV2 pre-trained trên ImageNet
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,           # Loại bỏ top classification layer
            weights='imagenet'           # Sử dụng weights đã train trên ImageNet
        )
        
        # 2. Đóng băng toàn bộ base model (freeze weights)
        for layer in base_model.layers:
            layer.trainable = False
        
        print(f"✅ Base model: MobileNetV2 với {len(base_model.layers)} layers")
        print(f"✅ Base model đã được đóng băng")
        
        # 3. Xây dựng top layers cho binary classification
        x = GlobalAveragePooling2D()(base_model.output)  # Global average pooling
        x = Dense(128, activation='relu', 
                 kernel_regularizer=l2(l2_reg))(x)       # Dense layer với L2 reg
        x = Dropout(dropout_rate)(x)                      # Dropout để ngăn overfitting
        predictions = Dense(self.num_classes, 
                           activation='sigmoid',          # Sigmoid cho binary classification
                           kernel_regularizer=l2(l2_reg))(x)
        
        # 4. Tạo model hoàn chỉnh
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        print("✅ Mô hình đã được xây dựng thành công!")
        self.model.summary()
        
    def compile_model(self, learning_rate=1e-3):
        """
        Compile mô hình với optimizer, loss function và metrics
        
        Args:
            learning_rate (float): Learning rate cho optimizer
        """
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',      # Binary crossentropy cho binary classification
            metrics=['accuracy']             # Accuracy metric
        )
        print(f"✅ Mô hình đã được compile với learning rate: {learning_rate}")
        
    def train_top_layers(self, epochs=20, patience=10):
        """
        Huấn luyện top layers (giai đoạn 1)
        
        Args:
            epochs (int): Số epochs tối đa
            patience (int): Số epochs chờ trước khi early stopping
        """
        print("🔄 Bắt đầu huấn luyện top layers...")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_model_top_layers.keras',
            monitor='val_loss',
            save_best_only=True
        )
        
        # Huấn luyện
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=[early_stopping, model_checkpoint]
        )
        
        print("✅ Hoàn thành huấn luyện top layers!")
        
    def fine_tune_model(self, unfreeze_layers=20, learning_rate=1e-5, epochs=30):
        """
        Fine-tune base model (giai đoạn 2)
        
        Args:
            unfreeze_layers (int): Số layers cuối cùng sẽ được unfreeze
            learning_rate (float): Learning rate thấp hơn cho fine-tuning
            epochs (int): Số epochs tối đa
        """
        print(f"🔄 Bắt đầu fine-tuning {unfreeze_layers} layers cuối...")
        
        # Unfreeze một số layers cuối của base model
        base_model = self.model.layers[1]  # MobileNetV2 layer
        for layer in base_model.layers[-unfreeze_layers:]:
            layer.trainable = True
        
        print(f"✅ Đã unfreeze {unfreeze_layers} layers cuối của base model")
        
        # Re-compile với learning rate thấp hơn
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_model_final.keras',
            monitor='val_loss',
            save_best_only=True
        )
        
        # Fine-tuning
        self.fine_tune_history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=[early_stopping, model_checkpoint]
        )
        
        print("✅ Hoàn thành fine-tuning!")
        
    def evaluate_model(self):
        """
        Đánh giá mô hình trên validation set
        """
        print("🔄 Đang đánh giá mô hình...")
        
        # Đánh giá trên validation set
        loss, accuracy = self.model.evaluate(self.val_ds)
        
        print(f"✅ Validation Loss: {loss:.4f}")
        print(f"✅ Validation Accuracy: {accuracy:.4f}")
        
        return loss, accuracy
        
    def plot_training_history(self):
        """
        Vẽ biểu đồ quá trình huấn luyện
        """
        if self.history is None:
            print("⚠️  Chưa có lịch sử huấn luyện để vẽ")
            return
            
        # Tạo subplot cho training history
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Training và Validation Accuracy (Top Layers)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Training loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Training và Validation Loss (Top Layers)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Fine-tuning history (nếu có)
        if self.fine_tune_history is not None:
            # Fine-tuning accuracy
            axes[1, 0].plot(self.fine_tune_history.history['accuracy'], label='Training Accuracy')
            axes[1, 0].plot(self.fine_tune_history.history['val_accuracy'], label='Validation Accuracy')
            axes[1, 0].set_title('Training và Validation Accuracy (Fine-tuning)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Fine-tuning loss
            axes[1, 1].plot(self.fine_tune_history.history['loss'], label='Training Loss')
            axes[1, 1].plot(self.fine_tune_history.history['val_loss'], label='Validation Loss')
            axes[1, 1].set_title('Training và Validation Loss (Fine-tuning)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def save_model(self, filepath='helmet_detection_model.keras'):
        """
        Lưu mô hình đã huấn luyện
        
        Args:
            filepath (str): Đường dẫn file để lưu model
        """
        self.model.save(filepath)
        print(f"✅ Mô hình đã được lưu tại: {filepath}")

def main():
    """
    Hàm chính để chạy toàn bộ pipeline huấn luyện
    """
    print("=" * 70)
    print("MÔ HÌNH PHÁT HIỆN MŨ BẢO HIỂM")
    print("=" * 70)
    
    # Khởi tạo mô hình
    model = HelmetDetectionModel(input_shape=(128, 128, 3))
    
    # Cấu hình đường dẫn dataset
    # Có thể thay đổi tùy theo môi trường
    data_dir_colab = '/content/drive/MyDrive/resized_dataset'
    data_dir_local = './resized_dataset'
    
    data_dir = data_dir_colab if os.path.exists(data_dir_colab) else data_dir_local
    
    print(f"📁 Dataset directory: {data_dir}")
    
    # Bước 1: Tải dữ liệu
    print("\n" + "=" * 50)
    print("BƯỚC 1: TẢI DỮ LIỆU")
    print("=" * 50)
    model.load_data(data_dir, validation_split=0.2, batch_size=32)
    
    # Bước 2: Xây dựng mô hình
    print("\n" + "=" * 50)
    print("BƯỚC 2: XÂY DỰNG MÔ HÌNH")
    print("=" * 50)
    model.build_model(dropout_rate=0.5, l2_reg=0.001)
    
    # Bước 3: Compile mô hình
    print("\n" + "=" * 50)
    print("BƯỚC 3: COMPILE MÔ HÌNH")
    print("=" * 50)
    model.compile_model(learning_rate=1e-3)
    
    # Bước 4: Huấn luyện top layers
    print("\n" + "=" * 50)
    print("BƯỚC 4: HUẤN LUYỆN TOP LAYERS")
    print("=" * 50)
    model.train_top_layers(epochs=20, patience=10)
    
    # Bước 5: Fine-tuning
    print("\n" + "=" * 50)
    print("BƯỚC 5: FINE-TUNING")
    print("=" * 50)
    model.fine_tune_model(unfreeze_layers=20, learning_rate=1e-5, epochs=30)
    
    # Bước 6: Đánh giá mô hình
    print("\n" + "=" * 50)
    print("BƯỚC 6: ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 50)
    loss, accuracy = model.evaluate_model()
    
    # Bước 7: Vẽ biểu đồ
    print("\n" + "=" * 50)
    print("BƯỚC 7: VẼ BIỂU ĐỒ")
    print("=" * 50)
    model.plot_training_history()
    
    # Bước 8: Lưu mô hình
    print("\n" + "=" * 50)
    print("BƯỚC 8: LƯU MÔ HÌNH")
    print("=" * 50)
    model.save_model('helmet_detection_final.keras')
    
    print("\n" + "=" * 70)
    print("🎉 HOÀN THÀNH HUẤN LUYỆN!")
    print("=" * 70)
    print(f"📊 Final Validation Accuracy: {accuracy:.4f}")
    print(f"📊 Final Validation Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
