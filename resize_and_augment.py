# =============================================================================
# RESIZE_AND_AUGMENT.PY - TIỀN XỬ LÝ VÀ TĂNG CƯỜNG DỮ LIỆU
# =============================================================================
# Mục đích: Resize ảnh về kích thước chuẩn và tăng cường dữ liệu
# Input: Dataset gốc từ Google Drive hoặc local
# Output: Dataset đã được resize và augmented
# Tác giả: Dự án phát hiện mũ bảo hiểm
# =============================================================================

import cv2
import numpy as np
import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

def resize_images(input_dir, output_dir, size=(128, 128)):
    """
    Resize tất cả ảnh trong thư mục về kích thước chuẩn
    
    Args:
        input_dir (str): Đường dẫn thư mục input
        output_dir (str): Đường dẫn thư mục output
        size (tuple): Kích thước mới (width, height)
    """
    print(f"🔄 Bắt đầu resize ảnh từ {input_dir} đến {output_dir}")
    
    # Tạo thư mục output nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Đã tạo thư mục output: {output_dir}")

    # Duyệt qua tất cả thư mục con và file
    for root, dirs, files in os.walk(input_dir):
        # Tạo thư mục con tương ứng trong output
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        # Xử lý từng file ảnh
        for filename in files:
            img_path = os.path.join(root, filename)
            
            # Kiểm tra xem có phải file ảnh không
            if os.path.isfile(img_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Đọc ảnh
                img = cv2.imread(img_path)
                if img is not None:
                    # Lấy kích thước gốc
                    original_height, original_width = img.shape[:2]
                    target_width, target_height = size

                    # Tính toán tỷ lệ khung hình để giữ nguyên aspect ratio
                    aspect_ratio = original_width / original_height
                    
                    # Tính kích thước mới dựa trên aspect ratio
                    if original_width > original_height:
                        new_width = target_width
                        new_height = int(new_width / aspect_ratio)
                    else:
                        new_height = target_height
                        new_width = int(new_height * aspect_ratio)

                    # Resize ảnh
                    resized_img = cv2.resize(img, (new_width, new_height), 
                                           interpolation=cv2.INTER_AREA)

                    # Tính toán padding để đạt kích thước mục tiêu
                    pad_width = target_width - new_width
                    pad_height = target_height - new_height
                    top_pad = pad_height // 2
                    bottom_pad = pad_height - top_pad
                    left_pad = pad_width // 2
                    right_pad = pad_width - left_pad

                    # Thêm padding với nền trắng
                    padded_img = cv2.copyMakeBorder(
                        resized_img, 
                        top_pad, bottom_pad, left_pad, right_pad, 
                        cv2.BORDER_CONSTANT, 
                        value=[255, 255, 255]
                    )

                    # Lưu ảnh đã xử lý
                    output_path = os.path.join(output_subdir, filename)
                    cv2.imwrite(output_path, padded_img)
                    
                    print(f"✅ Đã xử lý: {filename}")
                else:
                    print(f"⚠️  Không thể đọc ảnh: {filename}")
    
    print(f"🎉 Hoàn thành resize ảnh!")

def augment_pictures(input_dir, output_dir, num_augmented_images_per_image=10):
    """
    Tăng cường dữ liệu bằng cách tạo ra các biến thể của ảnh gốc
    
    Args:
        input_dir (str): Đường dẫn thư mục input (đã resize)
        output_dir (str): Đường dẫn thư mục output (augmented)
        num_augmented_images_per_image (int): Số ảnh augmented cho mỗi ảnh gốc
    """
    print(f"🔄 Bắt đầu augmentation từ {input_dir} đến {output_dir}")
    
    # Xóa thư mục output cũ nếu tồn tại và tạo mới
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Đã tạo thư mục output: {output_dir}")

    # Cấu hình ImageDataGenerator cho augmentation
    datagen = ImageDataGenerator(
        rotation_range=35,           # Xoay ảnh ±35 độ
        width_shift_range=0.2,      # Dịch chuyển ngang ±20%
        height_shift_range=0.2,     # Dịch chuyển dọc ±20%
        shear_range=0.2,            # Cắt xén ±20%
        zoom_range=0.1,             # Thu phóng ±10%
        horizontal_flip=True,        # Lật ngang
        brightness_range=[0.7, 1.3], # Điều chỉnh độ sáng ±30%
        fill_mode='reflect'         # Cách lấp đầy pixel mới
    )

    # Xử lý từng class (thư mục con)
    for class_name in os.listdir(input_dir):
        class_input_dir = os.path.join(input_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)
        
        if os.path.isdir(class_input_dir):
            os.makedirs(class_output_dir, exist_ok=True)
            print(f"🔄 Đang xử lý class: {class_name}")

            # Tạo generator cho class hiện tại
            generator = datagen.flow_from_directory(
                input_dir,                    # Thư mục cha
                classes=[class_name],        # Chỉ định class cần xử lý
                target_size=(128, 128),      # Kích thước ảnh
                batch_size=1,                # Xử lý 1 ảnh/lần
                save_to_dir=class_output_dir, # Thư mục lưu
                save_prefix='aug',           # Prefix cho tên file
                save_format='jpeg',          # Định dạng lưu
                shuffle=False                # Không shuffle để xử lý tuần tự
            )

            # Đếm số ảnh gốc trong class
            num_original_images = len([f for f in os.listdir(class_input_dir) 
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            if num_original_images > 0:
                # Tạo augmented images
                total_augmented = num_original_images * num_augmented_images_per_image
                for i in range(total_augmented):
                    next(generator)
                
                print(f"✅ Hoàn thành {class_name}: {total_augmented} ảnh augmented")
            else:
                print(f"⚠️  Không tìm thấy ảnh trong {class_name}")
        else:
            print(f"⚠️  Không phải thư mục: {class_name}")
    
    print(f"🎉 Hoàn thành augmentation!")

def main():
    """
    Hàm chính để thực hiện toàn bộ quá trình tiền xử lý
    """
    print("=" * 70)
    print("TIỀN XỬ LÝ VÀ TĂNG CƯỜNG DỮ LIỆU")
    print("=" * 70)
    
    # Cấu hình đường dẫn
    # Có thể thay đổi tùy theo môi trường (Google Drive hoặc local)
    
    # Đường dẫn cho Google Drive (Colab)
    input_dataset_dir_colab = '/content/drive/MyDrive/dataset'
    resized_output_dir_colab = '/content/resized_dataset'
    augmented_output_dir_colab = '/content/augmented_dataset'
    
    # Đường dẫn cho local
    input_dataset_dir_local = './dataset'
    resized_output_dir_local = './resized_dataset'
    augmented_output_dir_local = './augmented_dataset'
    
    # Chọn đường dẫn phù hợp
    input_dir = input_dataset_dir_colab if os.path.exists(input_dataset_dir_colab) else input_dataset_dir_local
    resized_dir = resized_output_dir_colab if os.path.exists(input_dataset_dir_colab) else resized_output_dir_local
    augmented_dir = augmented_output_dir_colab if os.path.exists(input_dataset_dir_colab) else augmented_output_dir_local
    
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Resized output: {resized_dir}")
    print(f"📁 Augmented output: {augmented_dir}")
    
    # Bước 1: Resize ảnh
    print("\n" + "=" * 50)
    print("BƯỚC 1: RESIZE ẢNH")
    print("=" * 50)
    resize_images(input_dir, resized_dir, size=(128, 128))
    
    # Bước 2: Augmentation
    print("\n" + "=" * 50)
    print("BƯỚC 2: TĂNG CƯỜNG DỮ LIỆU")
    print("=" * 50)
    augment_pictures(resized_dir, augmented_dir, num_augmented_images_per_image=10)
    
    print("\n" + "=" * 70)
    print("🎉 HOÀN THÀNH TIỀN XỬ LÝ!")
    print("=" * 70)
    print(f"✅ Ảnh đã resize: {resized_dir}")
    print(f"✅ Ảnh đã augmented: {augmented_dir}")

if __name__ == "__main__":
    main()
