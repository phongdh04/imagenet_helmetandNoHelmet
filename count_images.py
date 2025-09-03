# =============================================================================
# COUNT_IMAGES.PY - ĐẾM SỐ LƯỢNG ẢNH TRONG DATASET
# =============================================================================
# Mục đích: Đếm và thống kê số lượng ảnh trong từng thư mục của dataset
# Input: Dataset từ Google Drive hoặc local directory
# Output: Số lượng ảnh trong từng class (Helmet/NoHelmet)
# Tác giả: Dự án phát hiện mũ bảo hiểm
# =============================================================================

import os
from pathlib import Path

def count_images_in_directory(directory_path):
    """
    Đếm số lượng ảnh trong một thư mục
    
    Args:
        directory_path (str): Đường dẫn đến thư mục cần đếm
        
    Returns:
        int: Số lượng ảnh tìm thấy
    """
    # Các định dạng ảnh được hỗ trợ
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
    
    count = 0
    try:
        # Duyệt qua tất cả file trong thư mục
        for file in os.listdir(directory_path):
            # Kiểm tra xem file có phải là ảnh không bằng cách kiểm tra extension
            if os.path.splitext(file)[1].lower() in image_extensions:
                count += 1
        return count
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy thư mục {directory_path}")
        return 0
    except Exception as e:
        print(f"Lỗi khi đếm ảnh: {e}")
        return 0

def main():
    """
    Hàm chính để đếm ảnh trong cả hai thư mục dataset
    """
    print("=" * 60)
    print("ĐẾM SỐ LƯỢNG ẢNH TRONG DATASET")
    print("=" * 60)
    
    # Đường dẫn đến các thư mục dataset
    # Có thể thay đổi đường dẫn tùy theo môi trường (Google Drive hoặc local)
    
    # Đường dẫn cho Google Drive (Colab)
    helmet_dir_colab = '/content/gdrive/MyDrive/dataset/Clear_Helmet'
    no_helmet_dir_colab = '/content/gdrive/MyDrive/dataset/Clear_NoHelmet'
    
    # Đường dẫn cho local (nếu chạy trên máy local)
    helmet_dir_local = './resized_dataset/Clear_Helmet'
    no_helmet_dir_local = './resized_dataset/Clear_NoHelmet'
    
    # Thử đếm từ Google Drive trước
    print("\n1. ĐẾM ẢNH CÓ MŨ BẢO HIỂM:")
    print("-" * 40)
    
    helmet_count = count_images_in_directory(helmet_dir_colab)
    if helmet_count == 0:
        # Nếu không tìm thấy trên Google Drive, thử local
        helmet_count = count_images_in_directory(helmet_dir_local)
    
    print(f"Số lượng ảnh có mũ bảo hiểm: {helmet_count}")
    
    print("\n2. ĐẾM ẢNH KHÔNG CÓ MŨ BẢO HIỂM:")
    print("-" * 40)
    
    no_helmet_count = count_images_in_directory(no_helmet_dir_colab)
    if no_helmet_count == 0:
        # Nếu không tìm thấy trên Google Drive, thử local
        no_helmet_count = count_images_in_directory(no_helmet_dir_local)
    
    print(f"Số lượng ảnh không có mũ bảo hiểm: {no_helmet_count}")
    
    # Tính tổng và thống kê
    total_count = helmet_count + no_helmet_count
    
    print("\n3. THỐNG KÊ TỔNG QUAN:")
    print("-" * 40)
    print(f"Tổng số ảnh: {total_count}")
    print(f"Ảnh có mũ bảo hiểm: {helmet_count} ({helmet_count/total_count*100:.1f}%)")
    print(f"Ảnh không có mũ bảo hiểm: {no_helmet_count} ({no_helmet_count/total_count*100:.1f}%)")
    
    # Kiểm tra cân bằng dataset
    if abs(helmet_count - no_helmet_count) <= 10:
        print("\n✅ Dataset cân bằng tốt!")
    else:
        print(f"\n⚠️  Dataset không cân bằng (chênh lệch: {abs(helmet_count - no_helmet_count)} ảnh)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
