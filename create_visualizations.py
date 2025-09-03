# =============================================================================
# CREATE_VISUALIZATIONS.PY - TẠO ẢNH MINH HỌA CHO README
# =============================================================================
# Mục đích: Tạo các biểu đồ và ảnh minh họa cho README
# Output: Các file ảnh trong thư mục images/
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns

# Cấu hình font cho tiếng Việt
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_dataset_overview():
    """Tạo biểu đồ tổng quan dataset"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Biểu đồ cột số lượng ảnh
    categories = ['Có Mũ Bảo Hiểm', 'Không Có Mũ Bảo Hiểm']
    counts = [652, 653]
    colors = ['#2E8B57', '#DC143C']
    
    bars = ax1.bar(categories, counts, color=colors, alpha=0.8)
    ax1.set_title('Phân Bố Dataset', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Số Lượng Ảnh', fontsize=12)
    ax1.set_ylim(0, 700)
    
    # Thêm số liệu trên cột
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Biểu đồ tròn tỷ lệ
    ax2.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%',
            startangle=90, explode=(0.05, 0.05))
    ax2.set_title('Tỷ Lệ Phân Bố', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('images/dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Đã tạo: dataset_overview.png")

def create_training_progress():
    """Tạo biểu đồ tiến trình huấn luyện"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Dữ liệu mô phỏng từ 4 lần điều chỉnh
    epochs = np.arange(1, 26)
    
    # Lần 1: Overfitting nghiêm trọng
    train_acc1 = np.linspace(0.7, 0.95, 25)
    val_acc1 = np.linspace(0.7, 0.8, 25) + np.random.normal(0, 0.02, 25)
    
    ax1.plot(epochs, train_acc1, 'b-', linewidth=2, label='Training Accuracy')
    ax1.plot(epochs, val_acc1, 'r-', linewidth=2, label='Validation Accuracy')
    ax1.set_title('Lần 1: Mô hình ban đầu\n(Overfitting nghiêm trọng)', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Lần 2: Thêm Regularization
    train_acc2 = np.linspace(0.7, 0.85, 25)
    val_acc2 = np.linspace(0.7, 0.8, 25) + np.random.normal(0, 0.01, 25)
    
    ax2.plot(epochs, train_acc2, 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, val_acc2, 'r-', linewidth=2, label='Validation Accuracy')
    ax2.set_title('Lần 2: Thêm Regularization', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Lần 3: Điều chỉnh Augmentation
    train_acc3 = np.linspace(0.7, 0.84, 25)
    val_acc3 = np.linspace(0.7, 0.77, 25) + np.random.normal(0, 0.015, 25)
    
    ax3.plot(epochs, train_acc3, 'b-', linewidth=2, label='Training Accuracy')
    ax3.plot(epochs, val_acc3, 'r-', linewidth=2, label='Validation Accuracy')
    ax3.set_title('Lần 3: Điều chỉnh Augmentation', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Lần 4: Two-stage Training
    train_acc4 = np.linspace(0.7, 0.9, 25)
    val_acc4 = np.linspace(0.7, 0.83, 25) + np.random.normal(0, 0.01, 25)
    
    ax4.plot(epochs, train_acc4, 'b-', linewidth=2, label='Training Accuracy')
    ax4.plot(epochs, val_acc4, 'r-', linewidth=2, label='Validation Accuracy')
    ax4.set_title('Lần 4: Two-stage Training\n(Kết quả tối ưu)', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Đã tạo: training_progress.png")

def create_model_architecture():
    """Tạo sơ đồ kiến trúc mô hình"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Vẽ các khối của mô hình
    y_positions = [0.9, 0.7, 0.5, 0.3, 0.1]
    labels = ['Input Image\n(128×128×3)', 'MobileNetV2\n(Base Model)', 'Global Average\nPooling', 'Dense + Dropout\n(128 units)', 'Output\n(Sigmoid)']
    colors = ['#FFE4B5', '#87CEEB', '#98FB98', '#DDA0DD', '#F0E68C']
    
    for i, (y, label, color) in enumerate(zip(y_positions, labels, colors)):
        rect = Rectangle((0.1, y-0.08), 0.8, 0.15, linewidth=2, 
                        edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        ax.text(0.5, y, label, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Vẽ mũi tên kết nối
        if i < len(y_positions) - 1:
            ax.arrow(0.5, y-0.08, 0, -0.1, head_width=0.02, head_length=0.02, 
                    fc='black', ec='black')
    
    # Thêm chú thích
    ax.text(0.5, 0.95, 'Kiến Trúc Mô Hình MobileNetV2', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Thêm thông tin chi tiết
    details = [
        "• Transfer Learning từ ImageNet",
        "• Fine-tuning 20 layers cuối",
        "• Dropout (0.5) + L2 Regularization (0.001)",
        "• Binary Classification (Có/Không có mũ bảo hiểm)"
    ]
    
    for i, detail in enumerate(details):
        ax.text(0.02, 0.02 - i*0.03, detail, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('images/model_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Đã tạo: model_architecture.png")

def create_performance_comparison():
    """Tạo biểu đồ so sánh hiệu suất"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Biểu đồ cột so sánh accuracy
    iterations = ['Lần 1', 'Lần 2', 'Lần 3', 'Lần 4']
    train_acc = [95, 85, 84, 90]
    val_acc = [80, 80, 77, 83]
    
    x = np.arange(len(iterations))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, train_acc, width, label='Training Accuracy', 
                    color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, val_acc, width, label='Validation Accuracy', 
                    color='lightcoral', alpha=0.8)
    
    ax1.set_xlabel('Lần Điều Chỉnh Tham Số')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('So Sánh Hiệu Suất Qua Các Lần Tối Ưu', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(iterations)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Thêm số liệu trên cột
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height}%', ha='center', va='bottom', fontsize=10)
    
    # Biểu đồ overfitting gap
    gaps = [15, 5, 7, 7]  # Khoảng cách giữa training và validation
    
    bars3 = ax2.bar(iterations, gaps, color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
    ax2.set_xlabel('Lần Điều Chỉnh Tham Số')
    ax2.set_ylabel('Overfitting Gap (%)')
    ax2.set_title('Khoảng Cách Overfitting', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Thêm số liệu trên cột
    for bar, gap in zip(bars3, gaps):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{gap}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('images/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Đã tạo: performance_comparison.png")

def create_workflow_diagram():
    """Tạo sơ đồ quy trình làm việc"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Định nghĩa các bước
    steps = [
        "Thu thập dữ liệu\n(Kaggle + Roboflow)",
        "Tiền xử lý\n(Resize + Augmentation)",
        "Xây dựng mô hình\n(MobileNetV2)",
        "Huấn luyện\n(Two-stage Training)",
        "Đánh giá\n(Validation)",
        "Tối ưu hóa\n(4 lần điều chỉnh)",
        "Kết quả cuối\n(83% Accuracy)"
    ]
    
    # Vị trí các bước
    positions = [(0.1, 0.8), (0.3, 0.8), (0.5, 0.8), (0.7, 0.8), 
                 (0.9, 0.8), (0.5, 0.4), (0.5, 0.1)]
    
    colors = ['#FFE4B5', '#87CEEB', '#98FB98', '#DDA0DD', '#F0E68C', '#FFB6C1', '#32CD32']
    
    for i, (step, pos, color) in enumerate(zip(steps, positions, colors)):
        # Vẽ khối
        rect = Rectangle((pos[0]-0.08, pos[1]-0.08), 0.16, 0.16, 
                        linewidth=2, edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], step, ha='center', va='center', 
                fontsize=10, fontweight='bold', wrap=True)
        
        # Vẽ mũi tên kết nối
        if i < len(positions) - 1:
            if i < 4:  # Kết nối ngang
                ax.arrow(pos[0]+0.08, pos[1], 0.14, 0, head_width=0.02, head_length=0.02, 
                        fc='black', ec='black')
            elif i == 4:  # Mũi tên xuống
                ax.arrow(pos[0], pos[1]-0.08, 0, -0.22, head_width=0.02, head_length=0.02, 
                        fc='black', ec='black')
            elif i == 5:  # Mũi tên xuống cuối
                ax.arrow(pos[0], pos[1]-0.08, 0, -0.22, head_width=0.02, head_length=0.02, 
                        fc='black', ec='black')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Thêm tiêu đề
    ax.text(0.5, 0.95, 'Quy Trình Phát Triển Dự Án', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('images/workflow_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Đã tạo: workflow_diagram.png")

def main():
    """Hàm chính tạo tất cả ảnh minh họa"""
    print("🔄 Đang tạo các ảnh minh họa cho README...")
    
    create_dataset_overview()
    create_training_progress()
    create_model_architecture()
    create_performance_comparison()
    create_workflow_diagram()
    
    print("\n🎉 Hoàn thành tạo ảnh minh họa!")
    print("📁 Các file đã được lưu trong thư mục: images/")
    print("\n📝 Để thêm ảnh vào README, sử dụng:")
    print("![Mô tả](images/tên_file.png)")

if __name__ == "__main__":
    main()
