# Helmet Detection Project

A deep learning project for detecting helmet and no-helmet classification using MobileNetV2 with transfer learning.

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/phongdh04/imagenet_helmetandNoHelmet.git
   cd imagenet_helmetandNoHelmet
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the project:**
   ```bash
   python count_images.py              # Check dataset
   python resize_and_augment.py        # Preprocess data
   python helmet_detection_model.py    # Train model
   ```

## 📊 Results

- **Validation Accuracy**: 83%
- **Training Accuracy**: 90%
- **Overfitting Gap**: 7%
- **Model Architecture**: MobileNetV2 + Transfer Learning

## 🔗 Dataset Sources

- [Traffic Violation Dataset V3](https://www.kaggle.com/datasets/meliodassourav/traffic-violation-dataset-v3) (Kaggle)
- [Helmet No Helmet Detection Dataset](https://universe.roboflow.com/programa-delfn/helmet-no-helmet-detection-hjdvx/dataset/1) (Roboflow)

## 📁 Project Structure

```
├── README.md                 # Detailed documentation (Vietnamese)
├── requirements.txt          # Python dependencies
├── count_images.py           # Dataset counting script
├── resize_and_augment.py     # Data preprocessing script
├── helmet_detection_model.py # Main model training script
├── create_visualizations.py  # Visualization generation
├── CountImage.ipynb          # Original Jupyter notebooks
├── Resize.ipynb
├── Main.ipynb
└── LICENSE                   # MIT License
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
