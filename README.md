# Potato Disease Classification Project

A deep learning project that classifies potato leaf diseases using Convolutional Neural Networks (CNN). The model can identify three classes: **Healthy**, **Early Blight**, and **Late Blight** from potato leaf images.

##  Project Overview

This project implements an end-to-end machine learning solution for potato disease classification:

- **Training**: CNN model trained on the PlantVillage dataset
- **Web Application**: Interactive Streamlit app for real-time disease prediction
- **Model Performance**: Achieves high accuracy in classifying potato leaf diseases

##  Features

- **Deep Learning Model**: Custom CNN architecture built with TensorFlow/Keras
- **Data Augmentation**: Enhanced training with image transformations
- **Web Interface**: User-friendly Streamlit application
- **Real-time Prediction**: Upload images and get instant disease classification
- **Confidence Scores**: Prediction confidence percentage for each classification

##  Project Structure

```
DLProjectPotatoDiseaseClassification/
├── app.py                          # Streamlit web application
├── potatoes.h5                     # Trained model (H5 format)
├── requirements.txt                # Python dependencies
├── Training/                       # Training notebooks and data
│   ├── training.ipynb             # Main training notebook
│   ├── training_image_data_gen.ipynb  # Training with data augmentation
│   └── PlantVillage/              # Original dataset
├── models/                        # Saved model versions
│   ├── 1/                         # Model version 1
│   └── 2/                         # Model version 2
└── output/                        # Preprocessed dataset
    ├── train/                     # Training images
    ├── val/                       # Validation images
    └── test/                      # Test images
```

##  Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/VaibhavRox/Potato-Disease-Classification-Project.git
   cd Potato-Disease-Classification-Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - Upload a potato leaf image to get disease prediction

##  Model Details

### Architecture
- **Model Type**: Convolutional Neural Network (CNN)
- **Input Size**: 256x256x3 (RGB images)
- **Classes**: 3 (Healthy, Early Blight, Late Blight)
- **Framework**: TensorFlow/Keras

### Training Details
- **Dataset**: PlantVillage Potato Disease Dataset
- **Training Images**: 1,721 images
- **Validation Images**: 215 images
- **Data Augmentation**: Rotation, horizontal flip, rescaling
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy

### Performance
The model achieves high [95%+] accuracy on the test dataset with robust performance across all three disease categories.

##  Dataset

The project uses the PlantVillage dataset, which contains images of potato leaves in three categories:

1. **Potato___Early_blight**: Leaves affected by early blight disease
2. **Potato___Late_blight**: Leaves affected by late blight disease  
3. **Potato___healthy**: Healthy potato leaves

##  Usage

### Web Application
1. Start the Streamlit app: `streamlit run app.py`
2. Upload a potato leaf image (JPG, JPEG, or PNG)
3. View the prediction and confidence score
4. The app will classify the image as Healthy, Early Blight, or Late Blight

### Training Your Own Model
1. Navigate to the `Training/` directory
2. Open `training_image_data_gen.ipynb` in Jupyter Notebook
3. Run all cells to train a new model
4. The trained model will be saved as `potatoes.h5`

##  Requirements

The project dependencies are minimal and include:

- `streamlit`: Web application framework
- `tensorflow`: Deep learning framework
- `Pillow`: Image processing library
- `numpy`: Numerical computing library

##  Configuration

### Model Path
The application loads the model from `potatoes.h5`. If you train a new model or want to use a different model version, update the path in `app.py`:

```python
model = tf.keras.models.load_model("your_model_path.h5")
```

### Image Preprocessing
Images are automatically:
- Resized to 256x256 pixels
- Normalized to [0,1] range
- Converted to RGB format

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is open source and available under the [MIT License](LICENSE).

##  Acknowledgments

- PlantVillage dataset for providing the training data
- TensorFlow/Keras for the deep learning framework
- Streamlit for the web application framework

##  Contact

For questions or suggestions, please open an issue in the GitHub repository.

---


