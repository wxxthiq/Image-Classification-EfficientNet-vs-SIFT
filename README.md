# Deep Learning vs. Traditional Computer Vision for Image Classification

This project provides a comprehensive comparison between a modern deep learning approach (EfficientNetB0) and traditional computer vision methods (Bag of Words with SIFT) for image classification. The models are evaluated on two benchmark datasets: CIFAR-10 and STL-10.

The primary finding is that the deep learning model, enhanced by transfer learning and fine-tuning, significantly outperforms the traditional feature-based methods on both datasets.

## Key Features

* **Deep Learning Model**: Implementation and fine-tuning of EfficientNetB0 using transfer learning from ImageNet pre-trained weights.
* **Traditional CV Models**: Implementation of the Bag of Words (BoW) pipeline using:
    * SIFT with Difference of Gaussians (DoG) for keypoint detection.
    * SIFT with Harris-Laplace for corner-based keypoint detection.
* **Datasets**: CIFAR-10 and STL-10, with custom preprocessing, subsampling, and data augmentation to handle their different characteristics.
* **Techniques**:
    * Systematic hyperparameter tuning using Grid Search.
    * Data augmentation to improve model robustness.
    * Strategic fine-tuning by unfreezing layers and using adaptive learning rates.
    * Principal Component Analysis (PCA) for dimensionality reduction of SIFT descriptors.
    * Support Vector Machine (SVM) for classification in the BoW pipeline.

## Methodology

### 1. Deep Learning: EfficientNetB0

The deep learning approach uses the EfficientNetB0 architecture, a powerful and efficient Convolutional Neural Network (CNN).

* **Transfer Learning**: The model was pre-trained on ImageNet. The base layers were initially frozen, and a custom head was added for the 10-class classification task. This head includes a Global Average Pooling layer, a Dense layer with ReLU activation, and a final SoftMax output layer.
* **Data Preprocessing**: Input images from both datasets were resized to $224 \times 224$ pixels to match the model's input requirements. Pixel values were normalized to align with the pre-trained weight distributions.
* **Training & Fine-Tuning**:
    * A grid search was performed to find the optimal hyperparameters for learning rate, dropout rate, and batch size.
    * The best model was then fine-tuned by unfreezing the top layers (25 for CIFAR-10, 10 for STL-10) and training with an adaptive learning rate schedule (`ReduceLROnPlateau`).

### 2. Traditional CV: Bag of Words (BoW) + SIFT

The traditional approach follows the BoW pipeline, which consists of five stages:

1.  **Keypoint Detection**: Salient points in images were detected using two methods:
    * **Difference of Gaussians (DoG)**: An efficient method for detecting scale-invariant keypoints.
    * **Harris-Laplace**: A method that combines multi-scale Harris corner detection with Laplacian-of-Gaussian (LoG) scale selection.
2.  **Feature Description**: SIFT descriptors were extracted from each keypoint.
3.  **Codebook Generation**: The extracted descriptors were clustered using MiniBatchKMeans to create a visual vocabulary (codebook).
4.  **Histogram Representation**: Each image was represented as a histogram of its visual words. Spatial Pyramid Matching was used to retain coarse spatial information.
5.  **Classification**: A Support Vector Machine (SVM) with an RBF kernel was trained on the histograms to classify the images.

## Performance Results

The fine-tuned EfficientNetB0 model demonstrated vastly superior performance compared to both traditional computer vision methods.

| Method                       | CIFAR-10 Test Accuracy | STL-10 Test Accuracy |
| ---------------------------- | ---------------------- | -------------------- |
| **EfficientNetB0 (Fine-Tuned)** | **87.35%** | **95.91%** |
| SIFT + Harris-Laplace + SVM  | 40.31% | 50.51% |
| SIFT + DoG + SVM             | 25.59% | 38.77% |

<br>

**Fine-Tuned Model Performance on CIFAR-10 and STL-10:**

*(You can insert the accuracy-per-epoch graphs from Figure 2 in your report here. Replace `path/to/your/image.png` with the actual path or URL to the image.)*

![Fine-Tuned Model Performance](path/to/your/image.png)

## Technologies Used

* **Language**: Python
* **Libraries**:
    * TensorFlow & Keras
    * Scikit-learn
    * OpenCV
    * NumPy
    * Pandas
    * Matplotlib
    * TensorFlow Datasets

## Setup and Usage

To replicate the results, clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# It is recommended to use a virtual environment
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies from the provided code appendix
pip install tensorflow scikit-learn opencv-python numpy pandas matplotlib tensorflow-datasets

# Run the experiments (based on the Python scripts in the appendix)
# Example:
# python cifar10_cnn_finetuning.py
# python stl10_cv_harris_laplace.py
