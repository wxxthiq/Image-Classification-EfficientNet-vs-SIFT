# Deep Learning vs. Traditional Computer Vision for Image Classification

[cite_start]This project provides a comprehensive comparison between a modern deep learning approach (EfficientNetB0) [cite: 10] [cite_start]and traditional computer vision methods (Bag of Words with SIFT) [cite: 156, 157] for image classification. [cite_start]The models are evaluated on two benchmark datasets: CIFAR-10 and STL-10[cite: 4].

[cite_start]The primary finding is that the deep learning model, enhanced by transfer learning and fine-tuning, significantly outperforms the traditional feature-based methods on both datasets[cite: 228, 229].

## Key Features

* [cite_start]**Deep Learning Model**: Implementation and fine-tuning of EfficientNetB0 using transfer learning from ImageNet pre-trained weights[cite: 11, 12].
* [cite_start]**Traditional CV Models**: Implementation of the Bag of Words (BoW) pipeline [cite: 157] using:
    * [cite_start]SIFT with Difference of Gaussians (DoG) for keypoint detection[cite: 156, 172].
    * [cite_start]SIFT with Harris-Laplace for corner-based keypoint detection[cite: 156, 179].
* [cite_start]**Datasets**: CIFAR-10 and STL-10 [cite: 4][cite_start], with custom preprocessing, subsampling [cite: 6, 7][cite_start], and data augmentation to handle their different characteristics[cite: 70].
* **Techniques**:
    * [cite_start]Systematic hyperparameter tuning using Grid Search[cite: 73].
    * [cite_start]Data augmentation to improve model robustness[cite: 70].
    * [cite_start]Strategic fine-tuning by unfreezing layers and using adaptive learning rates[cite: 98].
    * [cite_start]Principal Component Analysis (PCA) for dimensionality reduction of SIFT descriptors[cite: 159].
    * [cite_start]Support Vector Machine (SVM) for classification in the BoW pipeline[cite: 166].

## Methodology

### 1. Deep Learning: EfficientNetB0

[cite_start]The deep learning approach uses the EfficientNetB0 architecture, a powerful and efficient Convolutional Neural Network (CNN)[cite: 11].

* [cite_start]**Transfer Learning**: The model was pre-trained on ImageNet[cite: 12]. [cite_start]The base layers were initially frozen [cite: 81][cite_start], and a custom head was added for the 10-class classification task[cite: 17, 61]. [cite_start]This head includes a Global Average Pooling layer [cite: 16, 62][cite_start], a Dense layer with ReLU activation [cite: 63][cite_start], and a final SoftMax output layer[cite: 17, 54].
* [cite_start]**Data Preprocessing**: Input images from both datasets were resized to $224 \times 224$ pixels to match the model's input requirements[cite: 61, 68]. [cite_start]Pixel values were normalized to align with the pre-trained weight distributions[cite: 69].
* **Training & Fine-Tuning**:
    * [cite_start]A grid search was performed to find the optimal hyperparameters for learning rate, dropout rate, and batch size[cite: 73].
    * [cite_start]The best model was then fine-tuned by unfreezing the top layers (25 for CIFAR-10, 10 for STL-10) [cite: 81] [cite_start]and training with an adaptive learning rate schedule (`ReduceLROnPlateau`)[cite: 82].

### 2. Traditional CV: Bag of Words (BoW) + SIFT

[cite_start]The traditional approach follows the BoW pipeline, which consists of five stages[cite: 157]:

1.  **Keypoint Detection**: Salient points in images were detected using two methods:
    * [cite_start]**Difference of Gaussians (DoG)**: An efficient method for detecting scale-invariant keypoints[cite: 156, 172].
    * [cite_start]**Harris-Laplace**: A method that combines multi-scale Harris corner detection with Laplacian-of-Gaussian (LoG) scale selection[cite: 156, 179].
2.  [cite_start]**Feature Description**: SIFT descriptors were extracted from each keypoint[cite: 157].
3.  [cite_start]**Codebook Generation**: The extracted descriptors were clustered using MiniBatchKMeans to create a visual vocabulary (codebook)[cite: 161].
4.  [cite_start]**Histogram Representation**: Each image was represented as a histogram of its visual words[cite: 157]. [cite_start]Spatial Pyramid Matching was used to retain coarse spatial information[cite: 164].
5.  [cite_start]**Classification**: A Support Vector Machine (SVM) with an RBF kernel was trained on the histograms to classify the images[cite: 166].

## Performance Results

[cite_start]The fine-tuned EfficientNetB0 model demonstrated vastly superior performance compared to both traditional computer vision methods[cite: 228].

| Method                       | CIFAR-10 Test Accuracy | STL-10 Test Accuracy |
| ---------------------------- | ---------------------- | -------------------- |
| **EfficientNetB0 (Fine-Tuned)** | [cite_start]**87.35%** [cite: 99]  | [cite_start]**95.91%** [cite: 101, 144] |
| SIFT + Harris-Laplace + SVM  | [cite_start]40.31% [cite: 198, 214] | [cite_start]50.51% [cite: 208, 217] |
| SIFT + DoG + SVM             | [cite_start]25.59% [cite: 184, 214] | [cite_start]38.77% [cite: 188, 217] |

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
