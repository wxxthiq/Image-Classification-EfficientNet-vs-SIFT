# Deep Learning vs. Traditional Computer Vision for Image Classification

This project provides a comprehensive comparison between a modern deep learning approach (EfficientNetB0) and traditional computer vision methods (Bag of Words with SIFT) for image classification. The models are evaluated on two benchmark datasets: CIFAR-10 and STL-10.

The primary finding is that the deep learning model, enhanced by transfer learning and fine-tuning, significantly outperforms the traditional feature-based methods on both datasets.

## Key Features

* **Deep Learning Model**: Implementation and fine-tuning of EfficientNetB0 using transfer learning from ImageNet pre-trained weights.
* **Traditional CV Models**: Implementation of the Bag of Words (BoW) pipeline using SIFT with two different keypoint detectors:
    * Difference of Gaussians (DoG).
    * Harris-Laplace.
* **Datasets**: CIFAR-10 and STL-10, with custom preprocessing, subsampling, and data augmentation to handle their different characteristics.
* **Techniques**:
    * Systematic hyperparameter tuning using Grid Search.
    * Data augmentation (rotations, shifts, flips) to improve model robustness.
    * Strategic fine-tuning by unfreezing layers and using adaptive learning rates.
    * Principal Component Analysis (PCA) to reduce SIFT descriptor dimensionality while retaining 90% of the variance.
    * Support Vector Machine (SVM) for classification in the BoW pipeline.

---

## Methodology

### Deep Learning: EfficientNetB0

The deep learning approach uses the EfficientNetB0 architecture, a powerful Convolutional Neural Network (CNN) designed for efficiency.

* **Transfer Learning**: The model was pre-trained on ImageNet, allowing it to leverage learned features for better performance on smaller datasets. The base layers were initially frozen, and a custom head was added for the 10-class classification task. This head includes a Global Average Pooling layer, a Dense layer with ReLU activation, and a final SoftMax output layer.
* **Data Preprocessing**: Input images from both datasets were resized to $224 \times 224$ pixels to match the model's input requirements. Pixel values were normalized to align with the pre-trained weight distributions.
* **Training & Fine-Tuning**: A grid search was performed to find the optimal hyperparameters for learning rate, dropout rate, and batch size. The best model was then fine-tuned by unfreezing the top layers (25 for CIFAR-10, 10 for STL-10) and training with an adaptive learning rate schedule (`ReduceLROnPlateau`) for more precise weight adjustments.

### Traditional CV: Bag of Words (BoW) + SIFT

The traditional approach follows the BoW pipeline, which consists of five stages:

1.  **Keypoint Detection**: Salient points in images were detected using two methods:
    * **Difference of Gaussians (DoG)**: An efficient method for detecting scale-invariant keypoints by finding extrema in the DoG scale space.
    * **Harris-Laplace**: A method that combines multi-scale Harris corner detection with Laplacian-of-Gaussian (LoG) scale selection for precise localization and scale invariance.
2.  **Feature Description**: SIFT descriptors were extracted from each keypoint.
3.  **Codebook Generation**: The extracted descriptors were clustered using **MiniBatchKMeans** to create a visual vocabulary (codebook) with vocabulary sizes of 750, 1000, and 1250 being tested.
4.  **Histogram Representation**: Each image was represented as a histogram of its visual words. **Spatial Pyramid Matching** was used to retain coarse spatial information.
5.  **Classification**: A Support Vector Machine (SVM) with an RBF kernel was trained on the L2-normalized histograms to classify the images.

---

## Performance Results

The fine-tuned EfficientNetB0 model demonstrated vastly superior performance compared to both traditional computer vision methods. While traditional methods performed better on classes with distinct edges like "ship" and "airplane," they struggled with textured classes like "cat" and "dog," a challenge the deep learning model easily overcame.

| Method                      | CIFAR-10 Test Accuracy | STL-10 Test Accuracy |
| :-------------------------- | :--------------------: | :------------------: |
| **EfficientNetB0 (Fine-Tuned)** |       **87.35%** |      **95.91%** |
| SIFT + Harris-Laplace + SVM |         40.31%         |        50.51%        |
| SIFT + DoG + SVM            |         25.59%         |        38.77%        |

<br>

---

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

---

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

# Install dependencies
pip install -r requirements.txt

# Run an experiment using main.py
# Example 1: Fine-tune the CNN on CIFAR-10
python main.py --dataset cifar10 --model cnn --cnn_mode finetune

# Example 2: Run the traditional CV experiment with SIFT+DoG on STL-10
python main.py --dataset stl10 --model cv --cv_method dog
