import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

def load_cifar10(num_train=5000, num_test=8000):
    """Loads and subsamples the CIFAR-10 dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Subsample training set
    train_images, train_labels = [], []
    for i in range(10):
        class_indices = np.where(y_train == i)[0]
        chosen_indices = np.random.choice(class_indices, num_train // 10, replace=False)
        train_images.append(x_train[chosen_indices])
        train_labels.append(y_train[chosen_indices])
    x_train = np.concatenate(train_images)
    y_train = np.concatenate(train_labels)

    # Subsample test set
    test_images, test_labels = [], []
    for i in range(10):
        class_indices = np.where(y_test == i)[0]
        chosen_indices = np.random.choice(class_indices, num_test // 10, replace=False)
        test_images.append(x_test[chosen_indices])
        test_labels.append(y_test[chosen_indices])
    x_test = np.concatenate(test_images)
    y_test = np.concatenate(test_labels)

    print(f"CIFAR-10 loaded: {x_train.shape[0]} training images, {x_test.shape[0]} test images.")
    return (x_train, y_train.flatten()), (x_test, y_test.flatten())

def load_stl10():
    """Loads the STL-10 dataset."""
    ds_train, ds_test = tfds.load('stl10', split=['train', 'test'], as_supervised=True, shuffle_files=True)

    def to_numpy(ds):
        images, labels = [], []
        for img, lbl in tfds.as_numpy(ds):
            images.append(img)
            labels.append(lbl)
        # STL-10 labels are 1-10, adjust to 0-9
        return np.array(images), np.array(labels) - 1
        
    x_train, y_train = to_numpy(ds_train)
    x_test, y_test = to_numpy(ds_test)
    
    print(f"STL-10 loaded: {x_train.shape[0]} training images, {x_test.shape[0]} test images.")
    return (x_train, y_train), (x_test, y_test)

def preprocess_for_cnn(x_train, x_test):
    """Prepares data for the EfficientNetB0 model."""
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train_preprocessed = efficientnet_preprocess(x_train)
    x_test_preprocessed = efficientnet_preprocess(x_test)
    return x_train_preprocessed, x_test_preprocessed
