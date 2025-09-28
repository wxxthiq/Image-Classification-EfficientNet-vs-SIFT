import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf

# Import our refactored modules
from src.data_loader import load_cifar10, load_stl10, preprocess_for_cnn
from src.feature_extractors import extract_sift_dog, extract_harris_laplace
from src.bow_pipeline import build_vocabulary, create_histograms
from src.cnn_model import build_cnn_model, train_cnn_model
from src.utils import plot_training_history

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def run_cv_experiment(dataset_name, method):
    # --- 1. Load Data ---
    if dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = load_cifar10()
        harris_sigmas = [1.0, 1.5, 2.0]
    else:
        (x_train, y_train), (x_test, y_test) = load_stl10()
        harris_sigmas = [1.0, 2.0, 4.0]
        
    # --- 2. Define Extractor ---
    if method == 'dog':
        extractor_fn = extract_sift_dog
        # Define hyperparameter grid for SIFT-DoG
        params_grid = {
            'contrast_threshold': [0.02, 0.04, 0.08],
            'edge_threshold': [7.5, 10, 12.5]
        }
    elif method == 'harris':
        extractor_fn = extract_harris_laplace
        # Define hyperparameter grid for SIFT-Harris
        params_grid = {
            'block_size': [3], # Simplified from report for demonstration
            'k': [0.04, 0.06],
            'sigma_values': [harris_sigmas],
            'contrast_threshold': [0.02, 0.04],
            'edge_threshold': [7.5, 10]
        }
    else:
        raise ValueError("Invalid CV method specified.")
        
    vocab_sizes = [750, 1000] # Simplified from report
    svm_cs = [1.0, 10.0]     # Simplified from report
    
    best_accuracy = 0
    best_config = {}
    
    # --- 3. Grid Search ---
    # This is a simplified grid search loop for demonstration
    # You can expand it to match the report's full grid
    extractor_params = list(params_grid.values())[0] # Using first param set as example
    
    for vocab_size in vocab_sizes:
        print(f"\n--- Running Grid Search: Vocab Size = {vocab_size} ---")
        
        # Build vocabulary on training data
        kmeans, pca, scaler = build_vocabulary(x_train, extractor_fn, dict(zip(params_grid.keys(), [extractor_params])), n_clusters=vocab_size)
        
        # Create histograms
        train_hist = create_histograms(x_train, extractor_fn, dict(zip(params_grid.keys(), [extractor_params])), kmeans, pca, scaler)
        test_hist = create_histograms(x_test, extractor_fn, dict(zip(params_grid.keys(), [extractor_params])), kmeans, pca, scaler)
        
        for C in svm_cs:
            print(f"Training SVM with C={C}...")
            svm = SVC(C=C, kernel='rbf', gamma='scale', random_state=42)
            svm.fit(train_hist, y_train)
            
            y_pred = svm.predict(test_hist)
            acc = accuracy_score(y_test, y_pred)
            
            print(f"Config: vocab={vocab_size}, C={C} -> Accuracy: {acc:.4f}")
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_config = {'vocab_size': vocab_size, 'C': C, 'params': extractor_params}

    print("\n--- Best CV Result ---")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Best Config: {best_config}")

def run_cnn_experiment(dataset_name, mode):
    # --- 1. Load and Preprocess Data ---
    if dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = load_cifar10()
        input_shape = (32, 32, 3)
    else:
        (x_train, y_train), (x_test, y_test) = load_stl10()
        input_shape = (96, 96, 3)
        
    x_train, x_test = preprocess_for_cnn(x_train, x_test)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # --- 2. Build and Train Model ---
    if mode == 'baseline':
        print("\n--- Training CNN Baseline Model ---")
        model = build_cnn_model(input_shape=input_shape, num_classes=10)
        history = train_cnn_model(model, x_train, y_train, x_val, y_val, batch_size=32, epochs=25)
        plot_training_history(history, f"{dataset_name.upper()} CNN Baseline")
        
    elif mode == 'finetune':
        print("\n--- Fine-tuning Best CNN Model ---")
        # Using best params from report for fine-tuning
        best_lr = 0.001
        best_dropout = 0.2
        best_batch_size = 128
        unfreeze_layers = 25 if dataset_name == 'cifar10' else 10

        model = build_cnn_model(input_shape=input_shape, num_classes=10, learning_rate=best_lr, dropout_rate=best_dropout)
        history = train_cnn_model(model, x_train, y_train, x_val, y_val, 
                                  batch_size=best_batch_size, epochs=40, 
                                  finetune=True, unfreeze_layers=unfreeze_layers)
        plot_training_history(history, f"{dataset_name.upper()} CNN Fine-tuned")
        
    # --- 3. Evaluate ---
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
    
    y_pred = model.predict(x_test)
    print("\nClassification Report:")
    report = classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Image Classification Experiments")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'stl10'], help='Dataset to use')
    parser.add_argument('--model', type=str, required=True, choices=['cnn', 'cv'], help='Model type: cnn or cv (traditional)')
    parser.add_argument('--cv_method', type=str, choices=['dog', 'harris'], default='dog', help='Method for CV keypoint detection')
    parser.add_argument('--cnn_mode', type=str, choices=['baseline', 'finetune'], default='baseline', help='Mode for CNN training')
    
    args = parser.parse_args()

    if args.model == 'cv':
        run_cv_experiment(args.dataset, args.cv_method)
    elif args.model == 'cnn':
        run_cnn_experiment(args.dataset, args.cnn_mode)
