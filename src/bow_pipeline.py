import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def build_vocabulary(images, extractor_fn, extractor_params, n_clusters=750, use_pca=True, pca_variance=0.9):
    """Builds a visual vocabulary using KMeans clustering."""
    print("Extracting features for vocabulary...")
    all_descriptors = []
    for img in tqdm(images):
        _, descriptors = extractor_fn(img, **extractor_params)
        if descriptors is not None:
            all_descriptors.append(descriptors)
    
    if not all_descriptors:
        raise ValueError("No descriptors were extracted. Check feature extraction parameters.")

    all_descriptors = np.vstack(all_descriptors)
    
    # Normalize and apply PCA
    scaler = StandardScaler()
    all_descriptors = scaler.fit_transform(all_descriptors)
    
    pca = None
    if use_pca:
        print(f"Applying PCA to retain {pca_variance*100}% variance...")
        pca = PCA(n_components=pca_variance, random_state=42)
        all_descriptors = pca.fit_transform(all_descriptors)
        print(f"PCA resulted in {pca.n_components_} components.")

    print(f"Building vocabulary with {n_clusters} clusters...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init='auto', random_state=42, batch_size=4096)
    kmeans.fit(all_descriptors)
    
    return kmeans, pca, scaler

def create_histograms(images, extractor_fn, extractor_params, kmeans, pca, scaler):
    """Creates BoW histograms for a set of images."""
    print("Creating BoW histograms...")
    histograms = []
    for img in tqdm(images):
        kps, des = extractor_fn(img, **extractor_params)

        if des is None or len(kps) == 0:
            histograms.append(np.zeros(kmeans.n_clusters))
            continue

        des = scaler.transform(des)
        if pca is not None:
            des = pca.transform(des)
        
        words = kmeans.predict(des)
        hist, _ = np.histogram(words, bins=range(kmeans.n_clusters + 1))
        
        # L2 Normalization
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
            
        histograms.append(hist)
        
    return np.array(histograms)
