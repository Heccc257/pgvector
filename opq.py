import argparse
import json
import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import orthogonal_procrustes
import random
from tqdm import tqdm

def compute_distortion(X, Y):
    return np.sum((X - Y) ** 2)

def update_sub_codebooks(X_rot, sub_codebooks, sub_indexes, M, sub_dim, k):
    n_samples = X_rot.shape[0]
    for m in range(M):
        sub_X = X_rot[:, m * sub_dim:(m + 1) * sub_dim]
        for j in range(k):
            points_in_cluster = sub_X[sub_indexes[:, m] == j]
            if len(points_in_cluster) > 0:
                sub_codebooks[m][j] = np.mean(points_in_cluster, axis=0)
        for i in range(n_samples):
            distances = np.linalg.norm(sub_X[i] - sub_codebooks[m], axis=1)
            sub_indexes[i, m] = np.argmin(distances)
    return sub_codebooks, sub_indexes

def optimize_product_quantization(X, M, k, max_iter=10, profile_distortion=False):
    n_samples, n_features = X.shape
    sub_dim = n_features // M
    R = np.eye(n_features)
    X_rot = X @ R
    sub_codebooks = []
    sub_indexes = np.zeros((n_samples, M), dtype=np.int32)
    for m in range(M):
        sub_X = X_rot[:, m * sub_dim:(m + 1) * sub_dim]
        kmeans = KMeans(n_clusters=k, n_init='auto').fit(sub_X)
        sub_codebooks.append(kmeans.cluster_centers_)
        sub_indexes[:, m] = kmeans.labels_
    Y = np.zeros_like(X_rot)
    for m in range(M):
        sub_centers = sub_codebooks[m][sub_indexes[:, m]]
        Y[:, m * sub_dim:(m + 1) * sub_dim] = sub_centers
    distortions = []
    if profile_distortion:
        distortion = compute_distortion(X_rot, Y)
        print(f'Initial Distortion: {distortion}')
        distortions = [distortion]
    for iter_num in tqdm(range(max_iter), desc='Optimizing'):
        if iter_num > 0:
            X_rot = X @ R
            sub_codebooks, sub_indexes = update_sub_codebooks(X_rot, sub_codebooks, sub_indexes, M, sub_dim, k)
            Y = np.zeros_like(X_rot)
            for m in range(M):
                sub_centers = sub_codebooks[m][sub_indexes[:, m]]
                Y[:, m * sub_dim:(m + 1) * sub_dim] = sub_centers
            if profile_distortion:
                distortion = compute_distortion(X_rot, Y)
                distortions.append(distortion)
        R, _ = orthogonal_procrustes(X, Y)
        if profile_distortion:
            for i, distortion in enumerate(distortions):
                print(f'Iter {i}: Distortion: {distortion}')
    return R, sub_codebooks, sub_indexes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize Product Quantization and Save R to File')
    parser.add_argument('--config_file', type=str, required=True, help='Path to configuration JSON file')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    if config.get("opq", 0) == 1:
        np.random.seed(0)
        random.seed(0)
        max_elements, num_elements = config.get('max_elements', 100000), 0
        data = []
        with open(config['data_path'], 'r') as f:
            for line in f:
                if num_elements >= max_elements:
                    break
                line = line.strip().split('\t')
                data.append([float(x) for x in line])
                num_elements += 1
                if num_elements % 10000 == 0:
                    print(f'Reading data: {num_elements}')
        data = np.array(data)
        indices = np.random.choice(data.shape[0], size=config['sample_size'], replace=False)
        X = np.take(data, indices, axis=0)
        R, _, _, _ = optimize_product_quantization(X, config['M'], config['k'], config['max_iter'])
        R.astype(np.float32).tofile(config['opq_matrix_file'])
        print(f'Saved R to {config["opq_matrix_file"]}')
    else:
        print('OPQ is not enabled in the configuration file., skip...')