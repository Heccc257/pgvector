import hnswlib
import numpy as np
import faiss
import sys
import pickle
import struct
import json
import argparse
import os

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode Data with Product Quantization')
    parser.add_argument('--config_file', type=str, required=True, help='Path to configuration JSON file')
    args = parser.parse_args()

    config = load_config(args.config_file)

    pq_m_list = config["pq_m_list"]
    data_path = config["data_path"]
    nbits_list = config["nbits_list"]
    output_dir = config["output_dir"]
    max_elements = config["max_elements"]
    d = config["d"]
    opq = config.get("opq", 0)
    opq_matrix_file = config.get("opq_matrix_file", "")

    data = []
    num_elements = 0

    with open(data_path, 'r') as f:
        for line in f:
            if num_elements >= max_elements:
                break
            line = line.strip().split('\t')
            data.append([float(x) for x in line])
            num_elements += 1
            if num_elements % 10000 == 0:
                print(f'Reading data: {num_elements}')
    data = np.array(data)

    if opq:
        R = np.fromfile(opq_matrix_file, dtype=np.float32).reshape(d, d)
        data = data @ R

    for pq_m in pq_m_list:
        for nbits in nbits_list:
            pq = faiss.IndexPQ(d, pq_m, nbits)
            pq.train(data)
            pq.add(data)
            encodes = pq.sa_encode(data)
            decodes = pq.sa_decode(encodes).reshape(-1)
            cents = faiss.vector_float_to_array(pq.pq.centroids).reshape(-1)

            integer_data = [max_elements, d, pq_m, nbits]
            file_name = f'encoded_data_{config.get("opq")}_{max_elements}_{pq_m}_{nbits}'
            with open(os.path.join(config["output_dir"], file_name), 'wb') as file:
                header = struct.pack('iiii', *integer_data)
                file.write(header)
                for val in cents:
                    file.write(struct.pack('f', val))

    print('Done')