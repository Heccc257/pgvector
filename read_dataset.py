import numpy as np
def read_dataset(file_path : str, max_elements : int =None) -> np.array:
    """
    读取数据集，支持 TSV 和 FVECs 格式，返回 np.array。

    参数:
    - file_path (str): 输入文件路径，支持 `.tsv` 和 `.fvecs`
    - max_elements (int, 可选): 读取的最大行数（用于限制数据规模）

    返回:
    - data (np.array): 读取的数据 (float32)
    """
    data = []
    num_elements = 0

    if file_path.endswith(".tsv"):

        with open(file_path, "r") as f:
            for line in f:
                if max_elements and num_elements >= max_elements:
                    break
                line = line.strip().split("\t")
                data.append([float(x) for x in line])
                num_elements += 1
                if num_elements % 10000 == 0:
                    print(f"Reading TSV data: {num_elements}")
        data = np.array(data, dtype=np.float32)

    elif file_path.endswith(".fvecs"):

        with open(file_path, "rb") as f:
            while True:
                d_bytes = f.read(4)  # 读取维度信息
                if not d_bytes:
                    break
                if max_elements and num_elements >= max_elements:
                    break
                d = np.frombuffer(d_bytes, dtype=np.int32)[0]
                vec = np.frombuffer(f.read(4 * d), dtype=np.float32)

                data.append(vec)
                num_elements += 1
                
                if num_elements % 10000 == 0:
                    print(f"Reading FVECs data: {num_elements}")

        data = np.array(data, dtype=np.float32)

    else:
        raise ValueError("Unsupported file format. Please use .tsv or .fvecs")
    return data