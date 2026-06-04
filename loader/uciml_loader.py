import os
import io
import gzip
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from pygments.lexers import configs

DATASET_CONFIGS = {
    "heart_failure": {"url": "https://archive.ics.uci.edu/static/public/145/statlog+heart.zip", "max_samples": 273,
                      "is_zip": True, "file_ext": ".dat", "delim": r"\s+"},
    "breast_cancer": {"url": "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
                      "max_samples": 500, "is_zip": True, "file_ext": ".data", "delim": ","},
    "wine_quality": {"url": "https://archive.ics.uci.edu/static/public/186/wine+quality.zip", "max_samples": 680,
                     "is_zip": True, "file_ext": ".csv", "delim": ";"},
    "avila_bible": {"url": "https://archive.ics.uci.edu/static/public/459/avila.zip", "max_samples": 2000,
                    "is_zip": True, "file_ext": ".txt", "delim": ","},
    "magic_telescope": {"url": "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip",
                        "max_samples": 5000, "is_zip": True, "file_ext": ".data", "delim": ","},
    "room_occupancy": {"url": "https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip",
                       "max_samples": 7500, "is_zip": True, "file_ext": ".txt", "delim": ","},
    "swarm_behaviour": {"url": "https://archive.ics.uci.edu/static/public/501/swarm+behaviour.zip",
                        "max_samples": 20000, "is_zip": True, "file_ext": ".csv", "delim": ","},
    "mini_boone": {"url": "https://archive.ics.uci.edu/static/public/199/miniboone+particle+identification.zip",
                   "max_samples": 120000, "is_zip": True, "file_ext": ".txt", "delim": None},
    "skin_nonskin": {"url": "https://archive.ics.uci.edu/static/public/229/skin+segmentation.zip",
                     "max_samples": 200000, "is_zip": True, "file_ext": ".txt", "delim": r"\s+"},
    "hepmass": {"url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00347/1000_test.csv.gz",
                "max_samples": 3500000, "is_zip": False, "is_gzip": True}
}


def binarize_labels(y_raw):
    unique_classes = np.unique(y_raw)
    if len(unique_classes) == 2:
        return np.where(y_raw == unique_classes[0], -1, 1).astype(int)
    else:
        
        classes, counts = np.unique(y_raw, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        return np.where(y_raw == majority_class, 1, -1).astype(int)


def fetch_raw_bytes(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        return response.read()


def parse_to_dataframe(name, config, raw_bytes):
    """Parses raw content securely, discarding text metadata and mixed dtypes."""
    if config.get("is_zip"):
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
            
            target_file = [f for f in z.namelist() if config["file_ext"] in f.lower() and not f.startswith('__')][0]
            with z.open(target_file) as f:
                content = f.read()
        df = pd.read_csv(io.BytesIO(content), delimiter=config["delim"], header=None, engine='python')
    elif config.get("is_gzip"):
        with gzip.GzipFile(fileobj=io.BytesIO(raw_bytes)) as f:
            df = pd.read_csv(f, header=None if name != "hepmass" else 0)
    else:
        df = pd.read_csv(io.BytesIO(raw_bytes), delimiter=config.get("delim", ","), header=None)

    
    df = df.select_dtypes(include=[np.number, object])
    for col in df.columns:
        if df[col].dtype == object:
            
            if col != df.columns[-1] and col != df.columns[0]:
                df = df.drop(columns=[col])

    return df

def manual_fixes(data_path='data',seed=42):
    rng = np.random.default_rng(seed=seed)
    datasets=['hepmass', ] #'swarm_behaviour','room_occupancy','breast_cancer'
    for name in datasets:
        count = DATASET_CONFIGS[name]['max_samples']
        folder_path=os.path.join(data_path, name)
        if name == "hepmass":
            df=pd.read_csv(os.path.join(folder_path, "all_test.csv"), header=0)
            y_raw = df.iloc[:, 0].to_numpy()
            X_raw = df.iloc[:, 1:].to_numpy()
            y_raw=binarize_labels(y_raw)
            np.savetxt(os.path.join(folder_path, "X.csv"), X_raw, delimiter=",")
            np.savetxt(os.path.join(folder_path, "y.csv"), y_raw, delimiter=",", fmt="%d")
        else:
            print(name,count)
            X=pd.read_csv(os.path.join(folder_path,'X.csv'), delimiter=',', header=None).to_numpy().astype(float)

            y=pd.read_csv(os.path.join(folder_path,'y.csv'), delimiter=',', header=None).to_numpy().reshape(-1).astype(int)

            ds_len = X.shape[0]
            if ds_len>count:
                indices = rng.choice(ds_len, size=count, replace=False)

                X = X[indices]
                y = y[indices]
                np.savetxt(os.path.join(folder_path, "X.csv"), X, delimiter=",")
                np.savetxt(os.path.join(folder_path, "y.csv"), y, delimiter=",", fmt="%d")
            print(X.shape)
            print(y.shape)



def prepare_data(base_dir="data", seed=42):
    rng = np.random.default_rng(seed=seed)

    for name, config in DATASET_CONFIGS.items():
        print(f"\nProcessing target dataset: {name}...")
        folder_path = os.path.join(base_dir, name)
        os.makedirs(folder_path, exist_ok=True)

        try:
            raw_bytes = fetch_raw_bytes(config["url"])
            df = parse_to_dataframe(name, config, raw_bytes)

            
            if name == "hepmass":
                y_raw = df.iloc[:, 0].to_numpy()
                X_raw = df.iloc[:, 1:].to_numpy()
            elif name == "mini_boone":
                
                with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
                    target_file = [f for f in z.namelist() if ".txt" in f][0]
                    lines = z.read(target_file).decode('utf-8').splitlines()
                
                num_pos, num_neg = map(int, lines[0].strip().split())
                raw_matrix = np.loadtxt(lines[1:])
                X_raw = raw_matrix
                y_raw = np.array([1] * num_pos + [-1] * num_neg)
            else:
                X_raw = df.iloc[:, :-1].to_numpy()
                y_raw = df.iloc[:, -1].to_numpy().squeeze()

            
            total_available = X_raw.shape[0]
            max_allowed = config["max_samples"]

            if total_available > max_allowed:
                indices = rng.choice(total_available, size=max_allowed, replace=False)
                X_sampled = X_raw[indices]
                y_sampled = y_raw[indices]
            else:
                X_sampled = X_raw
                y_sampled = y_raw

            y_sampled = binarize_labels(y_sampled)

            
            X_sampled = X_sampled.astype(float)

            np.savetxt(os.path.join(folder_path, "X.csv"), X_sampled, delimiter=",")
            np.savetxt(os.path.join(folder_path, "y.csv"), y_sampled, delimiter=",", fmt="%d")

            print(f"--> Successfully generated clean arrays inside '{folder_path}/'")
            print(f"    Features dimension: {X_sampled.shape} | Targets dimension: {y_sampled.shape}")

        except Exception as e:
            print(f"CRITICAL ERROR processing {name}: {e}")