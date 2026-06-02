import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from models.data_loader import load_all_datasets
from models.Kernels import RBFKernel
from models.Pegasos import PegasosBaseline
from models.SGD import SGDBaseline
from models.SCS import StochasticConjugateSubgradientAlgorithm

import warnings
warnings.filterwarnings('ignore')


from tqdm import tqdm

# def calc_gamma_var(X):
#     variance = X.var()
#     if variance == 0:
#         return 1.0 / X.shape[1]
#     return 1.0 / (X.shape[1] * variance)

def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def run_experiments(data_path="./data", epochs=5, runs=5, scs_batch_size=10):
    print("Loading datasets...")
    datasets = load_all_datasets(data_path=data_path)
    
    results = []
    histories = {}
    etas = [0.0001, 0.001, 0.01, 0.1]

    for dataset_name, (X, y) in datasets.items():
        print(f"\n{'='*50}\nDataset: {dataset_name} | Samples: {X.shape[0]} | Features: {X.shape[1]}\n{'='*50}")

        m_samples = int(X.shape[0]*0.8)
        gamma_var = 1/X.shape[1]
        gammas = [0.2 * gamma_var, gamma_var, 5.0 * gamma_var]
        
        std_max_iter = int(epochs * m_samples)
        scs_max_iter = max(int((epochs * m_samples) / scs_batch_size), 50) 

        # Define configurations to test
        models_configs = {
            "Pegasos": [{"gamma": g} for g in gammas],
            "SGD": [{"gamma": g, "eta": e} for g in gammas for e in etas],
            "SCS": [{"gamma": g} for g in gammas]
        }

        # Calculate total number of inner-loop runs for tqdm
        total_combinations = 0
        for model_name, configs in models_configs.items():
            total_combinations += len(configs)
        total_progress = total_combinations * runs

        dataset_metrics = {"Dataset": dataset_name, "Samples": X.shape[0], "Features": X.shape[1]}
        histories[dataset_name] = {}

        pbar = tqdm(total=total_progress, desc=f"GridSearch {dataset_name}", leave=False, ncols=80)

        for model_name, configs in models_configs.items():
            best_acc = -1.0
            best_time = 0.0
            best_cfg = None

            print(f"[{model_name}] Tuning over {len(configs)} configurations...")

            for cfg in configs:
                accs, times = [], []
                
                # For every run, use a different split of the data (different random_state)
                for run_idx in range(runs):
                    # Split data differently for each run by varying random_state
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=420 + run_idx, stratify=y
                    )

                    X_train, X_test = standardize_data(X_train, X_test)

                    if model_name == "Pegasos":
                        model = PegasosBaseline(kernel=RBFKernel(cfg["gamma"]), verbose=False)
                        kwargs = {"max_iter": std_max_iter, "precompute_kernel": False}
                    elif model_name == "SGD":
                        model = SGDBaseline(kernel=RBFKernel(cfg["gamma"]), eta=cfg["eta"], verbose=False)
                        kwargs = {"max_iter": std_max_iter, "precompute_kernel": False}
                    elif model_name == "SCS":
                        model = StochasticConjugateSubgradientAlgorithm(kernel=RBFKernel(cfg["gamma"]), verbose=False)
                        kwargs = {"max_iter": scs_max_iter, "batch_size": scs_batch_size}
                    
                    kwargs.update({"X_train": X_train, "y_train": y_train, "record_history": False})

                    start_time = time.time()
                    model.fit(**kwargs)
                    elapsed = time.time() - start_time
                    
                    y_pred = model.predict(X_test)
                    accs.append(accuracy_score(y_test, y_pred))
                    times.append(elapsed)
                    pbar.update(1)

                mean_acc = np.mean(accs)
                mean_time = np.mean(times)

                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_time = mean_time
                    best_cfg = cfg

            print(f"[{model_name}] Best Acc: {best_acc:.4f} | Time: {best_time:.2f}s | Params: {best_cfg}")

            # Run once more with best config to get history
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=2137, stratify=y
            )
            X_train, X_test = standardize_data(X_train, X_test)

            if model_name == "Pegasos":
                model = PegasosBaseline(kernel=RBFKernel(best_cfg["gamma"]), verbose=False)
                kwargs = {"max_iter": std_max_iter, "precompute_kernel": False}
            elif model_name == "SGD":
                model = SGDBaseline(kernel=RBFKernel(best_cfg["gamma"]), eta=best_cfg["eta"], verbose=False)
                kwargs = {"max_iter": std_max_iter, "precompute_kernel": False}
            elif model_name == "SCS":
                model = StochasticConjugateSubgradientAlgorithm(kernel=RBFKernel(best_cfg["gamma"]), verbose=False)
                kwargs = {"max_iter": scs_max_iter, "batch_size": scs_batch_size}

            kwargs.update({"X_train": X_train, "y_train": y_train, "record_history": True})
            model.fit(**kwargs)
            
            # Store results
            dataset_metrics[f"{model_name}_Acc"] = best_acc
            dataset_metrics[f"{model_name}_Time"] = best_time
            dataset_metrics[f"{model_name}_Params"] = str(best_cfg)
            histories[dataset_name][model_name] = model.history

        pbar.close()
        results.append(dataset_metrics)

    results_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS")
    print("="*50)
    print(results_df.to_string(index=False))
    
    return results_df, histories

if __name__ == "__main__":
    df, history_dict = run_experiments(data_path="./data1", epochs=3, runs=5, scs_batch_size=10)
    df.to_csv("experiment_results.csv", index=False)
    