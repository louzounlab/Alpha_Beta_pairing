import json
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import torch
import Loader
import os
from torch.utils.data import DataLoader

base_folder = "models/xgboost_run_5fold"
os.makedirs(base_folder, exist_ok=True)

SEED = 42

def dataloader_to_numpy(data_loader):
    """
    Converts a DataLoader to X, y numpy arrays for LightGBM.
    Works for your properties dataset (alpha/beta features + one-hot).
    """
    X_list = []
    y_list = []

    for batch in data_loader:
        # unpack batch
        alpha, beta, va, vb, ja, jb, label, _ = batch
        # alpha, beta are tensors [batch, feature_dim]
        # va,vb,... are one-hot tensors [batch, dim]
        batch_features = torch.cat([alpha, beta, va, vb, ja, jb], dim=1)
        X_list.append(batch_features.cpu().numpy())
        if label is not None:
            y_list.append(label.cpu().numpy())


    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y

def compute_stats(X):
    """
    Compute mean and std only on the first 8 features of X
    X: numpy array [num_samples, num_features]
    Returns:
        mean: shape [8]
        std: shape [8]
    """
    mean = X[:, :8].mean(axis=0)
    std  = X[:, :8].std(axis=0) + 1e-8
    return mean, std

def normalize(X, mean, std):
    """
    Normalize only the first 8 features, keep the rest as-is
    X: numpy array [num_samples, num_features]
    mean, std: shape [8]
    Returns: X_normalized
    """
    X_norm = X.copy()
    X_norm[:, :8] = (X_norm[:, :8] - mean) / std
    return X_norm

def objective_lightgbm(trial, X, y):
    """
    Optuna objective for LightGBM using DataLoader.
    """

    # Suggest hyperparameters
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 50, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
    }

    # K-fold
    k = 2
    skf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    aucs = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # --- compute stats on training fold ---
        mean, std = compute_stats(X_train)

        # --- normalize both train and validation ---
        X_train_norm = normalize(X_train, mean, std)

        X_val_norm = normalize(X_val, mean, std)

        train_data = lgb.Dataset(X_train_norm, label=y_train)
        val_data = lgb.Dataset(X_val_norm, label=y_val, reference=train_data)

        gbm = lgb.train(params, train_data, valid_sets=[train_data, val_data],
                        num_boost_round=params["n_estimators"])

        y_pred_val = gbm.predict(X_val_norm)
        auc = roc_auc_score(y_val, y_pred_val)
        aucs.append(auc)

    return np.mean(aucs)



def objective_xgboost(trial, X, y):
    """
    Optuna objective for XGBoost using same logic as LightGBM.
    """

    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 50, 1000)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",  # faster for large datasets
        "max_leaves": trial.suggest_int("max_leaves", 50, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "random_state": SEED,
    }
    # K-fold
    k = 2
    skf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    aucs = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # --- compute stats on training fold ---
        mean, std = compute_stats(X_train)

        # --- normalize both train and validation ---
        X_train_norm = normalize(X_train, mean, std)
        X_val_norm = normalize(X_val, mean, std)

        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train_norm, label=y_train)
        dval = xgb.DMatrix(X_val_norm, label=y_val)

        # Train model
        gbm = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, 'train'), (dval, 'val')],
            verbose_eval=False
        )

        # Predict and evaluate
        y_pred_val = gbm.predict(dval)
        auc = roc_auc_score(y_val, y_pred_val)
        aucs.append(auc)

    return np.mean(aucs)

def file_to_x(input_file, vj_data):
    pairs = Loader.read_data(input_file)
    batch_size = 64
    # Calculate the total length of all dictionaries combined
    full_dataset = Loader.ChainClassificationDataset(pairs, vj_data)
    full_data_loader = DataLoader(full_dataset, batch_size=batch_size,
                                  shuffle=True, drop_last=True, collate_fn=full_dataset.collate_fn)
    # Convert DataLoader to numpy arrays

    X, y = dataloader_to_numpy(full_data_loader)
    return X, y


def run_gbm(train_file, test_file, hyperparam_file, vj_data, use_xgboost=False):
    """
    Train LightGBM or XGBoost with hyperparameters from file and evaluate on test set.

    Args:
        train_file: Path to training data
        test_file: Path to test data
        hyperparam_file: Path to JSON file with hyperparameters
        use_xgboost: If True, use XGBoost; if False, use LightGBM (default)
    """

    # Load hyperparameters from file
    with open(hyperparam_file, 'r') as f:
        params = json.load(f)

    # Set model-specific defaults
    if use_xgboost:
        if 'objective' not in params:
            params['objective'] = 'binary:logistic'
        if 'eval_metric' not in params:
            params['eval_metric'] = 'auc'
        if 'tree_method' not in params:
            params['tree_method'] = 'hist'
        params['random_state'] = SEED
        model_name = "XGBoost"
        model_ext = ".json"
    else:
        if 'objective' not in params:
            params['objective'] = 'binary'
        if 'metric' not in params:
            params['metric'] = 'auc'
        if 'verbosity' not in params:
            params['verbosity'] = -1
        model_name = "LightGBM"
        model_ext = ".txt"

    X_train, y_train = file_to_x(train_file, vj_data)
    X_test, y_test = file_to_x(test_file, vj_data)
    train_mean, train_std = compute_stats(X_train)
    # K-fold
    k = 5
    skf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    fold_aucs_val = []
    fold_aucs_train = []

    best_auc = 0
    best_model = None
    best_mean = None
    best_std = None

    best_model_path = os.path.join(base_folder, f'best_model{model_ext}')

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_train_split, X_val = X_train[train_idx], X_train[val_idx]
        y_train_split, y_val = y_train[train_idx], y_train[val_idx]

        # --- compute stats on training fold ---
        mean, std = compute_stats(X_train_split)

        # --- normalize both train and validation ---
        X_train_norm = normalize(X_train_split, mean, std)
        X_val_norm = normalize(X_val, mean, std)

        if use_xgboost:
            # XGBoost training
            dtrain = xgb.DMatrix(X_train_norm, label=y_train_split)
            dval = xgb.DMatrix(X_val_norm, label=y_val)

            gbm = xgb.train(
                params,
                dtrain,
                num_boost_round=params["n_estimators"],
                evals=[(dtrain, 'train'), (dval, 'val')],
                verbose_eval=False
            )

            # Predict
            y_pred_train = gbm.predict(dtrain)
            y_pred_val = gbm.predict(dval)
        else:
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train_norm, label=y_train_split)
            val_data = lgb.Dataset(X_val_norm, label=y_val, reference=train_data)

            # Train model
            gbm = lgb.train(
                params,
                train_data,
                valid_sets=[train_data, val_data],
                num_boost_round=params["n_estimators"]
            )

            # Predict on train and validation (FIXED: you had these swapped)
            y_pred_train = gbm.predict(X_train_norm)
            y_pred_val = gbm.predict(X_val_norm)

        auc_train = roc_auc_score(y_train_split, y_pred_train)
        auc_val = roc_auc_score(y_val, y_pred_val)

        fold_aucs_train.append(auc_train)
        fold_aucs_val.append(auc_val)

        print(f'Fold {fold + 1}: Train AUC = {auc_train:.6f}, Val AUC = {auc_val:.6f}')

        # Save the model if the validation AUC is the best so far
        if auc_val > best_auc:
            best_auc = auc_val
            best_model = gbm
            best_mean = mean
            best_std = std
            print(f'New best validation AUC: {best_auc:.6f} found on fold {fold + 1}')

    # Save only the best model after all folds have been processed
    if best_model is not None:
        best_model.save_model(best_model_path)
        print(f'\nBest model saved to {best_model_path} with validation AUC: {best_auc:.6f}')

    # Save fold AUC values
    auc_file_path = os.path.join(base_folder, "auc_values.txt")
    with open(auc_file_path, "w") as f:
        f.write("Fold\tTrain_AUC\tVal_AUC\n")
        for i, (train_auc, val_auc) in enumerate(zip(fold_aucs_train, fold_aucs_val)):
            f.write(f"{i + 1}\t{train_auc:.6f}\t{val_auc:.6f}\n")
        f.write(f"\nMean\t{np.mean(fold_aucs_train):.6f}\t{np.mean(fold_aucs_val):.6f}\n")
        f.write(f"Std\t{np.std(fold_aucs_train):.6f}\t{np.std(fold_aucs_val):.6f}\n")

    print(f"AUC values saved to {auc_file_path}")

    # ========================================
    # Test the best model on the test set
    # ========================================
    print("\n" + "=" * 50)
    print("Evaluating best model on test set...")
    print("=" * 50)

    # Normalize test set using the stats from the best fold
    X_test_norm = normalize(X_test, train_mean, train_std)

    # Predict on test set
    if use_xgboost:
        dtest = xgb.DMatrix(X_test_norm)
        y_pred_test = best_model.predict(dtest)
    else:
        y_pred_test = best_model.predict(X_test_norm)
    test_auc = roc_auc_score(y_test, y_pred_test)

    print(f"Test AUC: {test_auc:.6f}")

    # Save test results
    test_results_path = os.path.join(base_folder, "test_results.txt")
    with open(test_results_path, "w") as f:
        f.write(f"Test AUC: {test_auc:.6f}\n")
        f.write(f"Best Validation AUC: {best_auc:.6f}\n")
        f.write(f"Mean CV Train AUC: {np.mean(fold_aucs_train):.6f}\n")
        f.write(f"Mean CV Val AUC: {np.mean(fold_aucs_val):.6f}\n")

    print(f"Test results saved to {test_results_path}")

    return {
        'test_auc': test_auc,
        'best_val_auc': best_auc,
        'mean_train_auc': np.mean(fold_aucs_train),
        'mean_val_auc': np.mean(fold_aucs_val)
    }