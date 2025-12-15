import torch
import Models
import Loader
from torch.utils.data import DataLoader
import Trainer
import numpy as np
import v_j
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold, train_test_split
import optuna
import pandas as pd
import os
import json
from matplotlib.font_manager import FontProperties
import argparse
import random


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# base_folder = os.getcwd()
base_folder = "models/pMHC-final_model"
os.makedirs(base_folder, exist_ok=True)
font = FontProperties()
font.set_family('serif')
font_size = 16
font.set_size(font_size)
SEED = 42

def read_dictionaries_from_file(file_path):
    """
    Read dictionaries from a JSON file.
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        tuple: Four dictionaries containing counts for va, vb, ja, and jb.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['va_counts'], data['vb_counts'], data['ja_counts'], data['jb_counts']


def process_data_with_k_fold(file_path, batch_size, k=5):
    """
    Process data using K-Fold cross-validation.
    Args:
        file_path (str): Path to the data file.
        batch_size (int): Batch size for DataLoader.
        k (int, optional): Number of folds for K-Fold cross-validation. Default is 5.
    Returns:
        list: List of tuples containing datasets and data loaders for each fold.
    """
    pairs = Loader.read_data(file_path)
    va_counts, vb_counts, ja_counts, jb_counts = read_dictionaries_from_file('filtered_counters.json')
    vj_data = (va_counts, vb_counts, ja_counts, jb_counts)
    # Calculate the total length of all dictionaries combined
    len_one_hot = len(va_counts) + len(vb_counts) + len(ja_counts) + len(jb_counts)
    if k == 1:
        train_dataset = Loader.ChainClassificationDataset(pairs, vj_data)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size,
                                       shuffle=True, drop_last=True, collate_fn=Loader.collate_fn)

        test_dataset = Loader.ChainClassificationDataset(pairs, vj_data)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size,
                                      shuffle=False, drop_last=True, collate_fn=Loader.collate_fn)

        fold_data = [(train_dataset, train_data_loader, pairs,
                      test_dataset, test_data_loader, pairs, len_one_hot)]
        return fold_data

    # Initialize the KFold class
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    # Lists to store the datasets and data loaders for each fold
    fold_data = []

    for train_index, test_index in kf.split(pairs):
        train_data = [pairs[i] for i in train_index]
        test_data = [pairs[i] for i in test_index]

        # Create the dataset and DataLoader for the training set
        train_dataset = Loader.ChainClassificationDataset(train_data, vj_data)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size,
                                       shuffle=True, drop_last=True, collate_fn=Loader.collate_fn)

        # Create the dataset and DataLoader for the testing set
        test_dataset = Loader.ChainClassificationDataset(test_data, vj_data)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size,
                                      shuffle=False, drop_last=True, collate_fn=Loader.collate_fn)

        # Store the datasets and data loaders for this fold
        fold_data.append(
            (train_dataset, train_data_loader, train_data, test_dataset, test_data_loader, test_data, len_one_hot))

    return fold_data


def train_models(vocab_size, hyperparameters, batch_size, acid_2_ix, train_data_loader,
                 len_one_hot, model_type, test_data_loader=None, ix_2_acid=None):
    """
    Train models with the given hyperparameters and data loaders.
    Args:
        vocab_size (int): Size of the vocabulary.
        hyperparameters (tuple): Tuple containing hyperparameters for the models.
        batch_size (int): Batch size for training.
        acid_2_ix (dict): Dictionary mapping amino acids to indices.
        train_data_loader (DataLoader): DataLoader for training data.
        len_one_hot (int): Length of the one-hot encoded vectors.
        model_type (str): Type of model to train.
        test_data_loader (DataLoader, optional): DataLoader for test data. Default is None.
        ix_2_acid (dict): Dictionary mapping indices to amino acid.
    Returns:
        tuple: Trained model and encoders/decoders for alpha and beta chains.
    """
    (embed_size, hidden_size, num_layers, latent_size, weight_decay_encoder, dropout_prob, layer_norm,
     nhead, dim_feedforward, weight_decay_cl, dropout_prob_cl, norm_cl, losses_weight, lr_encoder, lr_cl) = hyperparameters
    # Initialize the models
    if model_type == "BERT":
        model = Models.FFNN(768 * 2 + len_one_hot, dropout_prob_cl, norm_cl)
        model.to(DEVICE)
        Trainer.train_model_bert(tokenizer, tcrbert, model, train_data_loader, test_data_loader, ix_2_acid, DEVICE, base_folder, weight_decay_cl, lr_cl)
        return model, (None, None), (None, None)
    model = Models.FFNN(latent_size * 2 + len_one_hot, dropout_prob_cl, norm_cl)
    model.to(DEVICE)
    if model_type == "ALSTM":
        ALSTM = True
        BiLSTM = False
    elif model_type == "BiLSTM":
        ALSTM = False
        BiLSTM = True
    elif model_type == "ABiLSTM":
        ALSTM = True
        BiLSTM = True
    else:
        ALSTM = False
        BiLSTM = False
    alpha_encoder = Models.EncoderLstm(vocab_size, embed_size, hidden_size, latent_size, dropout_prob, layer_norm,
                                       num_layers, BiLSTM)
    beta_encoder = Models.EncoderLstm(vocab_size, embed_size, hidden_size, latent_size, dropout_prob, layer_norm,
                                      num_layers, BiLSTM)
    alpha_decoder = Models.DecoderLstm(vocab_size, embed_size, hidden_size, latent_size, dropout_prob, layer_norm,
                                       num_layers, ALSTM, BiLSTM)
    beta_decoder = Models.DecoderLstm(vocab_size, embed_size, hidden_size, latent_size, dropout_prob, layer_norm,
                                      num_layers, ALSTM, BiLSTM)

    alpha_encoder.to(DEVICE)
    beta_encoder.to(DEVICE)
    alpha_decoder.to(DEVICE)
    beta_decoder.to(DEVICE)

    Trainer.train_model(model, (alpha_encoder, alpha_decoder),
                        (beta_encoder, beta_decoder), train_data_loader, test_data_loader, DEVICE,
                        base_folder, batch_size, acid_2_ix, weight_decay_encoder, weight_decay_cl, losses_weight,
                        lr_encoder, lr_cl)
    return model, (alpha_encoder, alpha_decoder), (beta_encoder, beta_decoder)


def evaluate_model(model, encoders, data_loader):
    """
    Evaluate the model on the given data loader.
    Args:
        model (nn.Module): The model to evaluate.
        encoders (tuple): Tuple containing the alpha and beta encoders.
        data_loader (DataLoader): DataLoader for the evaluation data.
    Returns:
        tuple: AUC score, all labels, all predicted probabilities, alpha sequences, beta sequences,
               top positive alpha sequences, top positive beta sequences, bottom negative alpha sequences,
               bottom negative beta sequences.
    """
    (encoder_alpha, encoder_beta) = encoders
    correct = 0
    total = 0
    model.eval()
    encoder_alpha.eval()
    encoder_beta.eval()
    all_labels = []
    all_predicted_probs = []
    all_alpha = []
    all_beta = []
    all_va = []
    all_vb = []
    all_ja = []
    all_jb = []
    with torch.no_grad():
        for alpha, beta, va_one_hot, vb_one_hot, ja_one_hot, jb_one_hot, labels, stage1 in data_loader:
            labels = labels.to(DEVICE)
            va_one_hot = va_one_hot.to(DEVICE)
            vb_one_hot = vb_one_hot.to(DEVICE)
            ja_one_hot = ja_one_hot.to(DEVICE)
            jb_one_hot = jb_one_hot.to(DEVICE)

            alpha = alpha.to(DEVICE)
            beta = beta.to(DEVICE)
            # Encode alpha and beta sequences
            _, alpha_vector, _ = encoder_alpha(alpha)
            _, beta_vector, _ = encoder_beta(beta)
            if alpha_vector.shape[0] >= 2:
                alpha_vector = alpha_vector[1].unsqueeze(0)
            if beta_vector.shape[0] >= 2:
                beta_vector = beta_vector[1].unsqueeze(0)
            # Concatenate inputs
            concatenated_a_b = torch.cat((alpha_vector, beta_vector), dim=2)
            if stage1 is not None and any(o is not None for o in stage1):
                stage1 = stage1.to(DEVICE)
                stage1 = stage1.view(1, 64, 1)
                concatenated_inputs = torch.cat((concatenated_a_b, va_one_hot.unsqueeze(0),
                                                 vb_one_hot.unsqueeze(0), ja_one_hot.unsqueeze(0),
                                                 jb_one_hot.unsqueeze(0), stage1), dim=2)
            else:
                concatenated_inputs = torch.cat((concatenated_a_b, va_one_hot.unsqueeze(0),
                                                 vb_one_hot.unsqueeze(0), ja_one_hot.unsqueeze(0),
                                                 jb_one_hot.unsqueeze(0)), dim=2)
            # Get model outputs and predictions
            outputs = model(concatenated_inputs)
            predicted_probabilities = torch.sigmoid(outputs)
            predicted = (predicted_probabilities >= 0.5).squeeze().int()
            add = (predicted == labels).sum().item()
            total += len(predicted)
            correct += add
            # Convert labels and predictions to numpy arrays
            labels = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
            if predicted_probabilities.is_cuda:
                predicted_probabilities = predicted_probabilities.cpu().numpy()
            else:
                predicted_probabilities = predicted_probabilities.numpy()

            # Append the true labels and predicted probabilities to the lists
            all_alpha.extend(alpha.cpu().numpy() if alpha.is_cuda else alpha.numpy())
            all_beta.extend(beta.cpu().numpy() if beta.is_cuda else beta.numpy())
            all_va.extend(va_one_hot.cpu().numpy() if va_one_hot.is_cuda else va_one_hot.numpy())
            all_vb.extend(vb_one_hot.cpu().numpy() if vb_one_hot.is_cuda else vb_one_hot.numpy())
            all_ja.extend(ja_one_hot.cpu().numpy() if ja_one_hot.is_cuda else ja_one_hot.numpy())
            all_jb.extend(jb_one_hot.cpu().numpy() if jb_one_hot.is_cuda else jb_one_hot.numpy())
            all_labels.extend(labels)
            all_predicted_probs.extend(predicted_probabilities.squeeze())
    # Calculate AUC
    auc = roc_auc_score(all_labels, all_predicted_probs)
    print(f'AUC: {auc}')

    # Identify top positive and bottom negative cases
    positive_indices = [i for i, label in enumerate(all_labels) if label == 1]
    negative_indices = [i for i, label in enumerate(all_labels) if label == 0]
    positive_probs = [all_predicted_probs[i] for i in positive_indices]
    negative_probs = [all_predicted_probs[i] for i in negative_indices]
    sorted_positive_indices = np.argsort(positive_probs)
    sorted_negative_indices = np.argsort(negative_probs)

    top_positive_indices = [positive_indices[i] for i in sorted_positive_indices[-len(positive_probs) // 5:]]
    bottom_negative_indices = [negative_indices[i] for i in sorted_negative_indices[:len(negative_probs) // 5]]

    # Gather the corresponding alpha and beta values based on sorted indices
    top_alpha_positives = [all_alpha[i] for i in top_positive_indices]
    top_beta_positives = [all_beta[i] for i in top_positive_indices]
    bottom_alpha_negatives = [all_alpha[i] for i in bottom_negative_indices]
    bottom_beta_negatives = [all_beta[i] for i in bottom_negative_indices]
    top_va_positives = [all_va[i] for i in bottom_negative_indices]
    top_vb_positives = [all_vb[i] for i in bottom_negative_indices]
    top_ja_positives = [all_ja[i] for i in bottom_negative_indices]
    top_jb_positives = [all_jb[i] for i in bottom_negative_indices]
    return (auc, all_labels, all_predicted_probs, all_alpha, all_beta, top_alpha_positives, top_beta_positives,
            bottom_alpha_negatives, bottom_beta_negatives)

def evaluate_model_bert(model, data_loader, tokenizer, tcrbert, ix_2_acid):
        """
        Evaluate the model on the given data loader.
        Args:
            model (nn.Module): The model to evaluate.
            data_loader (DataLoader): DataLoader for the evaluation data.
            tokenizer: A pretrained HuggingFace tokenizer (AutoTokenizer) used to
            convert CDR3 sequences into token IDs suitable for the TCR-BERT model.
            tcrbert: A pretrained HuggingFace model (AutoModel) loaded from the
            specified checkpoint, used to generate contextual embeddings for the
            input CDR3 sequences.
            ix_2_acid (dict): Dictionary mapping indices to amino acid.
        Returns:
            tuple: AUC score, all labels, all predicted probabilities, alpha sequences, beta sequences,
                   top positive alpha sequences, top positive beta sequences, bottom negative alpha sequences,
                   bottom negative beta sequences.
        """
        correct = 0
        total = 0
        model.eval()
        all_labels = []
        all_predicted_probs = []
        with torch.no_grad():
            for alpha, beta, va_one_hot, vb_one_hot, ja_one_hot, jb_one_hot, labels, stage1 in data_loader:
                labels = labels.to(DEVICE)
                va_one_hot = va_one_hot.to(DEVICE)
                vb_one_hot = vb_one_hot.to(DEVICE)
                ja_one_hot = ja_one_hot.to(DEVICE)
                jb_one_hot = jb_one_hot.to(DEVICE)
                decoded_alpha = []
                for seq_tensor in alpha:
                    seq = seq_tensor.tolist()
                    decoded = "".join(ix_2_acid[i] for i in seq if ix_2_acid[i] not in ["<EOS>", "<PAD>"])
                    decoded_alpha.append(decoded)
                decoded_beta = []
                for seq_tensor in beta:
                    seq = seq_tensor.tolist()
                    decoded = "".join(ix_2_acid[i] for i in seq if ix_2_acid[i] not in ["<EOS>", "<PAD>"])
                    decoded_beta.append(decoded)
                concatenated_inputs = Trainer.pass_models_bert(tokenizer, tcrbert, decoded_alpha, decoded_beta,
                                                               va_one_hot, vb_one_hot, ja_one_hot, jb_one_hot, DEVICE,
                                                               stage1)
                # Get model outputs and predictions
                outputs = model(concatenated_inputs)
                predicted_probabilities = torch.sigmoid(outputs)
                predicted = (predicted_probabilities >= 0.5).squeeze().int()
                add = (predicted == labels).sum().item()
                total += len(predicted)
                correct += add
                # Convert labels and predictions to numpy arrays
                labels = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
                if predicted_probabilities.is_cuda:
                    predicted_probabilities = predicted_probabilities.cpu().numpy()
                else:
                    predicted_probabilities = predicted_probabilities.numpy()
                # Append the true labels and predicted probabilities to the lists
                all_labels.extend(labels)
                all_predicted_probs.extend(predicted_probabilities.squeeze())
        # Calculate AUC
        auc = roc_auc_score(all_labels, all_predicted_probs)
        print(f'AUC: {auc}')

        return auc, all_labels, all_predicted_probs


def plot_auc(all_labels, all_predicted_probs, all_labels2=None, all_predicted_probs2=None, ax=None, text=None,
             first_text=None, second_text=None):
    """
    Plot the ROC curve and compute the AUC score.
    Args:
        all_labels (list): True labels for the first dataset.
        all_predicted_probs (list): Predicted probabilities for the first dataset.
        all_labels2 (list, optional): True labels for the second dataset. Default is None.
        all_predicted_probs2 (list, optional): Predicted probabilities for the second dataset. Default is None.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Default is None.
        text (str, optional): Text to include in the saved plot filename. Default is None.
        first_text
        second_text
    """
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_predicted_probs)
    auc = roc_auc_score(all_labels, all_predicted_probs)

    # Plot ROC curve
    if ax is None:
        plt.figure()
        ax = plt.gca()

    if all_labels2 is not None and all_predicted_probs2 is not None:
        fpr2, tpr2, _ = roc_curve(all_labels2, all_predicted_probs2)
        auc2 = roc_auc_score(all_labels2, all_predicted_probs2)
        if second_text is None:
            second_text = "pMHC-I binding"
        ax.plot(fpr2, tpr2, color='green', lw=2, label=f'{second_text} (AUC = {auc2:.2f})')
    if first_text is None:
        first_text = "All T cells"
    ax.plot(fpr, tpr, color='orange', lw=2, label=f'{first_text} (AUC = {auc:.2f})')

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontproperties=font, fontsize=font_size + 5)
    ax.set_ylabel('True Positive Rate', fontproperties=font, fontsize=font_size + 5)
    ax.legend(loc="lower right", prop=font)
    ax.grid()

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    # Round the tick values to one decimal place
    rounded_xticks = np.round(xticks, 1)
    rounded_yticks = np.round(yticks, 1)
    # Set the rounded tick labels with font properties
    ax.set_xticklabels(rounded_xticks, fontproperties=font, fontsize=font_size + 5)
    ax.set_yticklabels(rounded_yticks, fontproperties=font, fontsize=font_size + 5)

    if ax is None:
        if text is not None:
            plt.savefig(os.path.join(base_folder, "Roc curve " + text + ".png"))
        else:
            Trainer.save_plot_with_incremental_filename(base_folder, "Roc curve.png")


def plot_precision_recall(all_labels, all_predicted_probs, all_labels2=None, all_predicted_probs2=None, ax=None,
                          text=None, first_text=None, second_text=None):
    """
    Plot the Precision-Recall curve and compute the Average Precision (AP) score.
    Args:
        all_labels (list): True labels for the first dataset.
        all_predicted_probs (list): Predicted probabilities for the first dataset.
        all_labels2 (list, optional): True labels for the second dataset. Default is None.
        all_predicted_probs2 (list, optional): Predicted probabilities for the second dataset. Default is None.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. Default is None.
        text (str, optional): Text to include in the saved plot filename. Default is None.
        first_text
        second_text
    """
    # Compute Precision-Recall curve and Average Precision (AP) for the first dataset
    precision, recall, _ = precision_recall_curve(all_labels, all_predicted_probs)
    avg_precision = average_precision_score(all_labels, all_predicted_probs)
    precision = np.insert(precision, 0, 0.0)
    recall = np.insert(recall, 0, 1.0)

    # Plot Precision-Recall curve for the first dataset
    if ax is None:
        plt.figure()
        ax = plt.gca()

    if all_labels2 is not None and all_predicted_probs2 is not None:
        precision2, recall2, _ = precision_recall_curve(all_labels2, all_predicted_probs2)
        avg_precision2 = average_precision_score(all_labels2, all_predicted_probs2)
        precision2 = np.insert(precision2, 0, 0.0)
        recall2 = np.insert(recall2, 0, 1.0)
        if second_text is None:
            second_text = "pMHC-I binding"
        ax.plot(recall2, precision2, color='green', lw=2, label=f'{second_text} (AP = {avg_precision2:.2f})')
    if first_text is None:
        first_text = "All T cells"
    ax.plot(recall, precision, color='orange', lw=2, label=f'{first_text} (AP = {avg_precision:.2f})')

    ax.plot([1, 1], [0, 0], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('Recall', fontproperties=font)
    ax.set_ylabel('Precision', fontproperties=font)
    ax.legend(loc="lower left", prop=font)
    ax.grid()

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    # Round the tick values to one decimal place
    rounded_xticks = np.round(xticks, 1)
    rounded_yticks = np.round(yticks, 1)
    # Set the rounded tick labels with font properties
    ax.set_xticklabels(rounded_xticks, fontproperties=font, fontsize=font_size)
    ax.set_yticklabels(rounded_yticks, fontproperties=font, fontsize=font_size)

    if ax is None:
        # Save the plot
        if text is not None:
            plt.savefig(os.path.join(base_folder, "Precision-Recall curve " + text + ".png"))
        else:
            plt.savefig(os.path.join(base_folder, "Precision-Recall curve.png"))


def objective(trial, file_path):
    """
    Objective function for Optuna hyperparameter optimization.
    Args:
        trial (optuna.trial.Trial): A trial object for hyperparameter suggestions.
        file_path (str): Path to the data file.
    Returns:
        float: Mean AUC score across K-folds.
    """
    # Define the search space for hyperparameters
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])
    weight_decay_cl = trial.suggest_float('weight_decay_cl', 1e-5, 1e-1, log=True)
    weight_decay_encoder = trial.suggest_float('weight_decay_encoder', 1e-5, 1e-1, log=True)
    latent_size = trial.suggest_categorical('latent_size', [128, 256, 512])
    num_layers = trial.suggest_categorical('num_layers', [1, 2])
    dropout_prob_en = trial.suggest_float('dropout_prob_en', 0.0, 0.5)
    layer_norm = trial.suggest_categorical('layer_norm', [True, False])
    dropout_prob_cl = trial.suggest_float('dropout_prob_cl', 0.0, 0.6)
    norm_cl = trial.suggest_categorical('norm_cl', [True, False])
    losses_weight = trial.suggest_int('losses_weight', 10, 80)
    embed_size = trial.suggest_categorical('embed_size', [64, 128, 256])
    lr_encoder = trial.suggest_float("lr_encoder", 1e-5, 1e-3, log=True)
    lr_cl = trial.suggest_float("lr_cl", 1e-4, 1e-2, log=True)

    encoder_type = "LSTM"
    if encoder_type == "LSTM" or encoder_type == "BiLSTM":
        nhead = None
        dim_feedforward = None
    else:
        nhead = trial.suggest_categorical('nhead', [4, 8, 2])
        dim_feedforward = trial.suggest_categorical('dim_feedforward', [2048, 1024])
        # Ensure that embed_size is divisible by nhead
        if embed_size % nhead != 0:
            raise ValueError("embed_size must be divisible by nhead")

    hyperparameter = (embed_size, hidden_size, num_layers, latent_size, weight_decay_encoder, dropout_prob_en, layer_norm,
                      nhead, dim_feedforward, weight_decay_cl, dropout_prob_cl, norm_cl, losses_weight, lr_encoder, lr_cl)

    batch_size = 64
    model_type = "LSTM"
    # K-fold cross-validation loop
    k = 2
    fold_data = process_data_with_k_fold(file_path, batch_size, k)
    fold_aucs = []
    vocab_size = len(fold_data[0][0].vocab)

    for fold in range(k):
        print(f"Training on fold {fold + 1}/{k}")
        (train_dataset, train_data_loader, train_data, test_dataset, test_data_loader, test_data,
         len_one_hot) = fold_data[fold]
        # Train models with current hyperparameters
        model, (alpha_encoder, alpha_decoder), (beta_encoder, beta_decoder) = train_models(
            vocab_size, hyperparameter, batch_size, train_dataset.acid_2_ix, train_data_loader,
            len_one_hot, model_type, test_data_loader
        )
        # Evaluate model and calculate AUC
        auc_test, _, _, _, _, _, _, _, _ = evaluate_model(
            model, (alpha_encoder, beta_encoder), test_data_loader
        )
        auc_train, _, _, _, _, _, _, _, _ = evaluate_model(
            model, (alpha_encoder, beta_encoder), train_data_loader
        )
        print("auc_train", auc_train)
        fold_aucs.append(auc_test)
        with open(os.path.join(base_folder, "auc_results.txt"), 'a') as f:
            f.write(f"Train AUC: {auc_train}, Test AUC: {auc_test}\n")

    return np.mean(fold_aucs)


def save_best_params(study, trial, file_path='best_hyperparameters_vdjdb.json'):
    """
    Save the best hyperparameters to a file if the current trial's score is better.
    Args:
        study (optuna.study.Study): The study object.
        trial (optuna.trial.Trial): The trial object.
        file_path (str, optional): Path to the file where best hyperparameters will be saved.
    """
    current_best_value = study.best_value
    print("in_saving")
    try:
        # Check the current saved best score
        with open(file_path, 'r') as f:
            saved_data = json.load(f)
            saved_best_value = saved_data.get("AUC", float('-inf'))
    except (FileNotFoundError, json.JSONDecodeError):
        # If the file doesn't exist or is invalid, treat as no prior best
        saved_best_value = float('-inf')

    # Save only if the current trial's score is better
    if current_best_value > saved_best_value:
        best_params = study.best_params
        best_params["AUC"] = current_best_value  # Add the best score
        with open(file_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"Updated best hyperparameters saved to {file_path}")


def run_optuna(input_file, hyperparameters_file_path, initial_hyperparameters=None):
    """
    Run Optuna hyperparameter optimization.
    Args:
        input_file (str): Path to the input data file.
        hyperparameters_file_path (str): Path to the file where best hyperparameters will be saved.
        initial_hyperparameters: start search from
    """
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))


    # Try to load initial seed hyperparameters from file
    if initial_hyperparameters is not None:
        try:
            with open(initial_hyperparameters, 'r') as f:
                loaded = json.load(f)

            # Map keys from file to Optuna trial parameter names
            initial_params = {
                "hidden_size": loaded["hidden_size"],
                "weight_decay_cl": loaded["weight_decay_cl"],
                "weight_decay_encoder": loaded["weight_decay_encoder"],
                "latent_size": loaded["latent_size"],
                "num_layers": loaded["num_layers"],
                "dropout_prob_en": loaded["dropout_prob_en"],
                "layer_norm_en": loaded["layer_norm"],
                "dropout_prob_cl": loaded["dropout_prob_cl"],
                "batch_norm_cl": loaded["norm_cl"],
                "losses_weight": loaded["losses_weight"],
                "embed_size": loaded["embed_size"],
                "lr_encoder": loaded["lr_encoder"],
                "lr_cl": loaded["lr_cl"]
            }
            #     # Enqueue the initial trial
            study.enqueue_trial(initial_params)
            print("Initial trial enqueued with parameters:", initial_params)
        except Exception as e:
            print("No valid initial hyperparameters found:", e)

    def wrapped_objective(trial):
        return objective(trial, input_file)

    study.optimize(wrapped_objective, n_trials=200,
                   callbacks=[lambda study, trial:save_best_params(study, trial, hyperparameters_file_path)])

    # Print best hyperparameters and their result
    print("Best trial:")
    trial = study.best_trial
    print(f"  AUC: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    # Save best hyperparameters to a file
    with open(hyperparameters_file_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print("Best hyperparameters saved")


def find_best_model(file_path, best_params_file, model_type):
    """
    Find the best model using the best hyperparameters.
    Args:
        file_path (str): Path to the data file.
        best_params_file (str): Path to the file containing the best hyperparameters.
        model_type (str): Type of model to train.
    Returns:
        tuple: Best model, alpha encoder, beta encoder, and test data loader.
    """
    print("Using folder:", base_folder)
    # Load the best hyperparameters from the JSON file
    with open(best_params_file, 'r') as f:
        best_params = json.load(f)

    # Extract the parameters
    embed_size = best_params.get('embed_size', None)
    hidden_size = best_params.get('hidden_size', None)
    num_layers = best_params.get('num_layers', None)
    latent_size = best_params.get('latent_size', None)
    weight_decay_cl = best_params.get('weight_decay_cl', None)
    weight_decay_encoder = best_params.get('weight_decay_encoder', None)
    dropout_prob_en = best_params['dropout_prob_en']
    dropout_prob_cl = best_params['dropout_prob_cl']
    layer_norm = best_params['layer_norm']
    norm_cl = best_params['norm_cl']
    losses_weight = best_params['losses_weight']
    lr_encoder = best_params['lr_encoder']
    lr_cl = best_params['lr_cl']

    # Extract parameters specific to non-LSTM models if available
    nhead = best_params.get('nhead', None)
    dim_feedforward = best_params.get('dim_feedforward', None)

    # Prepare the hyperparameter tuple
    hyperparameter = (embed_size, hidden_size, num_layers, latent_size, weight_decay_encoder, dropout_prob_en, layer_norm,
                      nhead, dim_feedforward, weight_decay_cl, dropout_prob_cl, norm_cl, losses_weight, lr_encoder, lr_cl)

    # Other fixed hyperparameters
    batch_size = 64
    k = 5

    # Process data
    fold_data = process_data_with_k_fold(file_path, batch_size, k)
    fold_aucs_test = []
    fold_aucs_train = []
    vocab_size = len(fold_data[0][0].vocab)

    best_auc = 0
    best_model_state_dict = None
    best_alpha_encoder = None
    best_beta_encoder = None
    best_model_path = os.path.join(base_folder, 'best_model.pth')
    best_alpha_encoder_path = os.path.join(base_folder,
                                           'best_alpha_encoder.pth')
    best_beta_encoder_path = os.path.join(base_folder,
                                          'best_beta_encoder.pth')

    for fold in range(k):
        print(f"Training on fold {fold + 1}/{k}")
        (train_dataset, train_data_loader, train_data, test_dataset, test_data_loader, test_data,
         len_one_hot) = fold_data[fold]
        if model_type == "BERT":
            ix_2_acid = train_dataset.ix_2_acid
        else:
            ix_2_acid = None
        # Train models with current hyperparameters
        model, (alpha_encoder, alpha_decoder), (beta_encoder, beta_decoder) = train_models(
            vocab_size, hyperparameter, batch_size, train_dataset.acid_2_ix, train_data_loader,
            len_one_hot, model_type, test_data_loader, ix_2_acid
        )

        # Evaluate model and calculate AUC
        if model_type == "BERT":
            auc_train, all_labels_train, all_predicted_probs_train = evaluate_model_bert(
                model, train_data_loader, tokenizer, tcrbert, ix_2_acid
            )
            plot_auc(all_labels_train, all_predicted_probs_train)
            auc_test, all_labels_test, all_predicted_probs_test = evaluate_model_bert(
                model, test_data_loader, tokenizer, tcrbert, ix_2_acid
            )
        else:
            auc_train, all_labels_train, all_predicted_probs_train, _, _, _, _, _, _ = evaluate_model(
                model, (alpha_encoder, beta_encoder), train_data_loader
            )
            plot_auc(all_labels_train, all_predicted_probs_train)
            auc_test, all_labels_test, all_predicted_probs_test, _, _, _, _, _, _ = evaluate_model(
                model, (alpha_encoder, beta_encoder), test_data_loader
            )
        fold_aucs_test.append(auc_test)
        fold_aucs_train.append(auc_train)

        # Save the model if the test AUC is the best so far
        if auc_test > best_auc:
            best_auc = auc_test
            best_model_state_dict = model.state_dict()
            if model_type != "BERT":
                best_alpha_encoder = alpha_encoder.state_dict()
                best_beta_encoder = beta_encoder.state_dict()
            print(f'New best AUC: {best_auc} found on fold {fold + 1}')

    # Save only the best model after all folds have been processed
    if best_model_state_dict is not None:
        torch.save(best_model_state_dict, best_model_path)
        if model_type != "BERT":
            torch.save(best_alpha_encoder, best_alpha_encoder_path)
            torch.save(best_beta_encoder, best_beta_encoder_path)
            print(f'Best alpha encoder saved to {best_alpha_encoder_path}')
            print(f'Best beta encoder saved to {best_beta_encoder_path}')
        print(f'Best model saved to {best_model_path} with AUC: {best_auc}')
    auc_file_path = os.path.join(base_folder, "auc_values.txt")
    with open(auc_file_path, "w") as f:
        f.write("Fold\tTrain_AUC\tVal_AUC\n")
        for i, (train_auc, val_auc) in enumerate(zip(fold_aucs_train, fold_aucs_test)):
            f.write(f"{i + 1}\t{train_auc:.6f}\t{val_auc:.6f}\n")

    print(f"AUC values saved to {auc_file_path}")

    # Return the mean AUC across the folds
    return model, alpha_encoder, beta_encoder, test_data_loader


def load_models(model_save_paths, len_one_hot, vocab_size, best_params_file, model_type):
    """
    Load the best models and their parameters.
    Args:
        model_save_paths (dict): Dictionary containing paths to the saved models.
        len_one_hot (int): Length of the one-hot encoded vectors.
        vocab_size (int): Size of the vocabulary.
        best_params_file (str): Path to the file containing the best hyperparameters.
        model_type (str): Type of model to train.
    Returns:
        tuple: Loaded main model, alpha encoder, and beta encoder.
    """
    # Load the best hyperparameters from the JSON file
    with open(best_params_file, 'r') as f:
        best_params = json.load(f)

    # Extract the parameters
    embed_size = best_params['embed_size']
    hidden_size = best_params['hidden_size']
    num_layers = best_params['num_layers']
    latent_size = best_params['latent_size']
    dropout_prob_en = best_params['dropout_prob_en']
    layer_norm = best_params['layer_norm']
    norm_cl = best_params['norm_cl']
    weight_decay_cl = best_params['weight_decay_cl']
    weight_decay_encoder = best_params['weight_decay_encoder']
    dropout_prob_cl = best_params['dropout_prob_cl']
    losses_weight = best_params['losses_weight']
    lr_encoder = best_params['lr_encoder']
    lr_cl = best_params['lr_cl']

    if model_type == "BERT":
        main_model = Models.FFNN(768 * 2 + len_one_hot, dropout_prob_cl, norm_cl).to(DEVICE)
        main_model.load_state_dict(torch.load(model_save_paths['model'], map_location=DEVICE, weights_only=False))
        main_model.eval()  # Set the model to evaluation mode
        return main_model

    # Load the main model
    main_model = Models.FFNN(latent_size * 2 + len_one_hot, dropout_prob_cl, norm_cl).to(DEVICE)

    main_model.load_state_dict(torch.load(model_save_paths['model'], map_location=DEVICE, weights_only=False))
    main_model.eval()  # Set the model to evaluation mode

    if model_type == "ALSTM":
        ALSTM = True
        BiLSTM = False
    elif model_type == "BiLSTM":
        ALSTM = False
        BiLSTM = True
    elif model_type == "ABiLSTM":
        ALSTM = True
        BiLSTM = True
    else:
        ALSTM = False
        BiLSTM = False

    # Load the alpha encoder
    alpha_encoder = Models.EncoderLstm(vocab_size, embed_size, hidden_size, latent_size, dropout_prob_en, layer_norm,
                                       num_layers, BiLSTM).to(DEVICE)
    alpha_encoder.load_state_dict(torch.load(model_save_paths['alpha_encoder'], map_location=DEVICE,
                                             weights_only=False))
    alpha_encoder.eval()

    # Load the beta encoder
    beta_encoder = Models.EncoderLstm(vocab_size, embed_size, hidden_size, latent_size, dropout_prob_en, layer_norm,
                                      num_layers, BiLSTM).to(DEVICE)
    beta_encoder.load_state_dict(torch.load(model_save_paths['beta_encoder'], map_location=DEVICE,
                                            weights_only=False))
    beta_encoder.eval()

    return main_model, alpha_encoder, beta_encoder


def load_data(file_path):
    """
    Load data and prepare the dataset and DataLoader.
    Args:
        file_path (str): Path to the data file.
    Returns:
        tuple: Full dataset, DataLoader, pairs, and length of one-hot encoded vectors.
    """
    pairs = Loader.read_data(file_path)
    va_counts, vb_counts, ja_counts, jb_counts = read_dictionaries_from_file('filtered_counters.json')
    vj_data = (va_counts, vb_counts, ja_counts, jb_counts)
    len_one_hot = len(va_counts) + len(vb_counts) + len(ja_counts) + len(jb_counts)
    batch_size = 64
    full_dataset = Loader.ChainClassificationDataset(pairs, vj_data)
    full_data_loader = DataLoader(full_dataset, batch_size=batch_size,
                                  shuffle=False, drop_last=False, collate_fn=Loader.collate_fn)
    return full_dataset, full_data_loader, pairs, len_one_hot


def load_and_evaluate_model(file_path, param_file, model_type, type_model="pMHC", model_save_paths=None):
    """
    Load and evaluate the model on the given dataset.
    Args:
        file_path (str): Path to the data file.
        param_file (str): Path to the file containing the best hyperparameters.
        model_type (str): Type of model to train.
        type_model (str, optional): Type of the model. Default is 'pMHC'.
        model_save_paths (str, optional): Path to saved models.
    Returns:
        tuple: All labels and predicted probabilities for the test set.
    """
    if model_save_paths is None:
        if type_model == "pMHC":
            model_save_paths = {
                'model': "models/combined_5fold/best_model.pth",
                'alpha_encoder': "models/combined_5fold/best_alpha_encoder.pth",
                'beta_encoder': "models/combined_5fold/best_beta_encoder.pth"
            }
        else:
            model_save_paths = {
                'model': "models/non_paired/best_model.pth",
                'alpha_encoder': "models/non_paired/best_alpha_encoder.pth",
                'beta_encoder': "models/non_paired/best_beta_encoder.pth"
            }
    # Load your dataset
    full_dataset, full_data_loader, full_data, len_one_hot = load_data(file_path)
    vocab_size = len(full_dataset.vocab)
    # Load the best models
    print(model_save_paths, file_path)
    if model_type == "BERT":
        model = load_models(model_save_paths, len_one_hot, vocab_size, param_file, model_type)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_safetensors=True)
        tcrbert = AutoModel.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            trust_remote_code=True,
            use_safetensors=True
        )

        tcrbert = tcrbert.to(DEVICE)
        (auc_test, all_labels_test, all_predicted_probs_test) = evaluate_model_bert(
            model, full_data_loader, tokenizer, tcrbert, full_dataset.ix_2_acid
        )
    else:
        model, alpha_encoder, beta_encoder = load_models(model_save_paths, len_one_hot, vocab_size, param_file,
                                                         model_type)
        # Evaluate the model on the test set
        (auc_test, all_labels_test, all_predicted_probs_test, all_alpha_te, all_beta_te, top_alpha_positives_te,
         top_beta_positives_te, bottom_alpha_negatives_te, bottom_beta_negatives_te) = evaluate_model(
            model, (alpha_encoder, beta_encoder), full_data_loader
        )
    print(f'Test AUC: {auc_test}')
    return all_labels_test, all_predicted_probs_test


def run_on_all_save_models(param_file, dataset_file, paths_for_best_models):
    """
    Train models on the entire dataset and save the best models.
    Args:
        param_file (str): Path to the file containing the best hyperparameters.
        dataset_file (str): Path to the dataset file.
        paths_for_best_models (dict): Dictionary containing paths to save the best models.
    """
    # Load the best hyperparameters from the JSON file
    with open(param_file, 'r') as f:
        best_params = json.load(f)

    # Extract the parameters
    embed_size = best_params.get('embed_size', None)
    hidden_size = best_params.get('hidden_size', None)
    num_layers = best_params.get('num_layers', None)
    latent_size = best_params.get('latent_size', None)
    weight_decay_cl = best_params.get('weight_decay_cl', None)
    weight_decay_encoder = best_params.get('weight_decay_encoder', None)
    dropout_prob_en = best_params['dropout_prob_en']
    dropout_prob_cl = best_params['dropout_prob_cl']
    layer_norm = best_params['layer_norm']
    norm_cl = best_params['norm_cl']
    losses_weight = best_params['losses_weight']
    lr_encoder = best_params['lr_encoder']
    lr_cl = best_params['lr_cl']

    # Extract parameters specific to non-LSTM models if available
    nhead = best_params.get('nhead', None)
    dim_feedforward = best_params.get('dim_feedforward', None)

    # Prepare the hyperparameter tuple
    hyperparameter = (embed_size, hidden_size, num_layers, latent_size, weight_decay_encoder, dropout_prob_en, layer_norm,
                      nhead, dim_feedforward, weight_decay_cl, dropout_prob_cl, norm_cl, losses_weight, lr_encoder, lr_cl)

    batch_size = 64
    pairs = Loader.read_data(dataset_file)
    va_counts, vb_counts, ja_counts, jb_counts = read_dictionaries_from_file('filtered_counters.json')
    vj_data = (va_counts, vb_counts, ja_counts, jb_counts)
    # Calculate the total length of all dictionaries combined
    len_one_hot = len(va_counts) + len(vb_counts) + len(ja_counts) + len(jb_counts)
    train_data = pairs

    # Create the dataset and DataLoader for the training set
    train_dataset = Loader.ChainClassificationDataset(train_data, vj_data)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, drop_last=True, collate_fn=Loader.collate_fn)
    vocab_size = len(train_dataset.vocab)

    best_model_path = os.path.join(base_folder, paths_for_best_models['model'])
    best_alpha_encoder_path = os.path.join(base_folder, paths_for_best_models['alpha_encoder'])
    best_beta_encoder_path = os.path.join(base_folder, paths_for_best_models['beta_encoder'])

    # Train models with current hyperparameters
    model, (alpha_encoder, alpha_decoder), (beta_encoder, beta_decoder) = train_models(
        vocab_size, hyperparameter, batch_size, train_dataset.acid_2_ix, train_data_loader,
        len_one_hot, "LSTM", ix_2_acid=train_dataset.ix_2_acid
    )

    best_model_state_dict = model.state_dict()
    best_alpha_encoder = alpha_encoder.state_dict()
    best_beta_encoder = beta_encoder.state_dict()

    # Save only the best model after all folds have been processed
    if best_model_state_dict is not None:
        torch.save(best_model_state_dict, best_model_path)
        torch.save(best_alpha_encoder, best_alpha_encoder_path)
        torch.save(best_beta_encoder, best_beta_encoder_path)
        print(f'Best model saved to {best_model_path}')
        print(f'Best alpha encoder saved to {best_alpha_encoder_path}')
        print(f'Best beta encoder saved to {best_beta_encoder_path}')


def filter_test(train_file, test_file, new_path):
    # Read both files
    df_test = pd.read_csv(test_file)
    df_train = pd.read_csv(train_file)

    # Remove exact row matches (same values in all columns)
    df_filtered = df_test[~df_test.apply(tuple, axis=1).isin(df_train.apply(tuple, axis=1))]

    # Save the result
    df_filtered.to_csv(new_path, index=False)


def predict(input_pair, model_of="pMHC"):
    """
    Predict the probability for a given input pair using the specified model.
    Args:
        input_pair (list): List containing the input pair data.
        model_of (str, optional): Type of the model. Default is "ireceptor".
    Returns:
        Tensor: Predicted probability.
    """
    if model_of == "pMHC":
        model_save_paths = {'model': "models/pMHC-final_model/best_model.pth", 'alpha_encoder': "models/pMHC-final_model/best_alpha_encoder.pth",
                            'beta_encoder': "models/pMHC-final_model/best_beta_encoder.pth"}
        best_param = "hyperparameters.json"

    else:
        model_save_paths = {'model': "models/non-paired/best_model.pth", 'alpha_encoder': "models/non-paired/best_alpha_encoder.pth",
                            'beta_encoder': "models/non-paired/best_beta_encoder.pth"}
        best_param = "hyperparameters.json"
    va_counts, vb_counts, ja_counts, jb_counts = read_dictionaries_from_file('filtered_counters.json')
    vj_data = (va_counts, vb_counts, ja_counts, jb_counts)
    len_one_hot = len(va_counts) + len(vb_counts) + len(ja_counts) + len(jb_counts)
    # Format the input pair
    va = v_j.v_j_format(input_pair[0][1], 1, "TRAV")
    vb = v_j.v_j_format(input_pair[1][1], 1, "TRBV")
    ja = v_j.v_j_format(input_pair[0][2], 1, "TRAJ")
    jb = v_j.v_j_format(input_pair[1][2], 2, "TRBJ")
    if len(input_pair) == 3:
        input_pair_ordered = [(input_pair[0][0], input_pair[1][0]), (va, vb, ja, jb), -1, input_pair[2]]
    else:
        input_pair_ordered = [(input_pair[0][0], input_pair[1][0]), (va, vb, ja, jb)]
    input_pair_ordered = [tuple(input_pair_ordered)]
    # Create the dataset and DataLoader
    dataset = Loader.ChainClassificationDataset(input_pair_ordered, vj_data)
    loader = DataLoader(dataset, batch_size=1, collate_fn=Loader.collate_fn)
    vocab_size = len(dataset.vocab)
    # Load the models
    model, alpha_encoder, beta_encoder = load_models(model_save_paths, len_one_hot, vocab_size, best_param, "LSTM")
    alpha_encoder.eval()
    beta_encoder.eval()
    model.eval()
    # Use torch.no_grad() to disable gradient computation
    with torch.no_grad():
        for alpha, beta, va_one_hot, vb_one_hot, ja_one_hot, jb_one_hot, _, base_score in loader:
            alpha = alpha.to(DEVICE)
            beta = beta.to(DEVICE)
            va_one_hot = va_one_hot.to(DEVICE)
            vb_one_hot = vb_one_hot.to(DEVICE)
            ja_one_hot = ja_one_hot.to(DEVICE)
            jb_one_hot = jb_one_hot.to(DEVICE)
            _, alpha_vector, _ = alpha_encoder(alpha)
            _, beta_vector, _ = beta_encoder(beta)
            if alpha_vector.shape[0] >= 2:
                alpha_vector = alpha_vector[1].unsqueeze(0)
            if beta_vector.shape[0] >= 2:
                beta_vector = beta_vector[1].unsqueeze(0)
            concatenated_a_b = torch.cat((alpha_vector, beta_vector), dim=2)
            if base_score is not None and any(o is not None for o in base_score):
                base_score = base_score.to(DEVICE)
                base_score = base_score.view(1, 1, 1)
                concatenated_inputs = torch.cat((concatenated_a_b, va_one_hot.unsqueeze(0),
                                                 vb_one_hot.unsqueeze(0), ja_one_hot.unsqueeze(0),
                                                 jb_one_hot.unsqueeze(0), base_score), dim=2)
            else:
                concatenated_inputs = torch.cat((concatenated_a_b, va_one_hot.unsqueeze(0),
                                                 vb_one_hot.unsqueeze(0), ja_one_hot.unsqueeze(0),
                                                 jb_one_hot.unsqueeze(0)), dim=2)
            output = model(concatenated_inputs)
            predicted_probability = torch.sigmoid(output)
    return predicted_probability


def validate_tr_prefix(value, prefix):
    """
    Validate that a value starts with a specific prefix.
    Args:
        value (str): The value to validate.
        prefix (str): The required prefix.
    Returns:
        str: The validated value.
    Raises:
        argparse.ArgumentTypeError: If the value does not start with the prefix.
    """
    if not value.startswith(prefix):
        raise argparse.ArgumentTypeError(f"{value} must start with {prefix}")
    return value


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process 7 input values")

    # Add arguments for the 7 parameters
    parser.add_argument('--tcra', type=str, required=True, help="Tcr alpha sequence")
    parser.add_argument('--va', type=lambda v: validate_tr_prefix(v, "TRAV"), required=True, help="V alpha")
    parser.add_argument('--ja', type=lambda v: validate_tr_prefix(v, "TRAJ"), required=True, help="J alpha")
    parser.add_argument('--tcrb', type=str, required=True, help="Tcr beta sequence")
    parser.add_argument('--vb', type=lambda v: validate_tr_prefix(v, "TRBV"), required=True, help="V beta")
    parser.add_argument('--jb', type=lambda v: validate_tr_prefix(v, "TRBJ"), required=True, help="J beta")
    parser.add_argument('--data_type', type=str, choices=['All T cells', 'pMHC'],
                        help="Optional: 'All T cells' or 'pMHC'")

    return parser.parse_args()


def set_seed(seed: int):
    print(f"Setting seed to: {seed}...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def filter_dummy_dup(original_file, out_file):
    df_original = pd.read_csv(original_file)
    df_unique = df_original.drop_duplicates(subset=["tcra", "tcrb"])
    df_unique.to_csv(out_file, index=False)


def combine_and_shuffle_csv(file1, file2, output_file, seed=42):
    """
    Combine two CSV files with identical columns and shuffle the rows.
    Args:
        file1 (str): Path to first CSV file.
        file2 (str): Path to second CSV file.
        output_file (str): Path to save the combined shuffled CSV.
        seed (int): Random seed for reproducible shuffling.
    """
    # Read both files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    # Combine them
    df = pd.concat([df1, df2], ignore_index=True)
    # Shuffle rows
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # Save result
    df.to_csv(output_file, index=False)
    print(f"Combined and shuffled file saved to: {output_file}")


if __name__ == "__main__":
    SEED = 42
    set_seed(SEED)
    # Parse command-line arguments
    args = parse_arguments()
    # Prepare the input pair from the parsed arguments
    input_pair = [[args.tcra, args.va, args.ja], [args.tcrb, args.vb, args.jb]]
    # Determine the model type based on the data type
    if args.data_type:
        output = predict(input_pair, args.data_type).item()
    else:
        output = predict(input_pair).item()
    print(f'The probability for given Alpha and Beta chains to pair is {output}.')
