import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef)
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

SEED = 42
MAX_LENGTH_ESM = 256
BATCH_SIZE_EMBEDDING_ESM = 16
USE_ATTENTION_POOLING_ESM = False

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

try:
    df1 = pd.read_excel('pnas.1616408114.sd01.xlsx')
    df2 = pd.read_excel('pnas.1616408114.sd02.xlsx')
    df3 = pd.read_excel('pnas.1616408114.sd03.xlsx')
    merged_df = df1.merge(df2, on='Name', how='outer').merge(df3, on='Name', how='outer')
    data_df_full = merged_df[['VH', 'VL', 'Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average']].copy()
    data_df_full = data_df_full.rename(columns={'Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average': 'agg'})
    data_df_full.dropna(subset=['VH', 'VL', 'agg'], inplace=True)
    print(f"Original data loaded. Shape: {data_df_full.shape}")
    if data_df_full.empty:
        raise ValueError("DataFrame is empty after loading. Check paths or column names.")
except FileNotFoundError:
    print("Error: One or more PNAS data files not found.")
    exit()
except KeyError as e:
    print(f"Error: Missing expected column in PNAS data: {e}")
    exit()

threshold = 10
data_df_full['safety_label'] = data_df_full['agg'].apply(lambda x: 1 if x > threshold else 0)

dataset_for_cv = data_df_full.copy()
print(f"Full dataset size for CV: {len(dataset_for_cv)}")

model_configurations = {
    "facebook/esm2_t6_8M_UR50D": (320, "simple"),
    "facebook/esm2_t12_35M_UR50D": (480, "medium"),
    "facebook/esm2_t30_150M_UR50D": (640, "medium"),
    "facebook/esm2_t33_650M_UR50D": (1280, "deep"),
}

esm_models_to_iterate = [
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    "facebook/esm2_t30_150M_UR50D",
    "facebook/esm2_t33_650M_UR50D"
]

tokenizer_esm_global = None

class AntibodyEmbeddingDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=MAX_LENGTH_ESM):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.df['VH'] = self.df['VH'].astype(str)
        self.df['VL'] = self.df['VL'].astype(str)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        heavy = str(row["VH"])
        light = str(row["VL"])
        label = row["safety_label"]
        heavy_inputs = self.tokenizer(heavy, truncation=True, max_length=self.max_length, return_tensors="pt")
        light_inputs = self.tokenizer(light, truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            "heavy_input_ids": heavy_inputs["input_ids"].squeeze(0),
            "heavy_attention_mask": heavy_inputs["attention_mask"].squeeze(0),
            "light_input_ids": light_inputs["input_ids"].squeeze(0),
            "light_attention_mask": light_inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

def collate_fn_esm(batch):
    if tokenizer_esm_global is None:
        raise ValueError("Global ESM tokenizer for collate_fn is not set.")
    heavy_ids = [item["heavy_input_ids"] for item in batch]
    heavy_masks = [item["heavy_attention_mask"] for item in batch]
    light_ids = [item["light_input_ids"] for item in batch]
    light_masks = [item["light_attention_mask"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    heavy_ids_padded = pad_sequence(heavy_ids, batch_first=True, padding_value=tokenizer_esm_global.pad_token_id)
    heavy_masks_padded = pad_sequence(heavy_masks, batch_first=True, padding_value=0)
    light_ids_padded = pad_sequence(light_ids, batch_first=True, padding_value=tokenizer_esm_global.pad_token_id)
    light_masks_padded = pad_sequence(light_masks, batch_first=True, padding_value=0)
    return {
        "heavy_input_ids": heavy_ids_padded,
        "heavy_attention_mask": heavy_masks_padded,
        "light_input_ids": light_ids_padded,
        "light_attention_mask": light_masks_padded,
        "label": labels
    }

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    def forward(self, token_embeddings, attention_mask):
        att_scores = self.attention(token_embeddings).squeeze(-1)
        att_scores = att_scores.masked_fill(~(attention_mask.bool()), float("-inf"))
        att_weights = torch.softmax(att_scores, dim=-1).unsqueeze(-1)
        pooled = torch.sum(token_embeddings * att_weights, dim=1)
        return pooled

class ESMEmbedder(nn.Module):
    def __init__(self, esm_model_instance, hidden_dim, use_attention_pool=USE_ATTENTION_POOLING_ESM):
        super().__init__()
        self.esm_model = esm_model_instance
        for param in self.esm_model.parameters():
            param.requires_grad = False
        self.use_attention_pool = use_attention_pool
        if use_attention_pool:
            self.pooler = AttentionPooling(hidden_dim)
        else:
            self.pooler = None
    def _mean_pool(self, hidden_state, attention_mask):
        mask_f = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden_state * mask_f).sum(dim=1)
        len_hidden = mask_f.sum(dim=1).clamp(min=1e-9)
        return sum_hidden / len_hidden
    def forward(self, heavy_ids, heavy_mask, light_ids, light_mask):
        heavy_out = self.esm_model(input_ids=heavy_ids, attention_mask=heavy_mask)
        light_out = self.esm_model(input_ids=light_ids, attention_mask=light_mask)
        heavy_hidden = heavy_out.last_hidden_state
        light_hidden = light_out.last_hidden_state
        if self.pooler:
            heavy_repr = self.pooler(heavy_hidden, heavy_mask)
            light_repr = self.pooler(light_hidden, light_mask)
        else:
            heavy_repr = self._mean_pool(heavy_hidden, heavy_mask)
            light_repr = self._mean_pool(light_hidden, light_mask)
        combined = torch.cat([heavy_repr, light_repr], dim=1)
        return combined

def extract_embeddings(embedder_model, data_loader, device_in_func):
    embedder_model.eval()
    embedder_model.to(device_in_func)
    all_feats, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting embeddings"):
            heavy_ids, heavy_mask = batch["heavy_input_ids"].to(device_in_func), batch["heavy_attention_mask"].to(device_in_func)
            light_ids, light_mask = batch["light_input_ids"].to(device_in_func), batch["light_attention_mask"].to(device_in_func)
            labels = batch["label"].cpu().numpy()
            embeddings = embedder_model(heavy_ids, heavy_mask, light_ids, light_mask)
            all_feats.append(embeddings.cpu().numpy())
            all_labels.append(labels)
    X = np.concatenate(all_feats, axis=0) if all_feats else np.array([])
    y = np.concatenate(all_labels, axis=0) if all_labels else np.array([])
    return X, y

all_esm_cv_results_list = []

for current_esm_model_name in esm_models_to_iterate:
    print(f"\n\n===== PROCESSING ESM MODEL: {current_esm_model_name} =====")
    current_esm_hidden_dim = model_configurations[current_esm_model_name][0]

    print(f"Loading {current_esm_model_name}...")
    try:
        tokenizer_esm_global = AutoTokenizer.from_pretrained(current_esm_model_name)
        base_esm_model_instance = AutoModel.from_pretrained(current_esm_model_name)
    except Exception as e:
        print(f"Error loading {current_esm_model_name}: {e}. Skipping this model.")
        continue

    esm_embedder_instance = ESMEmbedder(
        esm_model_instance=base_esm_model_instance,
        hidden_dim=current_esm_hidden_dim,
        use_attention_pool=USE_ATTENTION_POOLING_ESM
    ).to(device)

    print(f"Extracting embeddings for {current_esm_model_name} on the full dataset...")
    embedding_full_dataset = AntibodyEmbeddingDataset(dataset_for_cv, tokenizer_esm_global)
    embedding_full_loader = DataLoader(embedding_full_dataset, batch_size=BATCH_SIZE_EMBEDDING_ESM, shuffle=False, collate_fn=collate_fn_esm, num_workers=0)
    X_full_dataset_current_esm, y_full_dataset_current_esm_labels = extract_embeddings(esm_embedder_instance, embedding_full_loader, device)

    print(f"Embeddings for {current_esm_model_name} (Full Dataset):")
    print(f"  X_full_dataset shape: {X_full_dataset_current_esm.shape}, y_full_dataset shape: {y_full_dataset_current_esm_labels.shape}")

    if X_full_dataset_current_esm.size == 0:
        print(f"No embeddings extracted for {current_esm_model_name}. Skipping ML.")
        continue

    print(f"\n--- Classical ML Classification for {current_esm_model_name} (Full Dataset CV) ---")
    models_clf_current_esm = {
        "LogisticRegression": LogisticRegression(random_state=SEED, max_iter=1000, solver='liblinear', class_weight='balanced'),
        "SVC_RBF": SVC(kernel='rbf', C=1.0, probability=True, random_state=SEED, class_weight='balanced'),
        "SVC_Linear": SVC(kernel='linear', C=1.0, probability=True, random_state=SEED, class_weight='balanced'),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, class_weight='balanced'),
        "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        "XGBClassifier": XGBClassifier(n_estimators=100, random_state=SEED, verbosity=0, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
    }
    cv_results_this_esm = {ml_name: {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mcc']}
                            for ml_name in models_clf_current_esm.keys()}
    
    kfold_outer_current_esm = KFold(n_splits=5, shuffle=True, random_state=SEED)

    for i, (train_idx, val_idx) in enumerate(kfold_outer_current_esm.split(X_full_dataset_current_esm, y_full_dataset_current_esm_labels), start=1):
        print(f"\n  ESM: {current_esm_model_name} - Classification Fold {i}")
        X_train_fold, X_val_fold = X_full_dataset_current_esm[train_idx], X_full_dataset_current_esm[val_idx]
        y_train_fold, y_val_fold = y_full_dataset_current_esm_labels[train_idx], y_full_dataset_current_esm_labels[val_idx]

        for ml_model_name, model_instance_clf in models_clf_current_esm.items():
            current_model_clf = model_instance_clf
            try:
                if ml_model_name == "XGBClassifier":
                    counts_fold = np.bincount(y_train_fold.astype(int))
                    current_model_clf.scale_pos_weight = (counts_fold[0] / counts_fold[1]) if len(counts_fold) == 2 and counts_fold[1] > 0 else 1
                
                current_model_clf.fit(X_train_fold, y_train_fold)
                val_preds = current_model_clf.predict(X_val_fold)
                val_preds_proba = current_model_clf.predict_proba(X_val_fold)[:, 1] if hasattr(current_model_clf, "predict_proba") else np.full(len(y_val_fold), np.nan)

                cv_results_this_esm[ml_model_name]['accuracy'].append(accuracy_score(y_val_fold, val_preds))
                cv_results_this_esm[ml_model_name]['precision'].append(precision_score(y_val_fold, val_preds, zero_division=0))
                cv_results_this_esm[ml_model_name]['recall'].append(recall_score(y_val_fold, val_preds, zero_division=0))
                cv_results_this_esm[ml_model_name]['f1'].append(f1_score(y_val_fold, val_preds, zero_division=0))
                cv_results_this_esm[ml_model_name]['mcc'].append(matthews_corrcoef(y_val_fold, val_preds))
                
                roc_auc_val = np.nan
                if not np.all(np.isnan(val_preds_proba)) and len(np.unique(y_val_fold)) > 1:
                    try: roc_auc_val = roc_auc_score(y_val_fold, val_preds_proba)
                    except ValueError: pass
                cv_results_this_esm[ml_model_name]['roc_auc'].append(roc_auc_val)
            except Exception as e:
                print(f"      ERROR training/evaluating {ml_model_name} on fold {i} for {current_esm_model_name}: {e}")
                for metric_list in cv_results_this_esm[ml_model_name].values(): metric_list.append(np.nan)
    
    for ml_model_name_agg, metrics_data_agg in cv_results_this_esm.items():
        all_esm_cv_results_list.append({
            'ESM_Model': current_esm_model_name,
            'ML_Model': ml_model_name_agg,
            'Avg_Accuracy': np.nanmean(metrics_data_agg['accuracy']),
            'Std_Accuracy': np.nanstd(metrics_data_agg['accuracy']),
            'Avg_Precision': np.nanmean(metrics_data_agg['precision']),
            'Std_Precision': np.nanstd(metrics_data_agg['precision']),
            'Avg_Recall': np.nanmean(metrics_data_agg['recall']),
            'Std_Recall': np.nanstd(metrics_data_agg['recall']),
            'Avg_F1': np.nanmean(metrics_data_agg['f1']),
            'Std_F1': np.nanstd(metrics_data_agg['f1']),
            'Avg_ROC_AUC': np.nanmean(metrics_data_agg['roc_auc']),
            'Std_ROC_AUC': np.nanstd(metrics_data_agg['roc_auc']),
            'Avg_MCC': np.nanmean(metrics_data_agg['mcc']),
            'Std_MCC': np.nanstd(metrics_data_agg['mcc']),
        })

    del base_esm_model_instance, esm_embedder_instance
    if device.type == 'cuda': torch.cuda.empty_cache()
    elif device.type == 'mps': torch.mps.empty_cache()

results_df = pd.DataFrame(all_esm_cv_results_list)

results_df
