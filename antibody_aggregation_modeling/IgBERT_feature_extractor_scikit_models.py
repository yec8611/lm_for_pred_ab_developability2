import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

from sklearn.linear_model import Ridge, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

IGBERT_MODEL_NAME = "Exscientia/IgBERT"
MAX_LENGTH_PER_CHAIN = 256
BATCH_SIZE_EMBEDDING = 16
USE_ATTENTION_POOLING_FOR_FEATURES = True
SEED = 42

print(f"\nLoading IgBERT model and tokenizer: {IGBERT_MODEL_NAME}")
try:
    tokenizer_igbert = AutoTokenizer.from_pretrained(IGBERT_MODEL_NAME)
    base_igbert_model = AutoModel.from_pretrained(IGBERT_MODEL_NAME)
    base_igbert_model.to(DEVICE)
    base_igbert_model.eval()
    IGBERT_HIDDEN_DIM = base_igbert_model.config.hidden_size
    print(f"IgBERT model loaded. Hidden dimension: {IGBERT_HIDDEN_DIM}")
except Exception as e:
    print(f"Error loading IgBERT model or tokenizer: {e}")
    exit()

class AntibodyFeatureDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=MAX_LENGTH_PER_CHAIN):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.df['VH'] = self.df['VH'].astype(str)
        self.df['VL'] = self.df['VL'].astype(str)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        heavy = row["VH"]
        light = row["VL"]

        heavy_spaced = " ".join(list(heavy))
        light_spaced = " ".join(list(light))

        heavy_inputs = self.tokenizer(
            heavy_spaced, truncation=True, max_length=self.max_length,
            padding=False, return_tensors="pt"
        )
        light_inputs = self.tokenizer(
            light_spaced, truncation=True, max_length=self.max_length,
            padding=False, return_tensors="pt"
        )
        return {
            "heavy_input_ids": heavy_inputs["input_ids"].squeeze(0),
            "heavy_attention_mask": heavy_inputs["attention_mask"].squeeze(0),
            "light_input_ids": light_inputs["input_ids"].squeeze(0),
            "light_attention_mask": light_inputs["attention_mask"].squeeze(0),
        }

def collate_fn_features(batch):
    heavy_ids = [item["heavy_input_ids"] for item in batch]
    heavy_masks = [item["heavy_attention_mask"] for item in batch]
    light_ids = [item["light_input_ids"] for item in batch]
    light_masks = [item["light_attention_mask"] for item in batch]

    heavy_ids_padded = pad_sequence(heavy_ids, batch_first=True, padding_value=tokenizer_igbert.pad_token_id)
    heavy_masks_padded = pad_sequence(heavy_masks, batch_first=True, padding_value=0)
    light_ids_padded = pad_sequence(light_ids, batch_first=True, padding_value=tokenizer_igbert.pad_token_id)
    light_masks_padded = pad_sequence(light_masks, batch_first=True, padding_value=0)

    return {
        "heavy_input_ids": heavy_ids_padded,
        "heavy_attention_mask": heavy_masks_padded,
        "light_input_ids": light_ids_padded,
        "light_attention_mask": light_masks_padded,
    }

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, token_embeddings, attention_mask):
        att_scores = self.attention(token_embeddings).squeeze(-1)
        att_scores = att_scores.masked_fill(attention_mask == 0, float("-inf"))
        att_weights = torch.softmax(att_scores, dim=-1).unsqueeze(-1)
        pooled = torch.sum(token_embeddings * att_weights, dim=1)
        return pooled

def mean_pool(hidden_state, attention_mask):
    mask_f = attention_mask.unsqueeze(-1).float()
    sum_hidden = (hidden_state * mask_f).sum(dim=1)
    len_hidden = mask_f.sum(dim=1).clamp(min=1e-9)
    return sum_hidden / len_hidden

def get_igbert_embeddings(df_input, igbert_model, tokenizer, device, hidden_dim, use_attention_pooling=True):
    print(f"Generating IgBERT embeddings... Pooling: {'Attention' if use_attention_pooling else 'Mean'}")
    dataset = AntibodyFeatureDataset(df_input, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_EMBEDDING, shuffle=False, collate_fn=collate_fn_features)

    if use_attention_pooling:
        pooler = AttentionPooling(hidden_dim).to(device)
        pooler.eval()
    else:
        pooler = None

    all_embeddings = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"  Processing batch {batch_idx + 1}/{len(dataloader)} for embeddings")
            heavy_ids = batch["heavy_input_ids"].to(device)
            heavy_mask = batch["heavy_attention_mask"].to(device)
            light_ids = batch["light_input_ids"].to(device)
            light_mask = batch["light_attention_mask"].to(device)

            heavy_out = igbert_model(input_ids=heavy_ids, attention_mask=heavy_mask)
            light_out = igbert_model(input_ids=light_ids, attention_mask=light_mask)

            heavy_hidden = heavy_out.last_hidden_state
            light_hidden = light_out.last_hidden_state

            if use_attention_pooling and pooler:
                heavy_repr = pooler(heavy_hidden, heavy_mask)
                light_repr = pooler(light_hidden, light_mask)
            else:
                heavy_repr = mean_pool(heavy_hidden, heavy_mask)
                light_repr = mean_pool(light_hidden, light_mask)

            combined_repr = torch.cat([heavy_repr, light_repr], dim=1)
            all_embeddings.append(combined_repr.cpu().numpy())

    concatenated_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Embeddings generated. Shape: {concatenated_embeddings.shape}")
    return concatenated_embeddings

df1 = pd.read_excel('pnas.1616408114.sd01.xlsx')
df2 = pd.read_excel('pnas.1616408114.sd02.xlsx') 
df3 = pd.read_excel('pnas.1616408114.sd03.xlsx')

merged_df = df1.merge(df2, on='Name', how='outer').merge(df3, on='Name', how='outer')
             
df = merged_df[['VH', 'VL', 'Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average']].copy()
df = df.rename(columns={'Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average': 'target_value'})

train_val_df_orig = df.copy()
train_val_df_orig

X_igbert_features = get_igbert_embeddings(
    train_val_df_orig,
    base_igbert_model,
    tokenizer_igbert,
    DEVICE,
    IGBERT_HIDDEN_DIM,
    use_attention_pooling=USE_ATTENTION_POOLING_FOR_FEATURES
)
y_target_labels = train_val_df_orig['target_value'].values

print("\n--- Starting Classical ML Model Training with Cross-Validation ---")
inner_cv_folds = 3

models = {
    "Ridge": Ridge(alpha=1.0, random_state=SEED),
    "LassoCV": LassoCV(cv=inner_cv_folds, random_state=SEED, max_iter=10000, n_jobs=-1, tol=1e-3),
    "ElasticNetCV": ElasticNetCV(cv=inner_cv_folds, random_state=SEED, max_iter=10000, n_jobs=-1, tol=1e-3),
    "SVR_RBF": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "SVR_Linear": SVR(kernel='linear', C=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=SEED),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=SEED, verbosity=0, n_jobs=-1)
}

cv_results_ml = {model_name: {'mse': [], 'spearman': []} for model_name in models.keys()}
outer_kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)

for i, (train_idx, val_idx) in enumerate(outer_kfold.split(X_igbert_features, y_target_labels), start=1):
    print(f"\n==================== Outer Fold {i} for Classical ML ====================")

    X_train_fold, X_val_fold = X_igbert_features[train_idx], X_igbert_features[val_idx]
    y_train_orig_fold, y_val_orig_fold = y_target_labels[train_idx], y_target_labels[val_idx]

    pt_outer = PowerTransformer(method="yeo-johnson", standardize=False)
    sc_outer = StandardScaler()

    y_train_yj_fold = pt_outer.fit_transform(y_train_orig_fold.reshape(-1, 1))
    y_train_scaled_fold = sc_outer.fit_transform(y_train_yj_fold).ravel()

    print(f"  Fold {i}: X_train_fold shape: {X_train_fold.shape}, y_train_scaled_fold shape: {y_train_scaled_fold.shape}")
    print(f"  Fold {i}: X_val_fold shape: {X_val_fold.shape}, y_val_orig_fold shape: {y_val_orig_fold.shape}")


    for model_name, model_instance in models.items():
        print(f"    Training {model_name}...")
        current_model = model_instance

        try:
            current_model.fit(X_train_fold, y_train_scaled_fold)

            if hasattr(current_model, 'alpha_'):
                print(f"      [{model_name:16}] Trained. Best alpha: {current_model.alpha_:.4f}")
            else:
                print(f"      [{model_name:16}] Trained.")

            val_preds_scaled_fold = current_model.predict(X_val_fold)

            val_preds_yj_fold = sc_outer.inverse_transform(val_preds_scaled_fold.reshape(-1, 1))
            val_preds_orig_fold = pt_outer.inverse_transform(val_preds_yj_fold).ravel()

            val_mse = mean_squared_error(y_val_orig_fold, val_preds_orig_fold)
            val_spear = np.nan
            if len(np.unique(val_preds_orig_fold)) >= 2 and len(np.unique(y_val_orig_fold)) >= 2:
                val_spear, _ = spearmanr(y_val_orig_fold, val_preds_orig_fold)
            else:
                print(f"      -> Constant predictions or labels for {model_name}, Spearman set to NaN.")

            cv_results_ml[model_name]['mse'].append(val_mse)
            cv_results_ml[model_name]['spearman'].append(val_spear)
            print(f"      Val MSE: {val_mse:.4f} | Spearman: {val_spear:.4f}")

        except Exception as e:
            print(f"      [{model_name:16}] Failed: {e}")
            cv_results_ml[model_name]['mse'].append(np.nan)
            cv_results_ml[model_name]['spearman'].append(np.nan)

print("\n=== Classical ML Cross-Validation Summary (using IgBERT features) ===")
for model_name, results in cv_results_ml.items():
    avg_mse = np.nanmean(results['mse'])
    std_mse = np.nanstd(results['mse'])
    avg_spearman = np.nanmean(results['spearman'])
    std_spearman = np.nanstd(results['spearman'])

    print(f"[{model_name:16}] "
          f"Avg MSE: {avg_mse:.4f} +/- {std_mse:.4f} | "
          f"Avg Spearman: {avg_spearman:.4f} +/- {std_spearman:.4f}")



