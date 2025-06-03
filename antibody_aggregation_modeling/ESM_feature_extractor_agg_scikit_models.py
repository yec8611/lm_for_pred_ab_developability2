import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
import warnings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LassoCV, ElasticNetCV
from xgboost import XGBRegressor

df1 = pd.read_excel('pnas.1616408114.sd01.xlsx')
df2 = pd.read_excel('pnas.1616408114.sd02.xlsx') 
df3 = pd.read_excel('pnas.1616408114.sd03.xlsx')

merged_df = df1.merge(df2, on='Name', how='outer').merge(df3, on='Name', how='outer')
             
df = merged_df[['VH', 'VL', 'Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average']].copy()
df = df.rename(columns={'Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average': 'agg'})

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"\nUsing device: {device}\n")

model_configurations = {
    "facebook/esm2_t6_8M_UR50D": (320, "simple"),
    "facebook/esm2_t12_35M_UR50D": (480, "medium"),
    "facebook/esm2_t30_150M_UR50D": (640, "medium"),
    "facebook/esm2_t33_650M_UR50D": (1280, "deep"),
    "facebook/esm2_t36_3B_UR50D": (2560, "deeper")
}

class AntibodyEmbeddingDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        heavy = str(row["VH"])
        light = str(row["VL"])
        label = row["agg"]

        heavy_inputs = self.tokenizer(
            heavy, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        light_inputs = self.tokenizer(
            light, truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        return {
            "heavy_input_ids": heavy_inputs["input_ids"].squeeze(0),
            "heavy_attention_mask": heavy_inputs["attention_mask"].squeeze(0),
            "light_input_ids": light_inputs["input_ids"].squeeze(0),
            "light_attention_mask": light_inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }

def collate_fn(batch):
    heavy_ids = [item["heavy_input_ids"] for item in batch]
    heavy_masks = [item["heavy_attention_mask"] for item in batch]
    light_ids = [item["light_input_ids"] for item in batch]
    light_masks = [item["light_attention_mask"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    heavy_ids_padded = pad_sequence(heavy_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    heavy_masks_padded = pad_sequence(heavy_masks, batch_first=True, padding_value=0)
    light_ids_padded = pad_sequence(light_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
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
    def __init__(self, esm_model, hidden_dim, use_attention_pool=False):
        super().__init__()
        self.esm_model = esm_model
        for param in self.esm_model.parameters():
            param.requires_grad = False

        self.use_attention_pool = use_attention_pool
        if use_attention_pool:
            self.pooler = AttentionPooling(hidden_dim)
            print("Warning: Using AttentionPooling. Its weights won't be trained.")
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

def extract_embeddings(model, data_loader, device):
    model.eval()
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting embeddings"):
            heavy_ids = batch["heavy_input_ids"].to(device)
            heavy_mask = batch["heavy_attention_mask"].to(device)
            light_ids = batch["light_input_ids"].to(device)
            light_mask = batch["light_attention_mask"].to(device)
            labels = batch["label"].cpu().numpy()
            embeddings = model(heavy_ids, heavy_mask, light_ids, light_mask)
            all_feats.append(embeddings.cpu().numpy())
            all_labels.append(labels)

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y

train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
print(f"\nTrain+Val: {len(train_val_df)}, Test: {len(test_df)}")

chosen_model_name = "facebook/esm2_t6_8M_UR50D"
hidden_dim = 320
print(f"Using ESM Model: {chosen_model_name}")

tokenizer = AutoTokenizer.from_pretrained(chosen_model_name)
base_esm_model = AutoModel.from_pretrained(chosen_model_name)

embedder = ESMEmbedder(
    esm_model=base_esm_model,
    hidden_dim=hidden_dim,
    use_attention_pool=False
).to(device)

train_val_dataset = AntibodyEmbeddingDataset(train_val_df, tokenizer)
test_dataset = AntibodyEmbeddingDataset(test_df, tokenizer)

train_val_loader = DataLoader(train_val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

X_train_val, y_train_val = extract_embeddings(embedder, train_val_loader, device)
X_test, y_test = extract_embeddings(embedder, test_loader, device)

print(f"\nExtracted Embeddings:")
print(f"X_train_val shape: {X_train_val.shape}, y_train_val shape: {y_train_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

inner_cv_folds = 3

models = {
    "Ridge": Ridge(alpha=1.0),
    "LassoCV": LassoCV(cv=inner_cv_folds, random_state=42, max_iter=5000, n_jobs=-1),
    "ElasticNetCV": ElasticNetCV(cv=inner_cv_folds, random_state=42, max_iter=5000, n_jobs=-1),
    "SVR_RBF": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "SVR_Linear": SVR(kernel='linear', C=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=-1)
}

cv_results = {model_name: [] for model_name in models.keys()}
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for i, (train_idx, val_idx) in enumerate(kfold.split(X_train_val), start=1):
    print(f"\n==================== Fold {i} ====================")

    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train_orig, y_val_orig = y_train_val[train_idx], y_train_val[val_idx]

    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    sc = StandardScaler()

    y_train_yj = pt.fit_transform(y_train_orig.reshape(-1, 1))
    y_train_scaled = sc.fit_transform(y_train_yj).ravel()

    y_val_yj = pt.transform(y_val_orig.reshape(-1, 1))
    y_val_scaled = sc.transform(y_val_yj).ravel()

    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train_scaled)

            if hasattr(model, 'alpha_'):
                print(f"  [{model_name:16}] Trained. Best alpha: {model.alpha_:.4f}")
            elif hasattr(model, 'best_params_') and 'alpha' in model.best_params_:
                 print(f"  [{model_name:16}] Trained. Best alpha: {model.best_params_['alpha']:.4f}")
            else:
                print(f"  [{model_name:16}] Trained.")

            val_preds_scaled = model.predict(X_val)

            val_preds_yj = sc.inverse_transform(val_preds_scaled.reshape(-1, 1))
            val_preds_orig = pt.inverse_transform(val_preds_yj).ravel()

            val_mse = mean_squared_error(y_val_orig, val_preds_orig)
            if len(np.unique(val_preds_orig)) < 2 or len(np.unique(y_val_orig)) < 2:
                val_spear = np.nan
                print(f"    -> Constant predictions or labels for {model_name}, Spearman set to NaN.")
            else:
                val_spear, _ = spearmanr(y_val_orig, val_preds_orig)

            cv_results[model_name].append((val_mse, val_spear))
            print(f"    Val MSE: {val_mse:.4f} | Spearman: {val_spear:.4f}")

        except Exception as e:
            print(f"  [{model_name:16}] Failed: {e}")

print("\n=== Cross-Validation Summary ===")
for model_name, results in cv_results.items():
    if not results: continue
    mses = [r[0] for r in results]
    spearmans = [r[1] for r in results if not np.isnan(r[1])]
    if not spearmans:
        avg_spearman = np.nan
        std_spearman = np.nan
    else:
        avg_spearman = np.mean(spearmans)
        std_spearman = np.std(spearmans)

    avg_mse = np.mean(mses)
    std_mse = np.std(mses)

    print(f"[{model_name:16}] "
          f"Avg MSE: {avg_mse:.4f} +/- {std_mse:.4f} | "
          f"Avg Spearman: {avg_spearman:.4f} +/- {std_spearman:.4f}")


