import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

df1 = pd.read_excel('pnas.1616408114.sd01.xlsx')
df2 = pd.read_excel('pnas.1616408114.sd02.xlsx') 
df3 = pd.read_excel('pnas.1616408114.sd03.xlsx')

merged_df = df1.merge(df2, on='Name', how='outer').merge(df3, on='Name', how='outer')
             
df = merged_df[['VH', 'VL', 'Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average']].copy()
df = df.rename(columns={'Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average': 'agg'})

df

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

train_val_df_orig, test_df_orig = train_test_split(df, test_size=0.15, random_state=42)
print(f"Train/Validation set size: {len(train_val_df_orig)}")
print(f"Test set size: {len(test_df_orig)}")

class AntibodyDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
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
        label = row["scaled_label"]

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

tokenizer_global = None

def collate_fn_dynamic(batch):
    heavy_ids = [item["heavy_input_ids"] for item in batch]
    heavy_masks = [item["heavy_attention_mask"] for item in batch]
    light_ids = [item["light_input_ids"] for item in batch]
    light_masks = [item["light_attention_mask"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    heavy_ids_padded = pad_sequence(heavy_ids, batch_first=True, padding_value=tokenizer_global.pad_token_id)
    heavy_masks_padded = pad_sequence(heavy_masks, batch_first=True, padding_value=0)
    light_ids_padded = pad_sequence(light_ids, batch_first=True, padding_value=tokenizer_global.pad_token_id)
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
        att_scores = att_scores.masked_fill(attention_mask == 0, float("-inf"))
        att_weights = torch.softmax(att_scores, dim=-1).unsqueeze(-1)
        pooled = torch.sum(token_embeddings * att_weights, dim=1)
        return pooled

class SimpleHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    def forward(self, x): return self.fc(x).squeeze(-1)

class MediumHead(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(negative_slope=0.01), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.regressor(x).squeeze(-1)

class DeepHead(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(negative_slope=0.01), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LeakyReLU(negative_slope=0.01), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.regressor(x).squeeze(-1)

class RegressionModelWithLoRA(nn.Module):
    def __init__(self, esm_model, hidden_dim, head_type="simple", dropout=0.1, use_attention_pool=True):
        super().__init__()
        self.esm_model = esm_model
        self.use_attention_pool = use_attention_pool
        if self.use_attention_pool:
            self.pooler = AttentionPooling(hidden_dim)
        else:
            self.pooler = None

        input_dim = 2 * hidden_dim
        head_map = {
            "simple": SimpleHead(input_dim),
            "medium": MediumHead(input_dim, dropout),
            "deep": DeepHead(input_dim, dropout)
        }
        if head_type not in head_map: raise ValueError(f"Invalid head type: {head_type}")
        self.head = head_map[head_type]

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
        preds = self.head(combined)
        return preds

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def step(self, val_loss, model):
        if self.best_val_loss - val_loss > self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}"
    )

def train_one_fold(model, train_loader, val_loader, fold_num, num_epochs=20, lr=5e-5, grad_clip_value=1.0):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    early_stopper = EarlyStopping(patience=5)
    best_fold_spearman = -2.0

    print(f"Starting Fold {fold_num} training...")

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Fold {fold_num} Epoch {epoch+1} Train", leave=False):
            heavy_ids = batch["heavy_input_ids"].to(device)
            heavy_mask = batch["heavy_attention_mask"].to(device)
            light_ids = batch["light_input_ids"].to(device)
            light_mask = batch["light_attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            preds = model(heavy_ids, heavy_mask, light_ids, light_mask)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_clip_value)
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_losses, all_preds_val, all_labels_val = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                heavy_ids = batch["heavy_input_ids"].to(device)
                heavy_mask = batch["heavy_attention_mask"].to(device)
                light_ids = batch["light_input_ids"].to(device)
                light_mask = batch["light_attention_mask"].to(device)
                labels = batch["label"].to(device)
                preds = model(heavy_ids, heavy_mask, light_ids, light_mask)
                loss = criterion(preds, labels)
                val_losses.append(loss.item())
                all_preds_val.extend(preds.cpu().numpy())
                all_labels_val.extend(labels.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        spearman_corr_val = np.nan
        if len(all_preds_val) >= 2 and np.var(all_preds_val) > 1e-9 and np.var(all_labels_val) > 1e-9:
            spearman_corr_val, _ = spearmanr(all_preds_val, all_labels_val)

        scheduler.step(avg_val_loss)
        print(f"[Fold {fold_num} Ep {epoch+1:02d}] Tr Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Spearman: {spearman_corr_val:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if spearman_corr_val > best_fold_spearman:
             best_fold_spearman = spearman_corr_val
             early_stopper.best_model_state = model.state_dict()
             print(f"  -> New best Spearman for fold: {best_fold_spearman:.4f}")

        early_stopper.step(avg_val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at Fold {fold_num} Epoch {epoch+1}!")
            break

    if early_stopper.best_model_state:
        model.load_state_dict(early_stopper.best_model_state)
        print(f"Loaded best model state for Fold {fold_num} (Val Loss: {early_stopper.best_val_loss:.4f}).")
    else:
        print(f"Warning: No best model state found for Fold {fold_num}.")

    return model

chosen_model_name = "facebook/esm2_t6_8M_UR50D"
hidden_dim, head_type = model_configurations[chosen_model_name]

tokenizer_global = AutoTokenizer.from_pretrained(chosen_model_name)

lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["query", "key", "value"],
    lora_dropout=0.1, bias="none",
)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
trained_models = []
scalers = []

train_val_data_for_kfold = train_val_df_orig.reset_index(drop=True)

for fold_idx_cv, (train_indices, val_indices) in enumerate(kfold.split(train_val_data_for_kfold)):
    current_fold_num_cv = fold_idx_cv + 1
    print(f"\n==================== Fold {current_fold_num_cv} ====================")

    fold_train_df = train_val_data_for_kfold.iloc[train_indices].copy()
    fold_val_df = train_val_data_for_kfold.iloc[val_indices].copy()

    pt_fold = PowerTransformer(method="yeo-johnson", standardize=False)
    sc_fold = StandardScaler()

    fold_train_df["yj_ac"] = pt_fold.fit_transform(fold_train_df[["agg"]])
    fold_train_df["scaled_label"] = sc_fold.fit_transform(fold_train_df[["yj_ac"]])

    fold_val_df["yj_ac"] = pt_fold.transform(fold_val_df[["agg"]])
    fold_val_df["scaled_label"] = sc_fold.transform(fold_val_df[["yj_ac"]])
    scalers.append({'pt': pt_fold, 'sc': sc_fold})

    train_dataset = AntibodyDataset(fold_train_df, tokenizer_global)
    val_dataset = AntibodyDataset(fold_val_df, tokenizer_global)

    num_dl_workers = 0
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_dynamic, num_workers=num_dl_workers)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_dynamic, num_workers=num_dl_workers)

    base_esm_model_fold = AutoModel.from_pretrained(chosen_model_name)
    peft_model_fold = get_peft_model(base_esm_model_fold, lora_config)

    model_fold = RegressionModelWithLoRA(
        esm_model=peft_model_fold,
        hidden_dim=hidden_dim, head_type=head_type,
        dropout=0.1, use_attention_pool=True
    ).to(device)

    print(f"Full Model - Fold {current_fold_num_cv}. Trainable parameters:")
    print_trainable_parameters(model_fold)

    trained_model_fold = train_one_fold(
        model=model_fold, train_loader=train_loader, val_loader=val_loader,
        fold_num=current_fold_num_cv, num_epochs=15, lr=5e-5, grad_clip_value=1.0
    )
    trained_models.append(trained_model_fold)

    trained_model_fold.eval()
    val_preds_cv, val_labels_cv, val_losses_cv_fold = [], [], []
    criterion_eval = nn.MSELoss()
    with torch.no_grad():
        for batch in val_loader:
            heavy_ids = batch["heavy_input_ids"].to(device)
            heavy_mask = batch["heavy_attention_mask"].to(device)
            light_ids = batch["light_input_ids"].to(device)
            light_mask = batch["light_attention_mask"].to(device)
            labels = batch["label"].to(device)
            preds = trained_model_fold(heavy_ids, heavy_mask, light_ids, light_mask)
            loss = criterion_eval(preds, labels)
            val_losses_cv_fold.append(loss.item())
            val_preds_cv.extend(preds.cpu().numpy())
            val_labels_cv.extend(labels.cpu().numpy())

    avg_val_loss_cv = np.mean(val_losses_cv_fold)
    val_spearman_cv = np.nan
    if len(val_preds_cv) >= 2 and np.var(val_preds_cv) > 1e-9 and np.var(val_labels_cv) > 1e-9:
        val_spearman_cv, _ = spearmanr(val_preds_cv, val_labels_cv)

    fold_results.append((avg_val_loss_cv, val_spearman_cv))
    print(f"---- Fold {current_fold_num_cv} Final Validation (Scaled) ----")
    print(f"Loss: {avg_val_loss_cv:.4f}, Spearman: {val_spearman_cv:.4f}")

    del model_fold, base_esm_model_fold, peft_model_fold
    if device.type == 'cuda': torch.cuda.empty_cache()
    elif device.type == 'mps': torch.mps.empty_cache()

val_losses_all = [r[0] for r in fold_results if not np.isnan(r[0])]
val_spearmans_all = [r[1] for r in fold_results if not np.isnan(r[1])]

cv_val_loss_mean = np.mean(val_losses_all) if val_losses_all else np.nan
cv_val_loss_std = np.std(val_losses_all) if len(val_losses_all) > 1 else np.nan
cv_val_spearman_mean = np.mean(val_spearmans_all) if val_spearmans_all else np.nan
cv_val_spearman_std = np.std(val_spearmans_all) if len(val_spearmans_all) > 1 else np.nan

print("\n--- Cross-Validation Summary (on Scaled Data) ---")
print(f"Average Validation Loss: {cv_val_loss_mean:.4f} +/- {cv_val_loss_std:.4f}")
print(f"Average Validation Spearman: {cv_val_spearman_mean:.4f} +/- {cv_val_spearman_std:.4f}")


