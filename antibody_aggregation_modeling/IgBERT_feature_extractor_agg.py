import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer

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
        label = row["z_ac"] 

        heavy_spaced = " ".join(list(heavy))
        light_spaced = " ".join(list(light))

        heavy_inputs = self.tokenizer(
            heavy_spaced,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        light_inputs = self.tokenizer(
            light_spaced,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )

        return {
            "heavy_input_ids": heavy_inputs["input_ids"].squeeze(0),
            "heavy_attention_mask": heavy_inputs["attention_mask"].squeeze(0),
            "light_input_ids": light_inputs["input_ids"].squeeze(0),
            "light_attention_mask": light_inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }

tokenizer_igbert_global = None 

def collate_fn_dynamic_igbert(batch):
    heavy_ids = [item["heavy_input_ids"] for item in batch]
    heavy_masks = [item["heavy_attention_mask"] for item in batch]
    light_ids = [item["light_input_ids"] for item in batch]
    light_masks = [item["light_attention_mask"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    heavy_ids_padded = pad_sequence(
        heavy_ids, batch_first=True, padding_value=tokenizer_igbert_global.pad_token_id
    )
    heavy_masks_padded = pad_sequence(
        heavy_masks, batch_first=True, padding_value=0
    )
    light_ids_padded = pad_sequence(
        light_ids, batch_first=True, padding_value=tokenizer_igbert_global.pad_token_id
    )
    light_masks_padded = pad_sequence(
        light_masks, batch_first=True, padding_value=0
    )

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

class DeeperHead(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LeakyReLU(negative_slope=0.01), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LeakyReLU(negative_slope=0.01), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LeakyReLU(negative_slope=0.01), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.LeakyReLU(negative_slope=0.01), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.regressor(x).squeeze(-1)

class RegressionModel(nn.Module):
    def __init__(
        self, plm_model, hidden_dim, head_type="medium",
        dropout=0.1, use_attention_pool=True, freeze_plm=True
    ):
        super().__init__()
        self.plm_model = plm_model
        self.plm_hidden_dim = hidden_dim

        if freeze_plm:
            print("Freezing base PLM (IgBERT) parameters.")
            for param in self.plm_model.parameters():
                param.requires_grad = False
        else:
            print("WARNING: Base PLM (IgBERT) parameters are NOT frozen (full fine-tuning).")

        self.use_attention_pool = use_attention_pool
        if self.use_attention_pool:
            self.pooler = AttentionPooling(self.plm_hidden_dim)
        else:
            self.pooler = None

        head_input_dim = 2 * self.plm_hidden_dim 
        if head_type == "simple": self.head = SimpleHead(head_input_dim)
        elif head_type == "medium": self.head = MediumHead(head_input_dim, dropout)
        elif head_type == "deep": self.head = DeepHead(head_input_dim, dropout)
        elif head_type == "deeper": self.head = DeeperHead(head_input_dim, dropout)
        else: raise ValueError(f"Invalid head type: {head_type}")
        print(f"Regression head input dimension: {head_input_dim}")


    def forward(self, heavy_ids, heavy_mask, light_ids, light_mask):
        heavy_out = self.plm_model(input_ids=heavy_ids, attention_mask=heavy_mask)
        light_out = self.plm_model(input_ids=light_ids, attention_mask=light_mask)
        heavy_hidden = heavy_out.last_hidden_state
        light_hidden = light_out.last_hidden_state

        if self.pooler is not None:
            heavy_repr = self.pooler(heavy_hidden, heavy_mask)
            light_repr = self.pooler(light_hidden, light_mask)
        else:
            heavy_mask_f = heavy_mask.unsqueeze(-1).float()
            heavy_sum = (heavy_hidden * heavy_mask_f).sum(dim=1)
            heavy_len = heavy_mask_f.sum(dim=1).clamp(min=1e-9)
            heavy_repr = heavy_sum / heavy_len
            light_mask_f = light_mask.unsqueeze(-1).float()
            light_sum = (light_hidden * light_mask_f).sum(dim=1)
            light_len = light_mask_f.sum(dim=1).clamp(min=1e-9)
            light_repr = light_sum / light_len
            
        combined = torch.cat([heavy_repr, light_repr], dim=1)
        preds = self.head(combined)
        return preds

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss):
        if self.best_val_loss - val_loss > self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_one_fold(
    model, train_loader, val_loader, fold_num,
    num_epochs=15, lr=1e-3, weight_decay=0.01, use_scheduler=True, grad_clip_value=1.0
):
    model.to(device)
    criterion = nn.MSELoss()
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=False)
    
    early_stopper = EarlyStopping(patience=3)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            heavy_ids = batch["heavy_input_ids"].to(device)
            heavy_mask = batch["heavy_attention_mask"].to(device)
            light_ids = batch["light_input_ids"].to(device)
            light_mask = batch["light_attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            preds = model(heavy_ids, heavy_mask, light_ids, light_mask)
            loss = criterion(preds, labels)
            loss.backward()
            
            if grad_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 
                    grad_clip_value
                )
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses) if train_losses else float('nan')

        model.eval()
        val_losses = []
        all_preds_val, all_labels_val = [], []
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
        
        avg_val_loss = np.mean(val_losses) if val_losses else float('nan')
        
        spearman_corr_val = np.nan
        np_all_preds_val = np.array(all_preds_val)
        np_all_labels_val = np.array(all_labels_val)
        if len(np_all_preds_val) >= 2 and len(np_all_labels_val) >= 2:
            var_preds = np.var(np_all_preds_val)
            var_labels = np.var(np_all_labels_val)
            if var_preds > 1e-9 and var_labels > 1e-9:
                spearman_corr_val, _ = spearmanr(np_all_preds_val, np_all_labels_val)
            else:
                print(f"[Fold {fold_num} Epoch {epoch+1:02d}] Warning: Constant val input for Spearman. Preds var: {var_preds:.2e}, Labels var: {var_labels:.2e}")
        else:
            print(f"[Fold {fold_num} Epoch {epoch+1:02d}] Warning: Not enough val data for Spearman.")

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        print(f"[Fold {fold_num} Epoch {epoch+1:02d}] Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Spearman: {spearman_corr_val:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        early_stopper.step(avg_val_loss)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at Fold {fold_num} Epoch {epoch+1}!")
            break
    return model

igbert_model_name = "Exscientia/IgBERT"

print(f"\n--- Using IgBERT Model: {igbert_model_name} (Feature Extraction) ---")
tokenizer_igbert_global = AutoTokenizer.from_pretrained(igbert_model_name)
print("Loading base IgBERT model...")
base_igbert_model = AutoModel.from_pretrained(igbert_model_name)
igbert_actual_hidden_dim = base_igbert_model.config.hidden_size
print(f"IgBERT actual hidden dimension: {igbert_actual_hidden_dim}")

train_val_df = df.copy()

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
train_val_data = train_val_df.reset_index(drop=True)

for fold_idx_cv, (train_indices, val_indices) in enumerate(kfold.split(train_val_data)):
    current_fold_num_cv = fold_idx_cv + 1
    print(f"\n==== Fold {current_fold_num_cv} ====")

    fold_train_df = train_val_data.iloc[train_indices].copy()
    fold_val_df = train_val_data.iloc[val_indices].copy()

    pt_scaler_fold = PowerTransformer(method="yeo-johnson", standardize=False)

    fold_train_df["yj_ac"] = pt_scaler_fold.fit_transform(fold_train_df[["agg"]])
    fold_val_df["yj_ac"] = pt_scaler_fold.transform(fold_val_df[["agg"]])

    std_scaler_fold = StandardScaler()

    fold_train_df["z_ac"] = std_scaler_fold.fit_transform(fold_train_df[["yj_ac"]])
    fold_val_df["z_ac"] = std_scaler_fold.transform(fold_val_df[["yj_ac"]])

    train_dataset = AntibodyDataset(fold_train_df, tokenizer_igbert_global)
    val_dataset = AntibodyDataset(fold_val_df, tokenizer_igbert_global)

    num_dl_workers = 0
    batch_size_igbert = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size_igbert, shuffle=True, collate_fn=collate_fn_dynamic_igbert, num_workers=num_dl_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_igbert, shuffle=False, collate_fn=collate_fn_dynamic_igbert, num_workers=num_dl_workers)

    model_fold = RegressionModel(
        plm_model=base_igbert_model.to(device),
        hidden_dim=igbert_actual_hidden_dim,
        head_type="deep",
        dropout=0.1,
        use_attention_pool=True,
        freeze_plm=True
    ).to(device)

    print(f"RegressionModel with IgBERT created for Fold {current_fold_num_cv}.")

    lr_head_training = 1e-4
    trained_model_fold = train_one_fold(
        model=model_fold, train_loader=train_loader, val_loader=val_loader,
        fold_num=current_fold_num_cv,
        num_epochs=10, lr=lr_head_training,
        weight_decay=0.01, use_scheduler=True, grad_clip_value=1.0
    )

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

    avg_val_loss_cv = np.mean(val_losses_cv_fold) if val_losses_cv_fold else float('nan')
    val_spearman_cv = np.nan
    if len(val_preds_cv) >= 2 and len(val_labels_cv) >=2:
        var_preds_cv = np.var(val_preds_cv)
        var_labels_cv = np.var(val_labels_cv)
        if var_preds_cv > 1e-9 and var_labels_cv > 1e-9:
            val_spearman_cv, _ = spearmanr(val_preds_cv, val_labels_cv)

    fold_results.append((avg_val_loss_cv, val_spearman_cv))
    print(f"---- Fold {current_fold_num_cv} Final Validation ----")
    print(f"Loss: {avg_val_loss_cv:.4f}, Spearman: {val_spearman_cv:.4f}")

    del model_fold, trained_model_fold
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()

if fold_results:
    avg_losses = np.nanmean([res[0] for res in fold_results])
    avg_spearman = np.nanmean([res[1] for res in fold_results])
    print(f"\nAverage CV Loss: {avg_losses:.4f}")
    print(f"Average CV Spearman: {avg_spearman:.4f}")
else:
    print("No fold results to average.")

if fold_results:
    val_losses = [r[0] for r in fold_results]
    val_spearmans = [r[1] for r in fold_results]
    cv_val_loss = np.mean(val_losses)
    cv_val_spearman = np.mean(val_spearmans)
    sd_val_loss = np.std(val_losses, ddof=1) if len(val_losses) > 1 else 0.0
    sd_val_spearman = np.std(val_spearmans, ddof=1) if len(val_spearmans) > 1 else 0.0

    print(f"\n=== Cross-Validation Results ({kfold.n_splits}-Fold) ===")
    print(f"Average Validation Loss:       {cv_val_loss:.4f} +/- {sd_val_loss:.4f}")
    print(f"Average Validation Spearman:   {cv_val_spearman:.4f} +/- {sd_val_spearman:.4f}")
else:
    print("\nCross-validation was not performed (e.g. due to insufficient data).")


print("\nNote: For final test evaluation, retrain the model on the full Train+Val set")
print("      using the best hyperparameters and apply scaling fitted on that set.")
print("      Alternatively, use an ensemble of the k-fold models.")

print("\nIgBERT modeling script finished.")
