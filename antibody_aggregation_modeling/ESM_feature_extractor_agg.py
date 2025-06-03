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

model_configurations = {
    "facebook/esm2_t6_8M_UR50D": (320, "simple"),
    "facebook/esm2_t12_35M_UR50D": (480, "medium"),
    "facebook/esm2_t30_150M_UR50D": (640, "medium"),
    "facebook/esm2_t33_650M_UR50D": (1280, "deep"),
    "facebook/esm2_t36_3B_UR50D": (2560, "deeper")
}

chosen_model_name = "facebook/esm2_t6_8M_UR50D"
hidden_dim, head_type = model_configurations[chosen_model_name]

print(f"Using ESM Model: {chosen_model_name}")

tokenizer = AutoTokenizer.from_pretrained(chosen_model_name)
base_esm_model = AutoModel.from_pretrained(chosen_model_name)

class AntibodyDataset(Dataset):
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
        label = row["z_ac"]

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

    heavy_ids_padded = pad_sequence(
        heavy_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    heavy_masks_padded = pad_sequence(
        heavy_masks, batch_first=True, padding_value=0
    )
    light_ids_padded = pad_sequence(
        light_ids, batch_first=True, padding_value=tokenizer.pad_token_id
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
        att_scores = att_scores.masked_fill(~(attention_mask.bool()), float("-inf"))
        att_weights = torch.softmax(att_scores, dim=-1).unsqueeze(-1)
        pooled = torch.sum(token_embeddings * att_weights, dim=1)
        return pooled

class SimpleHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)

class MediumHead(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.regressor(x).squeeze(-1)

class DeepHead(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.regressor(x).squeeze(-1)

class DeeperHead(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.regressor(x).squeeze(-1)

class RegressionModel(nn.Module):
    def __init__(self, esm_model, hidden_dim, head_type="simple", dropout=0.1, use_attention_pool=True, freeze_esm=True):
        super().__init__()
        self.esm_model = esm_model
        if freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False

        if use_attention_pool:
            self.pooler = AttentionPooling(hidden_dim)
        else:
            self.pooler = None

        input_dim = 2 * hidden_dim
        head_map = {
            "simple": SimpleHead(input_dim),
            "medium": MediumHead(input_dim, dropout),
            "deep": DeepHead(input_dim, dropout),
            "deeper": DeeperHead(input_dim, dropout)
        }
        if head_type not in head_map:
            raise ValueError(f"Invalid head type: {head_type}")
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
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_one_fold(model, train_loader, val_loader, num_epochs=20, lr=1e-3, use_scheduler=True, patience=5):
    criterion = nn.MSELoss()
    
    trainable_params = list(model.head.parameters())
    if isinstance(model.pooler, AttentionPooling):
        trainable_params += list(model.pooler.parameters())
    
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=2, verbose=True
    ) if use_scheduler else None

    early_stopper = EarlyStopping(patience=patience)
    best_model_state = None
    best_val_spearman = -1.0

    print(f"Starting training for {num_epochs} epochs...")

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
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        all_preds, all_labels = [], []
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
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        spearman_corr, _ = spearmanr(all_preds, all_labels)

        print(f"[Epoch {epoch+1:02d}] Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Spearman: {spearman_corr:.4f}")

        if scheduler:
            scheduler.step(avg_val_loss)

        if spearman_corr > best_val_spearman:
            best_val_spearman = spearman_corr
            best_model_state = model.state_dict()
            print(f"   -> New best Spearman: {best_val_spearman:.4f}, saving model state.")

        early_stopper.step(avg_val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded best model state based on validation Spearman.")

    return model, best_val_spearman

train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
print(f"\nTotal Data: {len(df)}")
print(f"Train+Val: {len(train_val_df)}, Test: {len(test_df)}")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
fold_idx = 1
train_val_data = train_val_df.reset_index(drop=True)

for train_index, val_index in kfold.split(train_val_data):
    print(f"\n==================== Fold {fold_idx} ====================")

    fold_train_df = train_val_data.iloc[train_index].copy()
    fold_val_df = train_val_data.iloc[val_index].copy()

    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    sc = StandardScaler()

    fold_train_df["yj_ac"] = pt.fit_transform(fold_train_df[["agg"]])
    fold_train_df["z_ac"] = sc.fit_transform(fold_train_df[["yj_ac"]])

    fold_val_df["yj_ac"] = pt.transform(fold_val_df[["agg"]])
    fold_val_df["z_ac"] = sc.transform(fold_val_df[["yj_ac"]])
    print(f"Fold {fold_idx}: Train {len(fold_train_df)}, Val {len(fold_val_df)}")

    train_dataset = AntibodyDataset(fold_train_df, tokenizer)
    val_dataset = AntibodyDataset(fold_val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    current_esm_model = AutoModel.from_pretrained(chosen_model_name)
    model = RegressionModel(
        esm_model=current_esm_model,
        hidden_dim=hidden_dim,
        head_type=head_type,
        dropout=0.1, 
        use_attention_pool=True
    ).to(device)

    trained_model, fold_spearman = train_one_fold(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=15, 
        lr=5e-4, 
        use_scheduler=True,
        patience=5
    )

    trained_model.eval()
    val_preds, val_labels = [], []
    criterion = nn.MSELoss()
    val_losses = []

    with torch.no_grad():
        for batch in val_loader:
            heavy_ids = batch["heavy_input_ids"].to(device)
            heavy_mask = batch["heavy_attention_mask"].to(device)
            light_ids = batch["light_input_ids"].to(device)
            light_mask = batch["light_attention_mask"].to(device)
            labels = batch["label"].to(device)

            preds = trained_model(heavy_ids, heavy_mask, light_ids, light_mask)
            loss = criterion(preds, labels)
            val_losses.append(loss.item())
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    avg_val_loss = np.mean(val_losses)
    val_spearman, _ = spearmanr(val_preds, val_labels)
    fold_results.append((avg_val_loss, val_spearman))
    print(f"Fold {fold_idx} Final Validation -> Loss: {avg_val_loss:.4f}, Spearman: {val_spearman:.4f}")
    fold_idx += 1

val_losses = [r[0] for r in fold_results]
val_spearmans = [r[1] for r in fold_results]
cv_val_loss = np.mean(val_losses)
cv_val_spearman = np.mean(val_spearmans)
sd_val_loss = np.std(val_losses, ddof=1)
sd_val_spearman = np.std(val_spearmans, ddof=1)

print(f"\n=== Cross-Validation Results ({kfold.n_splits}-Fold) ===")
print(f"Average Validation Loss:     {cv_val_loss:.4f} +/- {sd_val_loss:.4f}")
print(f"Average Validation Spearman: {cv_val_spearman:.4f} +/- {sd_val_spearman:.4f}")
print("\nNote: For final test evaluation, retrain the model on the full Train+Val set")
print("      using the best hyperparameters and apply scaling fitted on that set.")
print("      Alternatively, use an ensemble of the k-fold models.")


