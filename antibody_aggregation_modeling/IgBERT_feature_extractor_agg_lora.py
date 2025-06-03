import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from scipy.stats import spearmanr
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm.auto import tqdm

IGBERT_MODEL_NAME = "Exscientia/IgBERT"
MAX_LENGTH_PER_CHAIN = 256
HEAD_TYPE = "deep"
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 5e-5
GRADIENT_CLIP_VAL = 1.0
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query", "key", "value"]
SEED = 42
NUM_DL_WORKERS = 0

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == 'mps': NUM_DL_WORKERS = 0

class AntibodyDataset(Dataset):
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
        heavy_raw = row["VH"]
        light_raw = row["VL"]
        label = row["scaled_label"]

        heavy_spaced = " ".join(list(heavy_raw))
        light_spaced = " ".join(list(light_raw))

        heavy_inputs = self.tokenizer(
            heavy_spaced, truncation=True, max_length=self.max_length,
            return_tensors="pt", padding=False
        )
        light_inputs = self.tokenizer(
            light_spaced, truncation=True, max_length=self.max_length,
            return_tensors="pt", padding=False
        )

        return {
            "heavy_input_ids": heavy_inputs["input_ids"].squeeze(0),
            "heavy_attention_mask": heavy_inputs["attention_mask"].squeeze(0),
            "light_input_ids": light_inputs["input_ids"].squeeze(0),
            "light_attention_mask": light_inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float)
        }

tokenizer_global_for_collate = None

def collate_fn_dynamic(batch):
    if tokenizer_global_for_collate is None:
        raise ValueError("Global tokenizer for collate_fn is not set.")

    heavy_ids = [item["heavy_input_ids"] for item in batch]
    heavy_masks = [item["heavy_attention_mask"] for item in batch]
    light_ids = [item["light_input_ids"] for item in batch]
    light_masks = [item["light_attention_mask"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    heavy_ids_padded = pad_sequence(heavy_ids, batch_first=True, padding_value=tokenizer_global_for_collate.pad_token_id)
    heavy_masks_padded = pad_sequence(heavy_masks, batch_first=True, padding_value=0)
    light_ids_padded = pad_sequence(light_ids, batch_first=True, padding_value=tokenizer_global_for_collate.pad_token_id)
    light_masks_padded = pad_sequence(light_masks, batch_first=True, padding_value=0)

    return {
        "heavy_input_ids": heavy_ids_padded.to(DEVICE),
        "heavy_attention_mask": heavy_masks_padded.to(DEVICE),
        "light_input_ids": light_ids_padded.to(DEVICE),
        "light_attention_mask": light_masks_padded.to(DEVICE),
        "label": labels.to(DEVICE)
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
    def __init__(self, plm_model, hidden_dim, head_type="simple", dropout=0.1, use_attention_pool=True):
        super().__init__()
        self.plm_model = plm_model
        self.use_attention_pool = use_attention_pool
        if self.use_attention_pool:
            self.pooler = AttentionPooling(hidden_dim)
        else:
            self.pooler = None

        input_dim_for_head = 2 * hidden_dim
        head_map = {
            "simple": SimpleHead(input_dim_for_head),
            "medium": MediumHead(input_dim_for_head, dropout),
            "deep": DeepHead(input_dim_for_head, dropout)
        }
        if head_type not in head_map: raise ValueError(f"Invalid head type: {head_type}")
        self.head = head_map[head_type]

    def _mean_pool(self, hidden_state, attention_mask):
        mask_f = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden_state * mask_f).sum(dim=1)
        len_hidden = mask_f.sum(dim=1).clamp(min=1e-9)
        return sum_hidden / len_hidden

    def forward(self, heavy_ids, heavy_mask, light_ids, light_mask):
        heavy_out = self.plm_model(input_ids=heavy_ids, attention_mask=heavy_mask)
        light_out = self.plm_model(input_ids=light_ids, attention_mask=light_mask)
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
        self.best_val_metric = -float('inf')
        self.metric_criterion = "spearman"
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None
        self.best_val_loss_for_stopping = float('inf')


    def step(self, current_metric_val, current_val_loss, model):
        improved = False
        if self.metric_criterion == "spearman":
            if current_metric_val > self.best_val_metric + self.min_delta :
                self.best_val_metric = current_metric_val
                self.best_model_state = model.state_dict()
                improved = True
        elif self.metric_criterion == "loss":
             if current_metric_val < self.best_val_metric - self.min_delta:
                self.best_val_metric = current_metric_val
                self.best_model_state = model.state_dict()
                improved = True
        else:
            raise ValueError("metric_criterion must be 'spearman' or 'loss'")

        if current_val_loss < self.best_val_loss_for_stopping - self.min_delta:
            self.best_val_loss_for_stopping = current_val_loss
            self.counter = 0
        else:
            if not improved :
                 self.counter +=1

        if self.counter >= self.patience:
            self.early_stop = True
        return improved


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

def train_one_fold(model, train_loader, val_loader, fold_num, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, grad_clip_value=GRADIENT_CLIP_VAL):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    early_stopper = EarlyStopping(patience=5, min_delta=0.001)
    early_stopper.metric_criterion = "spearman"


    print(f"Starting Fold {fold_num} training for {num_epochs} epochs...")
    print_trainable_parameters(model)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Fold {fold_num} Epoch {epoch+1} Train", leave=False):
            optimizer.zero_grad()
            preds = model(batch["heavy_input_ids"], batch["heavy_attention_mask"],
                          batch["light_input_ids"], batch["light_attention_mask"])
            loss = criterion(preds, batch["label"])
            loss.backward()
            if grad_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_clip_value)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_losses, all_preds_val, all_labels_val = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch["label"]
                preds = model(batch["heavy_input_ids"], batch["heavy_attention_mask"],
                              batch["light_input_ids"], batch["light_attention_mask"])
                loss = criterion(preds, labels)
                val_losses.append(loss.item())
                all_preds_val.extend(preds.cpu().numpy())
                all_labels_val.extend(labels.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        spearman_corr_val = np.nan
        if len(all_preds_val) >= 2 and np.var(all_preds_val) > 1e-9 and np.var(all_labels_val) > 1e-9:
            spearman_corr_val, _ = spearmanr(all_preds_val, all_labels_val)
        
        print(f"[Fold {fold_num} Ep {epoch+1:02d}] Tr Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Spearman: {spearman_corr_val:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        improved_metric = early_stopper.step(spearman_corr_val, avg_val_loss, model)
        if improved_metric:
             print(f"  -> New best Spearman for fold: {early_stopper.best_val_metric:.4f}. Model state saved.")


        if early_stopper.early_stop:
            print(f"Early stopping triggered at Fold {fold_num} Epoch {epoch+1} due to validation loss non-improvement!")
            break
            
    if early_stopper.best_model_state:
        print(f"Loading best model state for Fold {fold_num} (Best Spearman: {early_stopper.best_val_metric:.4f}, Val Loss for stopping: {early_stopper.best_val_loss_for_stopping:.4f}).")
        model.load_state_dict(early_stopper.best_model_state)
    else:
        print(f"Warning: No best model state recorded during training for Fold {fold_num}. Using last state.")
    return model

df1 = pd.read_excel('pnas.1616408114.sd01.xlsx')
df2 = pd.read_excel('pnas.1616408114.sd02.xlsx') 
df3 = pd.read_excel('pnas.1616408114.sd03.xlsx')

merged_df = df1.merge(df2, on='Name', how='outer').merge(df3, on='Name', how='outer')
             
df = merged_df[['VH', 'VL', 'Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average']].copy()
df = df.rename(columns={'Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average': 'agg'})

try:
    tokenizer_igbert = AutoTokenizer.from_pretrained(IGBERT_MODEL_NAME)
    tokenizer_global_for_collate = tokenizer_igbert
except Exception as e:
    print(f"Error loading IgBERT tokenizer: {e}")
    exit()

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
)

kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
fold_results = []
trained_peft_models_paths = []

train_val_data_for_kfold = df.reset_index(drop=True)

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

    train_dataset = AntibodyDataset(fold_train_df, tokenizer_igbert)
    val_dataset = AntibodyDataset(fold_val_df, tokenizer_igbert)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_dynamic, num_workers=NUM_DL_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_dynamic, num_workers=NUM_DL_WORKERS)

    base_igbert_model_fold = AutoModel.from_pretrained(IGBERT_MODEL_NAME)
    igbert_hidden_dim_fold = base_igbert_model_fold.config.hidden_size

    peft_igbert_model_fold = get_peft_model(base_igbert_model_fold, lora_config)
    print(f"PEFT IgBERT model created for Fold {current_fold_num_cv}.")
    peft_igbert_model_fold.print_trainable_parameters()


    model_fold = RegressionModelWithLoRA(
        plm_model=peft_igbert_model_fold,
        hidden_dim=igbert_hidden_dim_fold,
        head_type=HEAD_TYPE,
        dropout=0.1,
        use_attention_pool=True
    ).to(DEVICE)

    trained_model_fold = train_one_fold(
        model=model_fold, train_loader=train_loader, val_loader=val_loader,
        fold_num=current_fold_num_cv, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, grad_clip_value=GRADIENT_CLIP_VAL
    )

    peft_model_save_path = f"./igbert_lora_fold_{current_fold_num_cv}"
    trained_model_fold.plm_model.save_pretrained(peft_model_save_path)
    torch.save({
        'head_state_dict': trained_model_fold.head.state_dict(),
        'pooler_state_dict': trained_model_fold.pooler.state_dict() if trained_model_fold.pooler else None
    }, f"{peft_model_save_path}/custom_head_pooler.pth")
    trained_peft_models_paths.append(peft_model_save_path)
    print(f"Saved PEFT model for Fold {current_fold_num_cv} to {peft_model_save_path}")


    trained_model_fold.eval()
    val_preds_final_fold, val_labels_final_fold, val_losses_final_fold = [], [], []
    criterion_eval = nn.MSELoss()
    with torch.no_grad():
        for batch in val_loader:
            labels = batch["label"]
            preds = trained_model_fold(batch["heavy_input_ids"], batch["heavy_attention_mask"],
                                       batch["light_input_ids"], batch["light_attention_mask"])
            loss = criterion_eval(preds, labels)
            val_losses_final_fold.append(loss.item())
            val_preds_final_fold.extend(preds.cpu().numpy())
            val_labels_final_fold.extend(labels.cpu().numpy())

    avg_val_loss_fold = np.mean(val_losses_final_fold) if val_losses_final_fold else np.nan
    val_spearman_fold = np.nan
    if len(val_preds_final_fold) >= 2 and np.var(val_preds_final_fold) > 1e-9 and np.var(val_labels_final_fold) > 1e-9:
        val_spearman_fold, _ = spearmanr(val_preds_final_fold, val_labels_final_fold)

    fold_results.append({'loss': avg_val_loss_fold, 'spearman': val_spearman_fold})
    print(f"---- Fold {current_fold_num_cv} Final Validation (Scaled Labels) ----")
    print(f"Loss: {avg_val_loss_fold:.4f}, Spearman: {val_spearman_fold:.4f}")

    del model_fold, base_igbert_model_fold, peft_igbert_model_fold, trained_model_fold
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    elif DEVICE.type == 'mps': torch.mps.empty_cache()

if fold_results:
    avg_cv_loss = np.nanmean([res['loss'] for res in fold_results])
    std_cv_loss = np.nanstd([res['loss'] for res in fold_results if not np.isnan(res['loss'])])
    avg_cv_spearman = np.nanmean([res['spearman'] for res in fold_results])
    std_cv_spearman = np.nanstd([res['spearman'] for res in fold_results if not np.isnan(res['spearman'])])

    print("\n--- IgBERT LoRA Fine-tuning Cross-Validation Summary (on Scaled Data) ---")
    print(f"Average Validation Loss:     {avg_cv_loss:.4f} +/- {std_cv_loss:.4f}")
    print(f"Average Validation Spearman: {avg_cv_spearman:.4f} +/- {std_cv_spearman:.4f}")
    print(f"Trained PEFT model paths: {trained_peft_models_paths}")
else:
    print("No fold results to summarize.")


