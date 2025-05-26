# CIL

current ensemble code

```python

# --- START OF FILE train_log_ensemble_v4_dptloadfix.py ---
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForDepthEstimation, AutoImageProcessor, DPTImageProcessor, DPTForDepthEstimation
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import OrderedDict
import random




# --- Configuration (USING YOUR PROVIDED VALUES) ---
BASE_DATA_DIR = '/home/s/z/cil-project-ss25/data'
TRAIN_LIST_FILE = "/home/s/z/cil-project-ss25/data/train_list.txt"

TRAIN_RGB_AND_DEPTH_DIR = os.path.join(BASE_DATA_DIR, 'train/train')
TRAIN_RGB_AND_GT_ROOT_DIR = TRAIN_RGB_AND_DEPTH_DIR
MODEL1_WEIGHTS_PATH = '/home/s/z/cil-project-ss25/ensemble/valery/best_depth_anything_model.pth'
MODEL2_WEIGHTS_PATH = '/home/s/z/cil-project-ss25/ensemble/soto/soto_model.pth'

ENSEMBLE_OUTPUT_DIR = './working_log_ensemble_v4_dptloadfix' # New output
ENSEMBLE_RESULTS_DIR = os.path.join(ENSEMBLE_OUTPUT_DIR, 'results')
ENSEMBLE_MODEL_SAVE_PATH = os.path.join(ENSEMBLE_RESULTS_DIR, 'best_log_meta_learner.pth')
PRECOMPUTED_LOG_DEPTHS_DIR = os.path.join(ENSEMBLE_OUTPUT_DIR, 'precomputed_log_depths')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

OPERATING_RESOLUTION = (426, 560)
META_BATCH_SIZE = 4
META_LEARNING_RATE = 5e-5
META_WEIGHT_DECAY = 1e-4
META_NUM_EPOCHS = 50
META_VAL_SPLIT_FRAC = 0.15
NUM_META_WORKERS = 2
PIN_MEMORY_META = (DEVICE.type == 'cuda' and NUM_META_WORKERS > 0)
GRAD_CLIP_NORM = 1.0
LOG_MLP_HIDDEN_SIZE = 16 # From your training script that had the error

MODEL1_HF_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL2_HF_ID = "Intel/dpt-large"
MODEL2_OUTPUTS_LOG_DEPTH = True

EPS_FOR_LOG = 1e-6
FORCE_PRECOMPUTE_LOG_DEPTHS = True
# --- End Configuration ---

def ensure_dir(directory):
    if not os.path.exists(directory): os.makedirs(directory)

def si_loss_fn(log_outputs,log_targets, num_pixels):
    R = log_outputs - log_targets
    first_term = torch.pow(R,2).sum(dim=(1,2,3))/num_pixels
    sec_term = torch.pow(R.sum(dim=(1,2,3)),2)/(num_pixels**2)
    loss = first_term - sec_term
    return loss.mean()

def gradient_matching_loss_fn(log_outputs, log_targets, num_pixels):
    R = log_outputs - log_targets
    grad_x = R[:,:,:,1:] - R[:,:,:,:-1]
    grad_y = R[:,:,1:,:] - R[:,:,:-1,:]
    loss = (torch.abs(grad_x).sum(dim=(1,2,3)) + torch.abs(grad_y).sum(dim=(1,2,3))) / num_pixels
    return loss.mean()

def meta_criterion(log_outputs, log_targets):
    num_pixels_per_sample = log_outputs.shape[2] * log_outputs.shape[3]
    return si_loss_fn(log_outputs,log_targets, num_pixels_per_sample) + \
           0.5 * gradient_matching_loss_fn(log_outputs,log_targets, num_pixels_per_sample)

class DepthAnythingWrapper(nn.Module):
    def __init__(self, model_id=MODEL1_HF_ID):
        super(DepthAnythingWrapper, self).__init__()
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.predicted_depth

class DPTDepthEstimationHead(nn.Module):
    def __init__(self, config, log_space=False):
        super().__init__()
        self.config = config
        self.log_space = log_space
        self.projection = None
        if getattr(self.config, 'add_projection', False):
            self.projection = nn.Conv2d(
                getattr(self.config, 'encoder_hidden_size', 256),
                getattr(self.config, 'encoder_hidden_size', 256),
                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            )
        features = getattr(self.config, 'fusion_hidden_size', 256)
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU() if not log_space else nn.Identity(),
        )
    def forward(self, hidden_states_tuple):
        head_in_index = getattr(self.config, 'head_in_index', -1)
        hidden_states = hidden_states_tuple[head_in_index]
        if self.projection is not None:
            hidden_states = self.projection(hidden_states)
            hidden_states = nn.ReLU()(hidden_states)
        predicted_depth = self.head(hidden_states)
        return predicted_depth

class DPTDepthEstimator(nn.Module): # CORRECTED
    def __init__(self,hub_checkpoint=MODEL2_HF_ID, log_space=MODEL2_OUTPUTS_LOG_DEPTH):
        super().__init__()
        # Renamed self.model_hf to self.model
        self.model = DPTForDepthEstimation.from_pretrained(hub_checkpoint) 
        
        config = self.model.config # Use self.model.config
        if not hasattr(config, 'head_in_index'): config.head_in_index = -1
        if not hasattr(config, 'fusion_hidden_size'): config.fusion_hidden_size = 256
        if not hasattr(config, 'add_projection'): config.add_projection = False
        
        custom_head = DPTDepthEstimationHead(config, log_space=log_space)
        self.model.head = custom_head # Replace head of self.model

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values) # Call self.model
        predicted_depth = outputs.predicted_depth
        if predicted_depth.ndim == 3: predicted_depth = predicted_depth.unsqueeze(1)
        elif predicted_depth.ndim == 2: predicted_depth = predicted_depth.unsqueeze(0).unsqueeze(0)
        if predicted_depth.ndim == 5 and predicted_depth.shape[1] == 1 and predicted_depth.shape[2] == 1:
            predicted_depth = predicted_depth.squeeze(2)
        return predicted_depth

class LogSpaceWeightedAverageCombiner(nn.Module):
    def __init__(self, in_channels=2, hidden_mlp_size=LOG_MLP_HIDDEN_SIZE, out_channels=1):
        super(LogSpaceWeightedAverageCombiner, self).__init__(); self.mlp_layer1 = nn.Conv2d(in_channels, hidden_mlp_size, kernel_size=1, padding=0); self.relu = nn.ReLU(); self.mlp_layer2 = nn.Conv2d(hidden_mlp_size, hidden_mlp_size, kernel_size=1, padding=0); self.weight_m1_head = nn.Conv2d(hidden_mlp_size, 1, kernel_size=1, padding=0); self.sigmoid = nn.Sigmoid(); self.bias_head = nn.Conv2d(hidden_mlp_size, out_channels, kernel_size=1, padding=0)
    def forward(self, log_depth_m1, log_depth_m2):
        x = torch.cat((log_depth_m1, log_depth_m2), dim=1); hidden = self.relu(self.mlp_layer1(x)); hidden = self.relu(self.mlp_layer2(hidden)); w1_log = self.sigmoid(self.weight_m1_head(hidden)); w2_log = 1.0 - w1_log; combined_log_depth = w1_log * log_depth_m1 + w2_log * log_depth_m2; bias_log = self.bias_head(hidden); final_log_depth = combined_log_depth + bias_log; return final_log_depth

def precompute_base_log_depths(file_pairs_all, base_rgb_dir, model1, model2, m1_processor, m2_processor, target_resolution, device, cache_dir_base, force_recompute):
    print(f"Starting precomputation of base model LOG-DEPTHS for target_size: {target_resolution}...")
    m1_log_cache_dir = os.path.join(cache_dir_base, "model1_log_depth"); m2_log_cache_dir = os.path.join(cache_dir_base, "model2_log_depth")
    ensure_dir(m1_log_cache_dir); ensure_dir(m2_log_cache_dir)
    model1.to(device).eval(); model2.to(device).eval(); processed_pairs = []
    if force_recompute:
        for d in [m1_log_cache_dir, m2_log_cache_dir]:
            if os.path.exists(d): [os.remove(os.path.join(d,f)) for f in os.listdir(d) if f.endswith(".npy")]
        print("Cleared old precomputed log-depths.")
    for rgb_file_path_in_list, depth_gt_file_path_in_list in tqdm(file_pairs_all, desc="Precomputing log-depths"):
        actual_rgb_path = os.path.join(base_rgb_dir, rgb_file_path_in_list); base_name_for_cache = os.path.splitext(os.path.basename(rgb_file_path_in_list))[0]
        m1_log_cache_path = os.path.join(m1_log_cache_dir, f"{base_name_for_cache}_log_m1.npy"); m2_log_cache_path = os.path.join(m2_log_cache_dir, f"{base_name_for_cache}_log_m2.npy")
        if not force_recompute and os.path.exists(m1_log_cache_path) and os.path.exists(m2_log_cache_path):
            processed_pairs.append((rgb_file_path_in_list, depth_gt_file_path_in_list)); continue
        try: rgb_pil = Image.open(actual_rgb_path).convert('RGB')
        except Exception as e: print(f"Skipping {actual_rgb_path} due to load error: {e}"); continue
        with torch.no_grad():
            m1_pixel_values = m1_processor(images=rgb_pil, return_tensors="pt").pixel_values.to(device); m1_da_output_direct_depth = model1(m1_pixel_values)
            m1_direct_depth_processed = 10.0 - m1_da_output_direct_depth
            if m1_direct_depth_processed.ndim == 2: m1_direct_depth_processed = m1_direct_depth_processed.unsqueeze(0).unsqueeze(0)
            elif m1_direct_depth_processed.ndim == 3: m1_direct_depth_processed = m1_direct_depth_processed.unsqueeze(1)
            m1_direct_depth_resized = nn.functional.interpolate(m1_direct_depth_processed, size=target_resolution, mode='bilinear', align_corners=True)
            m1_log_depth = torch.log(torch.clamp(m1_direct_depth_resized, min=EPS_FOR_LOG)); np.save(m1_log_cache_path, m1_log_depth.squeeze().cpu().numpy())
            m2_pixel_values = m2_processor(images=rgb_pil, return_tensors="pt").pixel_values.to(device); m2_log_depth_raw = model2(m2_pixel_values)
            m2_log_depth_resized = nn.functional.interpolate(m2_log_depth_raw, size=target_resolution, mode='bilinear', align_corners=True)
            np.save(m2_log_cache_path, m2_log_depth_resized.squeeze().cpu().numpy())
        processed_pairs.append((rgb_file_path_in_list, depth_gt_file_path_in_list))
    print(f"Log-depth precomputation finished. {len(processed_pairs)} pairs processed."); return processed_pairs

class LogDepthEnsembleDataset(Dataset):
    def __init__(self, file_pairs_processed, gt_root_dir, precomputed_log_depth_cache_dir, target_resolution):
        self.file_pairs = file_pairs_processed; self.gt_root_dir = gt_root_dir; self.m1_log_cache_dir = os.path.join(precomputed_log_depth_cache_dir, "model1_log_depth"); self.m2_log_cache_dir = os.path.join(precomputed_log_depth_cache_dir, "model2_log_depth"); self.target_resolution = target_resolution; self.eps = EPS_FOR_LOG
    def __len__(self): return len(self.file_pairs)
    def __getitem__(self, idx):
        rgb_file_path_in_list, gt_depth_file_path_in_list = self.file_pairs[idx]; base_name_for_cache = os.path.splitext(os.path.basename(rgb_file_path_in_list))[0]
        m1_log_path = os.path.join(self.m1_log_cache_dir, f"{base_name_for_cache}_log_m1.npy"); m2_log_path = os.path.join(self.m2_log_cache_dir, f"{base_name_for_cache}_log_m2.npy"); actual_gt_depth_path = os.path.join(self.gt_root_dir, gt_depth_file_path_in_list)
        try:
            log_m1 = torch.from_numpy(np.load(m1_log_path)).float(); log_m2 = torch.from_numpy(np.load(m2_log_path)).float(); gt_depth_direct = torch.from_numpy(np.load(actual_gt_depth_path).astype(np.float32))
            if log_m1.ndim == 2: log_m1 = log_m1.unsqueeze(0)
            if log_m2.ndim == 2: log_m2 = log_m2.unsqueeze(0)
            if gt_depth_direct.ndim == 2: gt_depth_direct = gt_depth_direct.unsqueeze(0)
            gt_depth_direct_resized = nn.functional.interpolate(gt_depth_direct.unsqueeze(0), size=self.target_resolution, mode='bilinear', align_corners=True).squeeze(0)
            log_gt = torch.log(torch.clamp(gt_depth_direct_resized, min=self.eps))
            if random.random() > 0.5: log_m1=TF.hflip(log_m1); log_m2=TF.hflip(log_m2); log_gt=TF.hflip(log_gt)
            if random.random() > 0.5: angle = random.uniform(-5.0, 5.0); log_m1=TF.rotate(log_m1,angle,TF.InterpolationMode.NEAREST); log_m2=TF.rotate(log_m2,angle,TF.InterpolationMode.NEAREST); log_gt=TF.rotate(log_gt,angle,TF.InterpolationMode.NEAREST)
            return (log_m1, log_m2), log_gt
        except Exception as e: print(f"Error in LogDepthEnsembleDataset idx {idx} for {rgb_file_path_in_list}: {e}. Returning zeros."); dummy_log_depth = torch.zeros(1, *self.target_resolution); return (dummy_log_depth, dummy_log_depth), dummy_log_depth

def train_log_meta_learner(meta_model, train_loader, val_loader, criterion_fn, optimizer, scheduler, num_epochs, device, save_path, grad_clip_norm):
    best_val_loss = float('inf'); train_losses_hist, val_losses_hist = [], []; first_batch_stats_printed = False
    for epoch in range(num_epochs):
        meta_model.train(); running_train_loss = 0.0
        for i, ((log_m1, log_m2), log_gt) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")):
            log_m1, log_m2, log_gt = log_m1.to(device), log_m2.to(device), log_gt.to(device)
            if not first_batch_stats_printed and i == 0:
                print("\n--- Stats for first training batch inputs (log-depths): ---"); print(f"Log_M1 (shape, min/max/mean): {log_m1[0].shape}, {log_m1[0].min():.2f}/{log_m1[0].max():.2f}/{log_m1[0].mean():.2f}"); print(f"Log_M2 (shape, min/max/mean): {log_m2[0].shape}, {log_m2[0].min():.2f}/{log_m2[0].max():.2f}/{log_m2[0].mean():.2f}"); print(f"Log_GT (shape, min/max/mean): {log_gt[0].shape}, {log_gt[0].min():.2f}/{log_gt[0].max():.2f}/{log_gt[0].mean():.2f}"); print("------------------------------------------------------------\n"); first_batch_stats_printed = True
            optimizer.zero_grad(); combined_log_depth_pred = meta_model(log_m1, log_m2); loss = criterion_fn(combined_log_depth_pred, log_gt); loss.backward();
            if grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(meta_model.parameters(), grad_clip_norm)
            optimizer.step(); running_train_loss += loss.item() * log_m1.size(0)
        epoch_train_loss = running_train_loss / len(train_loader.dataset); train_losses_hist.append(epoch_train_loss)
        meta_model.eval(); running_val_loss = 0.0
        with torch.no_grad():
            for (log_m1_val, log_m2_val), log_gt_val in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                log_m1_val,log_m2_val,log_gt_val = log_m1_val.to(device),log_m2_val.to(device),log_gt_val.to(device)
                combined_log_depth_pred_val = meta_model(log_m1_val, log_m2_val); loss = criterion_fn(combined_log_depth_pred_val, log_gt_val); running_val_loss += loss.item() * log_m1_val.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset); val_losses_hist.append(epoch_val_loss)
        print(f"Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}"); scheduler.step(epoch_val_loss)
        if epoch_val_loss < best_val_loss: best_val_loss = epoch_val_loss; torch.save(meta_model.state_dict(), save_path); print(f"Best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
        plt.figure(figsize=(10,5)); plt.plot(train_losses_hist,label='Train Loss'); plt.plot(val_losses_hist,label='Val Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss (Meta Criterion on Log Depths)'); plt.title('Log-Space Meta-Learner Training Loss'); plt.legend(); plt.grid(True); plt.savefig(os.path.join(ENSEMBLE_RESULTS_DIR, 'log_meta_learner_loss.png')); plt.close()
    print("Log-space meta-learner training finished."); return meta_model

def main():
    ensure_dir(ENSEMBLE_OUTPUT_DIR); ensure_dir(ENSEMBLE_RESULTS_DIR); ensure_dir(PRECOMPUTED_LOG_DEPTHS_DIR)
    print("Loading base models for precomputation...")
    m1_processor = AutoImageProcessor.from_pretrained(MODEL1_HF_ID)
    base_model1 = DepthAnythingWrapper(model_id=MODEL1_HF_ID)
    if not os.path.exists(MODEL1_WEIGHTS_PATH): print(f"M1 weights not found: {MODEL1_WEIGHTS_PATH}"); sys.exit(1)
    base_model1.load_state_dict(torch.load(MODEL1_WEIGHTS_PATH, map_location='cpu'))
    
    m2_processor = DPTImageProcessor.from_pretrained(MODEL2_HF_ID)
    base_model2 = DPTDepthEstimator(log_space=MODEL2_OUTPUTS_LOG_DEPTH) # This now uses the corrected DPTDepthEstimator
    if not os.path.exists(MODEL2_WEIGHTS_PATH): print(f"M2 weights not found: {MODEL2_WEIGHTS_PATH}"); sys.exit(1)
    state_dict_m2 = torch.load(MODEL2_WEIGHTS_PATH, map_location='cpu')
    
    # Corrected loading logic for base_model2
    if next(iter(state_dict_m2)).startswith('module.'): # If soto_model.pth was saved from DataParallel
        # Create a new state_dict with 'module.' prefix removed from keys in soto_model.pth
        # AND ensure keys in soto_model.pth match base_model2 (which now has self.model internally)
        
        # Keys in soto_model.pth are like "module.model.dpt..."
        # Keys in base_model2 are like "model.dpt..."
        
        # First, remove 'module.' from soto_model.pth keys
        soto_keys_no_module = OrderedDict()
        for k, v in state_dict_m2.items():
            if k.startswith('module.'):
                soto_keys_no_module[k[len('module.'):]] = v
            else:
                soto_keys_no_module[k] = v # Should not happen if saved with DataParallel
        base_model2.load_state_dict(soto_keys_no_module)

    elif next(iter(state_dict_m2)).startswith('model.'): # If soto_model.pth keys start with "model."
        base_model2.load_state_dict(state_dict_m2) # Direct load should work if self.model is used in DPTDepthEstimator
    else:
        # This case implies soto_model.pth has keys directly matching DPTForDepthEstimation,
        # so we'd load into base_model2.model
        try:
            base_model2.model.load_state_dict(state_dict_m2) 
            print("Loaded soto_model.pth directly into base_model2.model")
        except RuntimeError as e:
            print(f"Could not directly load soto_model.pth into base_model2.model. Error: {e}")
            print("This might indicate soto_model.pth was saved from DPTForDepthEstimation directly, not the DPTDepthEstimator wrapper.")
            print("Or, the keys in soto_model.pth do not match the expected structure (e.g. 'model.dpt...').")
            sys.exit(1)
            
    print("Base models loaded.")
    with open(TRAIN_LIST_FILE, 'r') as f: all_train_val_file_pairs = [line.strip().split() for line in f if len(line.strip().split()) == 2]
    
    processed_log_depth_pairs = precompute_base_log_depths(
        all_train_val_file_pairs, TRAIN_RGB_AND_GT_ROOT_DIR,
        base_model1, base_model2, m1_processor, m2_processor,
        OPERATING_RESOLUTION, DEVICE, PRECOMPUTED_LOG_DEPTHS_DIR, FORCE_PRECOMPUTE_LOG_DEPTHS
    )
    if not processed_log_depth_pairs: print("Precomputation of log-depths failed."); sys.exit(1)
    if FORCE_PRECOMPUTE_LOG_DEPTHS: print(f"IMPORTANT: Log-depth precomputation was forced. Set FORCE_PRECOMPUTE_LOG_DEPTHS = False for future runs.")

    random.seed(42); random.shuffle(processed_log_depth_pairs)
    val_split_index = int(len(processed_log_depth_pairs) * (1 - META_VAL_SPLIT_FRAC))
    meta_train_pairs = processed_log_depth_pairs[:val_split_index]; meta_val_pairs = processed_log_depth_pairs[val_split_index:]
    print(f"Log-meta train size: {len(meta_train_pairs)}, Log-meta val size: {len(meta_val_pairs)}")

    train_log_dataset = LogDepthEnsembleDataset(meta_train_pairs, TRAIN_RGB_AND_GT_ROOT_DIR, PRECOMPUTED_LOG_DEPTHS_DIR, OPERATING_RESOLUTION)
    val_log_dataset = LogDepthEnsembleDataset(meta_val_pairs, TRAIN_RGB_AND_GT_ROOT_DIR, PRECOMPUTED_LOG_DEPTHS_DIR, OPERATING_RESOLUTION)
    
    train_log_loader = DataLoader(train_log_dataset, batch_size=META_BATCH_SIZE, shuffle=True, num_workers=NUM_META_WORKERS, pin_memory=PIN_MEMORY_META, drop_last=True)
    val_log_loader = DataLoader(val_log_dataset, batch_size=META_BATCH_SIZE, shuffle=False, num_workers=NUM_META_WORKERS, pin_memory=PIN_MEMORY_META)
    print("Initializing LogSpaceWeightedAverageCombiner...");
    log_meta_learner = LogSpaceWeightedAverageCombiner(hidden_mlp_size=LOG_MLP_HIDDEN_SIZE).to(DEVICE) # LOG_MLP_HIDDEN_SIZE used here
    optimizer = optim.AdamW(log_meta_learner.parameters(), lr=META_LEARNING_RATE, weight_decay=META_WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True, min_lr=1e-7)
    print("Starting Log-Space Meta-Learner training...")
    train_log_meta_learner(log_meta_learner, train_log_loader, val_log_loader, meta_criterion, optimizer, scheduler, META_NUM_EPOCHS, DEVICE, ENSEMBLE_MODEL_SAVE_PATH, GRAD_CLIP_NORM)
    print(f"Log-space meta-learner training complete. Best model saved to {ENSEMBLE_MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()
# --- END OF FILE train_log_ensemble_v4_dptloadfix.py ---

```


Example of function that is continuous but not uniformly continuous : 1/x


mu-strong convexity IMPLIES PL CONDITION

BUT

inverse is not true : counterexample : f(x) = 1/4(x^2 - 1)^2

# Philosophy notes

## Deleuze’s The Logic of Sense (1969)

- Denotation : How a proposition points to a specific **state of affairs** : matching words to things using "this"
- Manifestation : The speaker's role in a proposition
- Signification : The logical and conceptual structure of a proposition : how propositions relate to concepts and **imply other concepts**
- Sense : what a proposition expresses. Sense does not exists independently. Sense is what makes language meaningful


LDA (Latent Dirichlet Allocation)

Example of documents  : every doc is a mix of diff topics

Every topic is characterized by a distribution of words. The "Politics" topic is likely to contain words like "election," "vote," "candidate," "party," "government" frequently. The "Economics" topic might have "market," "stock," "trade," "inflation," "GDP."

LDA is a generative model : it assumes a hypothetical process for how the documents you see could have been generated. 

So we have to imagine we want to write a new document

We pick the proportions of topics : the mix is influenced by the Dirichlet distribution
For each word in the document we first pick  which topic it will cover. The probability of each topic is based on the mix we choose at step 1.
We then pick a word based on the topic we choose. This time the probability of the word we choose is based on the distribution of words for that topic
Repeat that process for each word we want to have in the document

LDA doesn't actually generate documents. It observes the finished documents (just the sequence of words) and tries to work backward to figure out the hidden structure:
Most probable mix of topics
Most probable word distribution for each topic
Which topic likely generated each specific word occurrence in each document?$


Initialization

The algorithm starts with a random (or semi-random) assignment of topics to each word instance in the corpus. The initial topics are meaningless garbage.

Then iteration by iteration, the algorithm looks at one word at a time and for each of them decide which topic should have generated it

-> how well does this word fit with each topic’s current word distribution:
 Based on al the others words currently assigned to Topic X across the entire corpus, how likely is “gene” to belong to topic X ?
- > Based on all the other words in this specific document, what’s the current estimate of its topic mixture

Using the response from those 2 questions, we re assign the word to a topic
-> if “GENE” frequently appears in documents that also contain “DNA” and “sequence”, words assigned to topic 3, then “gene” becomes more likely to be assigned to topic 3.
We consider both GLOBAL TOPIC DEFINITION and LOCAL DOCUMENT CONTEXT

The algorithm converges when the topic assignment stabilize

UNSUPERVISED


-> hyperparameter K, the number of topics we want to extract

-> The topics emerge purely from the statistical patterns of word co-occurrence in the documents.



# Projection matrix : self-adjointness property

![2025-03-10-095524_650x43_scrot](https://github.com/user-attachments/assets/7a435f4b-63eb-46e5-b530-1904ce8d6892)


![2025-03-10-094909_663x131_scrot](https://github.com/user-attachments/assets/917d5d90-c090-40b4-8626-4c5d1f8bde3f)


# CGAL

```
K::Point_2 a;
K::Point_2 b;

CGAL::right_turn(a, b, c) // true if c makes a right turn relative to the directed line segment from a to b 
```

# AML

# Conditional entropy

$H(X \mid Y) = - \sum_{y \in \mathcal{Y}} \sum_{x \in \mathcal{X}} P(Y = y) P(X = x \mid Y = y) \log P(X = x \mid Y = y)$

$= - \sum_{x \in \mathcal{X}, y \in \mathcal{Y}} P(X = x, Y = y) \log \frac{P(X = x, Y = y)}{P(Y = y)},$

# Mutual information

$I(X; Y) := H(X) - H(X \mid Y)$

Measures the amount of information of X left after Y is revealed.

![2025-01-03-184515_818x62_scrot](https://github.com/user-attachments/assets/88b623b4-0a0e-4ad9-a714-ddecd518e070)

# Matrix calculus

We have

![2025-01-03-180910_189x110_scrot](https://github.com/user-attachments/assets/289a2781-670c-48c2-8fff-409102f5cc00)

# Recursion pro tip : if there are some overlap, just substract it !

![2024-12-16-151305_1733x397_scrot](https://github.com/user-attachments/assets/8532118d-b2fc-452b-bc04-ca0a15491991)

