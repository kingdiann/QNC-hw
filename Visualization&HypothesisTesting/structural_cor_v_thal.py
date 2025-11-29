#%% ##SCATTERPLOTS FOR ROI VALUES VS AGE - SPLIT THALAMIC NUCELI TOP10##
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

measure = "thickness"
hemi = "lh"

# --- Load data ---
GMmeasure_df = pd.read_csv(f"/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/cortical/7schaefer_{measure}_stats.csv")
age_df = pd.read_csv("/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/subject_ages.csv")

# Load JSON file
with open(f"/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/functional_measures/top10_parcels_{hemi}.json", "r") as f:
    parcel_dict = json.load(f)

# --- Merge age and measure ---
merged = GMmeasure_df.merge(age_df, on="subject_id", how="inner")
print(f"Merged dataset has {len(merged)} subjects")

# --- Define thalamic nuclei ---
if hemi == 'rh':
    thal_nuc = [2] + list(range(4, 17))
else: 
    thal_nuc = [16] + list(range(18, 29))

# --- Create 12 plots (one per nucleus) ---
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for idx, nuc_id in enumerate(thal_nuc):
    ax = axes[idx]
    nucleus_key = f"Nucleus_{nuc_id}"
    
    # Get parcels for this nucleus
    if nucleus_key not in parcel_dict:
        print(f"Warning: {nucleus_key} not found in JSON")
        continue
    
    parcels_for_nucleus = parcel_dict[nucleus_key]
    
    print(f"\n{nucleus_key}: Looking for {len(parcels_for_nucleus)} parcels")
    
    # Filter to parcels that exist in the data using flexible matching
    valid_parcels = []
    for parcel in parcels_for_nucleus:
        # Split the parcel name into parts
        parts = parcel.split('_')
        
        # Create a regex pattern: match parts 0-2 and the last part (number)
        # Pattern: lh_7Networks_LH_Default*_9
        if len(parts) >= 4:
            pattern = f"^{re.escape('_'.join(parts[:3]))}.*{re.escape(parts[-1])}$"
            
            # Find matching columns
            matches = [col for col in merged.columns if re.match(pattern, col)]
            
            if matches:
                valid_parcels.append(matches[0])  # Take the first match
            else:
                # Fallback: exact match
                if parcel in merged.columns:
                    valid_parcels.append(parcel)
        else:
            # If less than 4 parts, just do exact match
            if parcel in merged.columns:
                valid_parcels.append(parcel)
    
    print(f"  ✓ Found {len(valid_parcels)} valid parcels")
    if len(valid_parcels) > 0:
        print(f"    Examples: {valid_parcels[:3]}")
    
    if len(valid_parcels) == 0:
        ax.text(0.5, 0.5, f"No valid parcels\nfor {nucleus_key}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(nucleus_key)
        continue
    
    # --- Process each parcel ---
    normalized_data = []
    
    for parcel in valid_parcels:
        # Extract this parcel's data
        temp_df = merged[["subject_id", "age", parcel]].copy()
        
        #Normalize by 18-29 age group mean
        age_mask = (temp_df["age"] >= 18) & (temp_df["age"] < 30)
        norm_val = temp_df.loc[age_mask, parcel].mean()
        
        temp_df[f"{parcel}_norm"] = temp_df[parcel] - norm_val
        
        # Bin ages into 5-year intervals
        temp_df["age_bin"] = (temp_df["age"] // 5) * 5
        
        # Compute mean per bin for this parcel
        binned = temp_df.groupby("age_bin")[f"{parcel}_norm"].mean().reset_index()
        binned.columns = ["age_bin", parcel]
        
        normalized_data.append(binned.set_index("age_bin"))
    
    # Combine all parcels for this nucleus
    combined = pd.concat(normalized_data, axis=1)
    
    # --- Plot each parcel as a light line ---
    for parcel in combined.columns:
        ax.plot(combined.index, combined[parcel], alpha=0.3, linewidth=1, color='gray')
    
    # --- Plot mean across all parcels ---
    mean_trajectory = combined.mean(axis=1)
    ax.plot(mean_trajectory.index, mean_trajectory, 
            linewidth=3, color='red', label='Mean')
    
    # --- Formatting ---
    ax.set_xlabel("Age (years, 5-year bins)")
    ax.set_ylabel(f"Normalized {measure.capitalize()}")
    ax.set_title(f"{nucleus_key}")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

plt.suptitle(f"{measure.capitalize()} trajectories for top 10% parcels by thalamic nucleus ({hemi})", 
             fontsize=14, y=1.00)
plt.tight_layout()
plt.show()
# %%
#%% ##SCATTERPLOTS FOR ROI VALUES VS AGE - SPLIT THALAMIC NUCELI TOP10 (not normalized)##
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

measure = "thickness"
hemi = "lh"

# --- Load data ---
GMmeasure_df = pd.read_csv(f"/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/cortical/7schaefer_{measure}_stats.csv")
age_df = pd.read_csv("/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/subject_ages.csv")

# Load JSON file
with open(f"/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/functional_measures/top10_parcels_{hemi}.json", "r") as f:
    parcel_dict = json.load(f)

# --- Merge age and measure ---
merged = GMmeasure_df.merge(age_df, on="subject_id", how="inner")
print(f"Merged dataset has {len(merged)} subjects")

# --- Define thalamic nuclei ---
if hemi == 'rh':
    thal_nuc = [2] + list(range(4, 17))
else: 
    thal_nuc = [16] + list(range(18, 29))

# --- Create 12 plots (one per nucleus) ---
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for idx, nuc_id in enumerate(thal_nuc):
    ax = axes[idx]
    nucleus_key = f"Nucleus_{nuc_id}"
    
    # Get parcels for this nucleus
    if nucleus_key not in parcel_dict:
        print(f"Warning: {nucleus_key} not found in JSON")
        continue
    
    parcels_for_nucleus = parcel_dict[nucleus_key]
    
    print(f"\n{nucleus_key}: Looking for {len(parcels_for_nucleus)} parcels")
    
    # Filter to parcels that exist in the data using flexible matching
    valid_parcels = []
    for parcel in parcels_for_nucleus:
        # Split the parcel name into parts
        parts = parcel.split('_')
        
        # Create a regex pattern: match parts 0-2 and the last part (number)
        # Pattern: lh_7Networks_LH_Default*_9
        if len(parts) >= 4:
            pattern = f"^{re.escape('_'.join(parts[:3]))}.*{re.escape(parts[-1])}$"
            
            # Find matching columns
            matches = [col for col in merged.columns if re.match(pattern, col)]
            
            if matches:
                valid_parcels.append(matches[0])  # Take the first match
            else:
                # Fallback: exact match
                if parcel in merged.columns:
                    valid_parcels.append(parcel)
        else:
            # If less than 4 parts, just do exact match
            if parcel in merged.columns:
                valid_parcels.append(parcel)
    
    print(f"  ✓ Found {len(valid_parcels)} valid parcels")
    if len(valid_parcels) > 0:
        print(f"    Examples: {valid_parcels[:3]}")
    
    if len(valid_parcels) == 0:
        ax.text(0.5, 0.5, f"No valid parcels\nfor {nucleus_key}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(nucleus_key)
        continue
    
    # --- Process each parcel ---
    data = []
    
    for parcel in valid_parcels:
        # Extract this parcel's data
        temp_df = merged[["subject_id", "age", parcel]].copy()
        
        # #Normalize by 18-29 age group mean
        # age_mask = (temp_df["age"] >= 18) & (temp_df["age"] < 30)
        # norm_val = temp_df.loc[age_mask, parcel].mean()
        
        # temp_df[f"{parcel}_norm"] = temp_df[parcel] - norm_val
        
        # Bin ages into 5-year intervals
        temp_df["age_bin"] = (temp_df["age"] // 5) * 5
        
        # Compute mean per bin for this parcel
        binned = temp_df.groupby("age_bin")[f"{parcel}"].mean().reset_index()
        binned.columns = ["age_bin", parcel]
        
        data.append(binned.set_index("age_bin"))
    
    # Combine all parcels for this nucleus
    combined = pd.concat(data, axis=1)
    
    # --- Plot each parcel as a light line ---
    for parcel in combined.columns:
        ax.plot(combined.index, combined[parcel], alpha=0.3, linewidth=1, color='gray')
    
    # --- Plot mean across all parcels ---
    mean_trajectory = combined.mean(axis=1)
    ax.plot(mean_trajectory.index, mean_trajectory, 
            linewidth=3, color='red', label='Mean')
    
    # --- Formatting ---
    ax.set_xlabel("Age (years, 5-year bins)")
    ax.set_ylabel(f"Normalized {measure.capitalize()}")
    ax.set_title(f"{nucleus_key}")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.axhline(y=1.5, color='black', linestyle='--', linewidth=0.5)

plt.suptitle(f"{measure.capitalize()} trajectories for top 10% parcels by thalamic nucleus ({hemi})", 
             fontsize=14, y=1.00)
plt.tight_layout()
plt.show()
# %% STRUCTURAL-AGE EFFECT MATRIX (ROI x nuclei)
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import os

measure = "volume"

# Load data
resid_cor = pd.read_csv(f"/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/cortical/17Schaefer_{measure}_residualized.csv")
resid_thal = pd.read_csv("/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/thalamus/thomas_volumes_residualized.csv")

# Get thalamic nucleus columns (columns 2, 4-14)
thal_cols = [resid_thal.columns[2]] + list(resid_thal.columns[3:15])

# Get cortical parcel columns (400 columns)
cor_cols = [col for col in resid_cor.columns if col not in ["subject_id", "age", "age_resid"]]

# define age bins
age_bins = np.arange(18, 89, 3)  # 18-88 with step of 3 (two ages per bin)
resid_thal["bins"] = pd.cut(resid_thal["age"], bins=age_bins, include_lowest=True)
resid_cor["bins"] = pd.cut(resid_cor["age"], bins=age_bins, include_lowest=True)

# age effect for each thalamic nucleus (correlation with age_resid) 
# Compute correlation per nucleus per bin
thal_correlations = []  # Will be 35 bins x 12 nuclei
for bin_val in resid_thal["bins"].cat.categories:
    bin_data = resid_thal[resid_thal["bins"] == bin_val]
    correlations = []
    for nucleus in thal_cols:
        X = bin_data["age_resid"].values
        y = bin_data[nucleus].values
        if len(X) > 1:  # Need at least 2 points for correlation
            # np.corrcoef returns a 2x2 matrix, we want the off-diagonal element
            corr = np.corrcoef(X, y)[0, 1]
            print(f"for {bin_val} (nuc: {nucleus}) # subjects included: {len(X)}, corr = {corr}")
        else:
            corr = np.nan
        correlations.append(corr)
    thal_correlations.append(correlations)

thal_correlations = np.array(thal_correlations)  # Shape: (23 bins, 12 nuclei)
print("thalamic nuc correlations shape:", thal_correlations.shape)

# age effect for each cortical ROI
cor_correlations = []  # Will be 35 bins x 400 parcels
for bin_val in resid_cor["bins"].cat.categories:
    bin_data = resid_cor[resid_cor["bins"] == bin_val]
    correlations = []
    for parcel in cor_cols:
        X = bin_data["age_resid"].values
        y = bin_data[parcel].values
        if len(X) > 1:
            corr = np.corrcoef(X, y)[0, 1]
        else:
            corr = np.nan
        correlations.append(corr)
    cor_correlations.append(correlations)
    print(f"for {bin_val} (cortical) # subjects included: {len(bin_data)}")

cor_correlations = np.array(cor_correlations)  # Shape: (23 bins, 400 parcels)
print("cortical correlations shape:", cor_correlations.shape)

# Create 12 x 400 x 35 matrix
structural_matrix = np.zeros((len(thal_cols), len(cor_cols), len(age_bins)-1))
for bin_idx in range(len(age_bins)-1):
    # For each bin, compute outer product of thalamic and cortical correlations
    structural_matrix[:, :, bin_idx] = np.outer(thal_correlations[bin_idx], cor_correlations[bin_idx])

print("matrix shape:", structural_matrix.shape)  # Should be (12, 400, 23)
save_dir = '/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures'
np.save(os.path.join(save_dir, f'Tvol_17Cvol_age_matrix.npy'), structural_matrix)
#%% RAW STRUCTURAL MATRIX (ROI x nuclei)
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import os

measure = "volume"

# Load data
resid_cor = pd.read_csv(f"/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/cortical/17Schaefer_{measure}_residualized.csv")
resid_thal = pd.read_csv("/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/thalamus/thomas_volumes_residualized.csv")

# Identify columns
thal_cols = [c for c in resid_thal.columns if c not in ["subject_id", "age", "age_resid"]]     # 12 nuclei
cor_cols = [c for c in resid_cor.columns if c not in ["subject_id", "age", "age_resid"]]  # 400 parcels

# Define bins
age_bins = np.arange(18, 89, 3)
resid_thal["bins"] = pd.cut(resid_thal["age"], bins=age_bins, include_lowest=True)
resid_cor["bins"] = pd.cut(resid_cor["age"], bins=age_bins, include_lowest=True)

# Drop age to avoid conflict during merge
resid_thal_clean = resid_thal.drop(columns=["age"])
resid_cor_clean  = resid_cor.drop(columns=["age"])

# Merge
merged = pd.merge(
    resid_thal_clean[["subject_id", "bins"] + thal_cols],
    resid_cor_clean[["subject_id", "bins"] + cor_cols],
    on=["subject_id", "bins"],
    suffixes=("_thal", "_cor")
)

# Preallocate matrix: 12 nuclei × 400 parcels × number of bins
n_bins = len(age_bins) - 1
matrix = np.zeros((len(thal_cols), len(cor_cols), n_bins))

# Compute correlations bin-by-bin, nucleus-by-parcel
for b_idx, bin_val in enumerate(merged["bins"].cat.categories):
    bin_data = merged[merged["bins"] == bin_val]

    for i, nuc in enumerate(thal_cols):
        x = bin_data[nuc].values  # shape (N subjects in bin)

        for j, parcel in enumerate(cor_cols):
            y = bin_data[parcel].values

            # Correlation requires same-length vectors
            if len(x) > 1:
                matrix[i, j, b_idx] = np.corrcoef(x, y)[0, 1]
            else:
                matrix[i, j, b_idx] = np.nan


print("matrix shape:", matrix.shape)  # Should be (12, 400, 23)
# %% VISUALIZE STRUCTURAL MATRIX
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Visualize a single age bin as a heatmap (12 nuclei x 400 parcels)
bin_idx = 10  # Choose an age bin
plt.figure(figsize=(15, 4))
sns.heatmap(structural_matrix[:, :, bin_idx], cmap='RdBu_r', center=0, 
            xticklabels=False, yticklabels=thal_cols)
plt.title(f'Thalamus-Cortex Correlations for Age Bin {bin_idx}')
plt.xlabel('Cortical Parcels')
plt.ylabel('Thalamic Nuclei')
plt.tight_layout()
plt.show()
# %% VISUALIZE STRUCTURAL MATRIX (parcels ordered by SA ranking)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the S-A axis ranking CSV
sa_axis = pd.read_csv('/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/Sensorimotor_Association_Axis_AverageRanks.csv')

# Match parcel names and get the ordering based on finalrank.wholebrain
# Create a dictionary mapping parcel names to their S-A axis rank
parcel_to_rank = dict(zip(sa_axis['label'], sa_axis['finalrank.wholebrain']))

# Get the rank for each of your cortical parcels
# If parcel names don't match exactly, you may need to adjust the matching
parcel_ranks = [parcel_to_rank.get(parcel, 999) for parcel in cor_cols]  # 999 for missing parcels

# Get the indices that would sort parcels by S-A axis rank
sort_indices = np.argsort(parcel_ranks)

# Reorder the structural matrix columns according to S-A axis
structural_matrix_sorted = structural_matrix[:, sort_indices, :]

# Now visualize with ordered parcels
bin_idx = 1
fig, ax = plt.subplots(figsize=(15, 4))
sns.heatmap(structural_matrix_sorted[:, :, bin_idx], vmin=-0.6, vmax=0.6, cmap='RdBu_r', center=0, 
            xticklabels=False, yticklabels=thal_cols, ax=ax)
ax.vlines([133, 266], ymin=0, ymax=len(thal_cols), colors='black', linestyles='dashed', linewidth=2)
ax.set_title(f'Thalamus-Cortex Correlations for Age Bin {bin_idx}\n(Parcels ordered by S-A Axis)')
ax.set_xlabel('Cortical Parcels (Sensorimotor → Association)')
ax.set_ylabel('Thalamic Nuclei')
plt.tight_layout()
plt.show()
# %%
# %% VISUALIZE STRUCTURAL MATRIX (parcels ordered by 7 networks)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the S-A axis ranking CSV
sa_axis = pd.read_csv('/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/Sensorimotor_Association_Axis_AverageRanks.csv')

# Match parcel names and get the ordering based on finalrank.wholebrain
# Create a dictionary mapping parcel names to their S-A axis rank
parcel_to_rank = dict(zip(sa_axis['label'], sa_axis['finalrank.wholebrain']))

# Get the rank for each of your cortical parcels
# If parcel names don't match exactly, you may need to adjust the matching
parcel_ranks = [parcel_to_rank.get(parcel, 999) for parcel in cor_cols]  # 999 for missing parcels

# Get the indices that would sort parcels by S-A axis rank
sort_indices = np.argsort(parcel_ranks)

# Reorder the structural matrix columns according to S-A axis
structural_matrix_sorted = structural_matrix[:, sort_indices, :]

# Now visualize with ordered parcels
bin_idx = 1
fig, ax = plt.subplots(figsize=(15, 4))
sns.heatmap(structural_matrix_sorted[:, :, bin_idx], vmin=-0.6, vmax=0.6, cmap='RdBu_r', center=0, 
            xticklabels=False, yticklabels=thal_cols, ax=ax)
ax.vlines([133, 266], ymin=0, ymax=len(thal_cols), colors='black', linestyles='dashed', linewidth=2)
ax.set_title(f'Thalamus-Cortex Correlations for Age Bin {bin_idx}\n(Parcels ordered by S-A Axis)')
ax.set_xlabel('Cortical Parcels (Sensorimotor → Association)')
ax.set_ylabel('Thalamic Nuclei')
plt.tight_layout()
plt.show()
