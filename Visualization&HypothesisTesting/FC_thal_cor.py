# %%
import os 
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

# %% ** as of 11/17: using this script to find top 10% of correlated regions in the young **
# additional preprocessing after fmriprep before fingerprint analysis
# do it for cortex / thalamus for each hemi separately
# 1. smooth with 2mm sigma gaussian kernel within the mask (provided in atlas folder)
# 2. linear detrend
# 3. zscore across all timepoints
# 4. concatenate all timepoints across all runs for each subject
# 5. optional for roi level data (i.e., cortex), average across all voxels within the roi
# 6. save the concatenated data as a numpy array

# %% set path and parameters
# --------------------------
# This cell should be the only part that needs to be changed
# --------------------------

# the root path of the dataset
dataset_root = '/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/functional_measures'  # change to your own path
indi_dir = os.path.join(dataset_root, 'indiv_timeseries')

# subject list which list all the subjects, which will be used to match the source and target data
# | File format: one column, no header
hemi = 'rh'
sublist_f = f'/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/subject_ages.csv'
sublist = pd.read_csv(sublist_f, sep=',')
age_bins = np.arange(18, 89, 3)
sublist['age_bin'] = pd.cut(sublist['age'], bins=age_bins, right=False, labels=False)

parcel_file = os.path.join(dataset_root, "Schaefer2018_400Parcels_7Networks_order.txt")
all_parcel_names = [line.strip() for line in open(parcel_file, "r")]

# Filter to only the current hemisphere
if hemi == 'lh':
    parcel_names = [p for p in all_parcel_names if '_LH_' in p or p.startswith('L_')]
else:
    parcel_names = [p for p in all_parcel_names if '_RH_' in p or p.startswith('R_')]

#, names=['sub-CC'], dtype=str

fingerprint_source_dataf = f'xxx_Schaefer_hemi-{hemi}_processed.npy'  # change to your own file name
fingerprint_target_dataf = f'nucleus_timeseries/xxx_THOMAS12_hemi-{hemi[0].upper()}.npy' # change to your own file name
source_dataf_tpl = os.path.join(indi_dir, fingerprint_source_dataf)
target_dataf_tpl = os.path.join(indi_dir, fingerprint_target_dataf)

# # the fingerprint FC will be saved as a numpy array in the following directory after the analysis
# save_dir = os.path.join(dataset_root, 'result')
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# save_f_tpl = f'fc-xxx_hemi-{hemi}_subgroup-{age_bin}.npy'

# %% regular thalamus - cortical FC measure
import numpy as np
import json
import os
from scipy.spatial.distance import cdist

# Group by age bin and compute FC for each
age_bin_FCs = {}

for age_bin in sublist['age_bin'].dropna().unique():
    # Get subjects in this age bin
    bin_subjects = sublist[sublist['age_bin'] == age_bin]['subject_id']
    
    THAL_list = []
    SCH_list = []
    
    for subid in bin_subjects:
        src = source_dataf_tpl.replace("xxx", subid)
        trg = target_dataf_tpl.replace("xxx", subid)

        print(f"Looking for source: {src}")
        print(f"Looking for target: {trg}")
        print(f"Source exists: {os.path.exists(src)}")
        print(f"Target exists: {os.path.exists(trg)}")
        
        if not os.path.exists(src) or not os.path.exists(trg):
            print(f"Skipping {subid}: missing files")
            continue
        
        thal = np.load(trg)
        sch = np.load(src)
        
        THAL_list.append(thal)
        SCH_list.append(sch)
    
    if len(THAL_list) == 0:
        print(f"No subjects found for age bin {age_bin}")
        continue
    
    # Average within this age bin
    THAL_avg = np.mean(np.stack(THAL_list, axis=0), axis=0)
    SCH_avg = np.mean(np.stack(SCH_list, axis=0), axis=0)
    
    # Compute FC for this age bin
    FC = 1 - cdist(THAL_avg, SCH_avg, metric="correlation")
    age_bin_FCs[age_bin] = FC
    
    print(f"Age bin {age_bin}: FC shape {FC.shape}, n_subjects={len(THAL_list)}")

# Save all matrices
np.save(os.path.join(dataset_root, f'age_bin_FCs_hemi-{hemi}.npy'), age_bin_FCs)
# %%
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_fc_matrix(age_bin, age_bin_FCs, parcel_names, nucleus_names=None):
    """
    Visualize FC matrix for a specific age bin
    
    Parameters:
    -----------
    age_bin : int
        Age bin index to visualize
    age_bin_FCs : dict
        Dictionary of FC matrices keyed by age bin
    parcel_names : list
        List of cortical parcel names
    nucleus_names : list, optional
        List of thalamic nucleus names
    """
    if age_bin not in age_bin_FCs:
        print(f"Age bin {age_bin} not found")
        return
    
    FC = age_bin_FCs[age_bin]
    age_range = f"{age_bins[age_bin]}-{age_bins[age_bin+1]}" if age_bin < len(age_bins)-1 else f"{age_bins[age_bin]}+"
    
    plt.figure(figsize=(15, 6))
    sns.heatmap(FC, cmap='coolwarm', center=0, 
                xticklabels=parcel_names if len(parcel_names) < 50 else False,
                yticklabels=nucleus_names if nucleus_names else True,
                cbar_kws={'label': 'Correlation'})
    plt.title(f'Thalamic-Cortical FC | Age: {age_range} years | Hemisphere: {hemi.upper()}')
    plt.xlabel('Cortical Parcels')
    plt.ylabel('Thalamic Nuclei')
    plt.tight_layout()
    plt.show()

# Example usage:
visualize_fc_matrix(age_bin=22, age_bin_FCs=age_bin_FCs, parcel_names=parcel_names)


# %% 3. FIND TOP 10% parcels per nucleus
# # the REAL thalamic labels you extracted earlier:
# if hemi == 'rh':
#     thal_labels = [2] + list(range(4, 17))
# else: 
#     thal_labels = [16] + list(range(18, 29))
# thal_labels = thal_labels[:n_nuclei]         # make sure it matches 12

# top10_dict = {}

# for i in range(n_nuclei):

#     row = FC[i, :]                    # FC values for nucleus i
#     idx = np.argsort(row)[-n_top:]    # top 10% indices

#     def clean_parcel_name(name, hemi):
#         parts = name.split('\t')[1].split('_')
#         # Remove 4th element if it's not a number
#         if len(parts) >= 4:
#             try:
#                 int(parts[3])
#             except ValueError:
#                 parts = parts[:3] + parts[4:]
#         return f"{hemi}_" + "_".join(parts)

#     parcels = [clean_parcel_name(parcel_names[j], hemi) for j in idx]

#     true_label = thal_labels[i]
#     top10_dict[f"Nucleus_{true_label}"] = parcels

# # ---------------------------------------
# # 4. Save JSON
# # ---------------------------------------

# save_json = f"/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/functional_measures/top10_parcels_{hemi}.json"

# with open(save_json, "w") as f:
#     json.dump(top10_dict, f, indent=4)

# print(f"Saved top parcel list → {save_json}")

# # %% COLLAPSE THALAMUS INTO NUCLEI (12, T) FROM VOXEL-WISE TIME SERIES
# import numpy as np
# import os
# import nibabel as nib

# hemi = "rh"  # 'lh' or 'rh'

# # paths
# thal_atlas = "/mnt/DataDrive3/diana/atlases/atlas-THOMAS_space-MNI152NLin2009cAsym_resampled.nii.gz"
# indiv_dir = "/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/functional_measures/indiv_timeseries"
# save_dir = "/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/functional_measures/indiv_timeseries/nucleus_timeseries"

# os.makedirs(save_dir, exist_ok=True)

# mask_data = nib.load(thal_atlas).get_fdata().astype(int)
# mask_flat = mask_data.flatten()
# print("Atlas shape:", mask_flat.shape)

# # Correct THOMAS nucleus labels (12 per hemisphere)
# if hemi == "lh":
#     nucleus_labels = [16] + list(range(18, 29))
#     prefix = "hemi-L"
# else:
#     nucleus_labels = [2] + list(range(4, 15))
#     prefix = "hemi-R"

# subjects = [s for s in os.listdir(indiv_dir) if s.endswith("rh_processed.npy")]
# print ("Subjects: ", subjects[:10])

# for f in subjects:
#     print(f"Processing file: {f}")
#     subid = f.split("_")[0]

#     ts = np.load(os.path.join(indiv_dir, f))  # (n_vox, T)

#     # Recreate the voxel order mask used earlier
#     used_mask = np.isin(mask_flat, nucleus_labels)
#     label_values = mask_flat[used_mask]      # per voxel label (length n_vox)

#     nucleus_ts_list = []

#     for lab in nucleus_labels:
#         vox_idx = np.where(label_values == lab)[0]
#         nucleus_ts = ts[vox_idx].mean(axis=0)   # mean across voxels
#         nucleus_ts_list.append(nucleus_ts)

#     nucleus_ts = np.vstack(nucleus_ts_list)   # (12, T)

#     out = os.path.join(save_dir, f"{subid}_THOMAS12_{prefix}.npy")
#     np.save(out, nucleus_ts)
#     print("Saved:", out)