#%%
import nibabel as nib
import os
import numpy as np
import pandas as pd

# -----------------------------
# PATHS AND SETTINGS
# -----------------------------
sub_dir = '/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/resampled'
subjects = [d for d in os.listdir(sub_dir) if d.startswith('sub-')]
label_range = range(2, 28)  # 2–14 inclusive

# Store results
all_raw = []
all_norm = []

# # -----------------------------
# # LOOP THROUGH SUBJECTS (direct)
# # -----------------------------
# for subject in subjects:
#     subj_data_raw = {"Subject": subject}
#     subj_data_norm = {"Subject": subject}

#     for hemi in ['left', 'right']:
#         hemi_short = 'L' if hemi == 'left' else 'R'
#         nii_path = f"{sub_dir}/{subject}/{hemi}/thomasfull_{hemi_short}.nii.gz"
#         if not os.path.exists(nii_path):
#             print(f"Missing: {nii_path}")
#             continue

#         # Load NIfTI
#         img = nib.load(nii_path)
#         data = img.get_fdata()
#         voxel_volume = np.prod(img.header.get_zooms())

#         # Total volume (nonzero)
#         total_volume = np.sum(data > 0) * voxel_volume

#         # Label-specific volumes
#         labels, counts = np.unique(data, return_counts=True)
#         label_dict = dict(zip(labels.astype(int), counts))

#         for label in label_range:
#             count = label_dict.get(label, 0)
#             vol = count * voxel_volume
#             norm_vol = vol / total_volume if total_volume > 0 else np.nan
#             subj_data_raw[f"{hemi_short}_{label}"] = vol
#             subj_data_norm[f"{hemi_short}_{label}"] = norm_vol

#     all_raw.append(subj_data_raw)
#     all_norm.append(subj_data_norm)

# -----------------------------
# LOOP THROUGH SUBJECTS (MNI)
# -----------------------------
for subject in subjects:
    subj_data_raw = {"Subject": subject}
    subj_data_norm = {"Subject": subject}

    nii_path = f"{sub_dir}/{subject}/THOMAS_T1W_fromMNI.nii.gz"
    if not os.path.exists(nii_path):
        print(f"Missing: {nii_path}")
        continue

    # Load NIfTI
    img = nib.load(nii_path)
    data = img.get_fdata()
    voxel_volume = np.prod(img.header.get_zooms())

    # Total volume (nonzero)
    total_volume = np.sum(data > 0) * voxel_volume

    # Label-specific volumes
    labels, counts = np.unique(data, return_counts=True)
    label_dict = dict(zip(labels.astype(int), counts))

    for label in label_range:
        count = label_dict.get(label, 0)
        vol = count * voxel_volume
        norm_vol = vol / total_volume if total_volume > 0 else np.nan
        subj_data_raw[f"{label}"] = vol
        subj_data_norm[f"{label}"] = norm_vol

    all_raw.append(subj_data_raw)
    all_norm.append(subj_data_norm)

# -----------------------------
# CREATE AND SAVE COMPOSITE FILES
# -----------------------------
df_raw = pd.DataFrame(all_raw).sort_values("Subject")
df_norm = pd.DataFrame(all_norm).sort_values("Subject")

out_raw = os.path.join(sub_dir, "thomasMNI_volumes_raw.csv")
out_norm = os.path.join(sub_dir, "thomasMNI_volumes_normalized.csv")

df_raw.to_csv(out_raw, index=False)
df_norm.to_csv(out_norm, index=False)

print(f"\n Saved composite raw volumes to: {out_raw}")
print(f"Saved composite normalized volumes to: {out_norm}")

# %%
