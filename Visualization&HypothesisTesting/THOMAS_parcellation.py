#%%
import nibabel as nib
import numpy as np
import os

out_dir = '/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/functional_measures/THOMAS_parcellation'
base_dir = '/mnt/DataDrive1/data_preproc/human_mri/CamCAN'
subjects = [d for d in os.listdir(base_dir) if d.startswith('sub-')]

# Load fMRI and atlas images
thalamus_mask_img = nib.load('/mnt/DataDrive3/diana/atlases/atlas-THOMAS_space-MNI152NLin2009cAsym_resampled.nii.gz')
mask_data = thalamus_mask_img.get_fdata().astype(int)  # shape: (X, Y, Z)


for subject in subjects:
        print(f"Processing subject: {subject}")

        if not os.path.exists(f'{base_dir}/{subject}/func/{subject}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'):
            print(f"Missing file for {subject}, skipping.")
            continue

        #if os.path.exists(f'{indi_dir}/{subject}_space-MNI152NLin2009cAsym_atlas-THALAMUS_hemi-L_THOMAS.npy'):
            #print(f"{subject} already completed, skipping.")
            #continue

        func_img = nib.load(f'{base_dir}/{subject}/func/{subject}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz') 
        func_data = func_img.get_fdata()        # shape: (X, Y, Z, T)

        if mask_data.shape != func_data.shape[:3]:
                print(f"skipping {subject}: shape mismatch - mask {mask_data.shape} vs fMRI {func_data.shape}")
                continue

        # Extract left and right thalamus voxel-wise time series
        rh_labels = np.arange(2,15)
        rh_mask = np.isin(mask_data, rh_labels)
        lh_labels = np.arange(16,29)
        lh_mask = np.isin(mask_data, lh_labels)

        # Extract voxel-wise time series (n_voxels, n_timepoints)
        lh_timeseries = func_data[lh_mask, :]  # shape: (n_lh_voxels, T)
        rh_timeseries = func_data[rh_mask, :]  # shape: (n_rh_voxels, T)

        print("LH shape:", lh_timeseries.shape)
        print("RH shape:", rh_timeseries.shape)

        np.save(
            os.path.join(out_dir, f'{subject}_space-MNI152NLin2009cAsym_atlas-THALAMUS_hemi-L_THOMAS.npy'),
            lh_timeseries
        )
        np.save(
            os.path.join(out_dir, f'{subject}_space-MNI152NLin2009cAsym_atlas-THALAMUS_hemi-R_THOMAS.npy'),
            rh_timeseries
        )

# %%
