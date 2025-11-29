#%%
import nibabel as nib
import numpy as np
import os

out_dir = '/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/functional_measures/Schaefer_parcellation'
base_dir = '/mnt/DataDrive1/data_preproc/human_mri/CamCAN'
subjects = [d for d in os.listdir(base_dir) if d.startswith('sub-')]

# Load atlas image once (since it's the same for all subjects/runs)
atlas_img = nib.load('/mnt/DataDrive3/diana/atlases/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_resampled.nii.gz')
atlas_data = atlas_img.get_fdata().astype(int)  # shape: (X, Y, Z)

# Get unique ROI labels (excluding background 0)
roi_labels = np.unique(atlas_data)
roi_labels = roi_labels[roi_labels != 0]

# Define left and right hemisphere labels
lh_labels = roi_labels[roi_labels <= 200]
rh_labels = roi_labels[roi_labels > 200]

# Loop over subjects and runs
for subject in subjects:
    print(f"Processing subject: {subject}")

    if not os.path.exists(f'{base_dir}/{subject}/func/{subject}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'):
        print(f"Missing file for {subject}, skipping.")
        continue

    if os.path.exists(f'{out_dir}/{subject}_space-MNI152NLin2009cAsym_atlas-Schaefer_hemi-rh.npy'):
        print(f"{subject} already completed, skipping.")
        continue

    # Construct functional image path
    func_path = f'{base_dir}/{subject}/func/{subject}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    
    # Check if file exists before processing
    if not os.path.exists(func_path):
        print(f"Warning: File not found - {func_path}")
        continue
        
    # Load fMRI image
    func_img = nib.load(func_path)
    func_data = func_img.get_fdata()  # shape: (X, Y, Z, T)
    
    n_timepoints = func_data.shape[-1]

    if atlas_data.shape != func_data.shape[:3]:
        print(f"skipping {subject}: shape mismatch - mask {atlas_data.shape} vs fMRI {func_data.shape}")
        continue    

    # Initialize separate timeseries arrays
    lh_timeseries = np.zeros((len(lh_labels), n_timepoints))
    rh_timeseries = np.zeros((len(rh_labels), n_timepoints))
    
    # Extract left hemisphere timeseries
    for i, roi in enumerate(lh_labels):
        roi_mask = atlas_data == roi
        roi_voxels = func_data[roi_mask, :]
        lh_timeseries[i] = roi_voxels.mean(axis=0)
    
    # Extract right hemisphere timeseries
    for i, roi in enumerate(rh_labels):
        roi_mask = atlas_data == roi
        roi_voxels = func_data[roi_mask, :]
        rh_timeseries[i] = roi_voxels.mean(axis=0)
    
    # Save timeseries 
    lh_filename = f'{subject}_space-MNI152NLin2009cAsym_atlas-Schaefer_hemi-lh.npy'
    rh_filename = f'{subject}_space-MNI152NLin2009cAsym_atlas-Schaefer_hemi-rh.npy'
    
    np.save(os.path.join(out_dir, lh_filename), lh_timeseries)
    np.save(os.path.join(out_dir, rh_filename), rh_timeseries)
    
    # (#parcels, #volumes)
    print(f"  LH Timeseries shape: {lh_timeseries.shape}")
    print(f"  RH Timeseries shape: {rh_timeseries.shape}")
    print(f"  Saved: {lh_filename} and {rh_filename}")
print("Processing complete!")
print(f"Atlas ROI labels: {np.unique(atlas_data)}")

# %%
