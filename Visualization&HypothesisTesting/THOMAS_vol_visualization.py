
#%% ##SCATTERPLOTS FOR NUCLEI VOLUME VS AGE (each nuclei as a fraction of total thalamus vol) ##
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# -----------------------------
# Load merged CSV
# -----------------------------
df = pd.read_csv("/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/thalamus/thomas_volumes_raw.csv")

# Automatically find the age column
age_col = next((c for c in df.columns if 'age' in c.lower()), None)
if age_col is None:
    raise ValueError("No column containing 'age' found in your CSV!")
print(f"Using age column: {age_col}")

# -----------------------------
# Identify nucleus labels
# -----------------------------
volume_cols = [c for c in df.columns if c.startswith(("L_", "R_"))]
labels = sorted(set(int(c.split("_")[1]) for c in volume_cols))

# Compute bilateral volumes
for label in labels:
    left = f"L_{label}"
    right = f"R_{label}"
    if left in df.columns and right in df.columns:
        df[f"Both_{label}"] = df[left] + df[right]

# -----------------------------
# Plot: separate regression per nucleus
# -----------------------------
output_dir = "thomas_plots"
os.makedirs(output_dir, exist_ok=True)

for label in labels:
    col = f"Both_{label}"
    if col not in df.columns:
        continue

    x = df[age_col]
    y = df[col]

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line = slope * x + intercept

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.6, label="Subjects")
    plt.xlabel("Age (years)")
    plt.ylabel("Bilateral Volume (mm³)")
    plt.title(f"Nucleus {label}")
    plt.tight_layout()
    plt.show()
#%% ##THALAMIC NUCLEI VOLUME LOSS 20 V 80
import numpy as np
import pandas as pd
from scipy import stats
import nibabel as nib
import matplotlib.pyplot as plt

df = pd.read_csv("/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/thalamus/thomas_volumes_raw.csv")
parcel_cols = [c for c in df.columns if c not in ["subject_id", "age"]]
 
# --- Identify nucleus labels ---
volume_cols = [c for c in df.columns if c.startswith(("L_", "R_"))]
labels = sorted(set(int(c.split("_")[1]) for c in volume_cols))

# --- COMPUTE PERCENT CHANGE PER PARCEL ---
stat_dict = {}

for parcel in parcel_cols:
    stat_dict[parcel] = {}

    low_group = df.loc[
        (df["age"] >= 18) & (df["age"] < 30),
        parcel
    ].dropna()
    high_group = df.loc[
        (df["age"] >= 80) & (df["age"] < 90),
        parcel
    ].dropna()
    percent_change = (high_group.mean() - low_group.mean()) / low_group.mean() * 100
    stat_dict[parcel][f"20_vs_80"] = percent_change
    label = f"20_vs_80"

# ----------------------------
# LOAD TEMPLATE (thalamus nuclei map)
# ----------------------------
template_path = "/mnt/DataDrive3/diana/atlases/atlas-THOMAS_space-MNI152NLin2009cAsym_res-01.nii.gz"
template_img = nib.load(template_path)
template_img = nib.as_closest_canonical(template_img) # Ensure RAS orientation
template_data = template_img.get_fdata()
affine = template_img.affine

unique_labels = np.unique(template_data)
unique_labels = unique_labels[unique_labels != 0]

# ----------------------------
# VISUALIZE EACH COMPARISON IN 3D
# ----------------------------
for comparison in next(iter(stat_dict.values())).keys():

    # Initialize voxel map
    stat_map = np.full_like(template_data, np.nan, dtype=float)

    for label_val in unique_labels:
        if 2 <= label_val <= 14:
            parcel_name = f"L_{int(label_val)}"
        elif 16 <= label_val <= 28:
            parcel_name = f"R_{int(label_val - 14)}"
        else:
            continue

        if parcel_name in stat_dict:
            percent_val = stat_dict[parcel_name].get(comparison, np.nan)
            stat_map[template_data == label_val] = percent_val

    # Extract voxel coordinates and color values
    coords = np.column_stack(np.nonzero(~np.isnan(stat_map)))
    if coords.size == 0:
        print("No significant nuclei (p<0.05).")
        continue

    # Convert voxel indices to MNI coordinates
    mni_coords = nib.affines.apply_affine(affine, coords)
    colors = stat_map[~np.isnan(stat_map)]

    # ----------------------------
    # 3D PLOT
    # ----------------------------
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        mni_coords[:, 0],
        mni_coords[:, 1],
        mni_coords[:, 2],
        c=colors,
        cmap='hot',
        s=15
    )
    plt.colorbar(sc, ax=ax, shrink=0.6, label='percent change (%)')
    ax.view_init(elev=30, azim=-190)
    ax.set_title('Thalamus percent change')
    ax.set_xlabel('X (MNI)')
    ax.set_ylabel('Y (MNI)')
    ax.set_zlabel('Z (MNI)')
    plt.show()

    out_path = "/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/thalamus"
    stat_img = nib.Nifti1Image(stat_map, affine)
    nib.save(stat_img, out_path)
#%% ##THALAMIC NUCLEI VOLUME CHANGES ACROSS DECADES (t stat) - grouped plot ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import ttest_ind

# --- Load data ---
df = pd.read_csv("/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/thalamus/thomas_volumes_raw.csv")

# --- Identify nucleus labels ---
volume_cols = [c for c in df.columns if c.startswith(("L_", "R_"))]
labels = sorted(set(int(c.split("_")[1]) for c in volume_cols))

# --- Combine left and right hemispheres ---
for label in labels:
    left = f"L_{label}"
    right = f"R_{label}"
    if left in df.columns and right in df.columns:
        df[f"whole_{label}"] = df[left] + df[right]

# --- Define decade bins ---
bins = list(range(20, 100, 10))  # 20–30, 30–40, ..., 80–90
df["age_group"] = pd.cut(df["age"], bins=bins, labels=[f"{b}" for b in bins[:-1]])

# --- Prepare comparisons (20 vs 30, 30 vs 40, …) ---
comparisons = [(f"{bins[i]}", f"{bins[i+1]}") for i in range(len(bins)-2)]

# --- Compute percent change and t-tests ---
percent_dict = {}
tstat_dict = {}

for label in labels:
    parcel = f"whole_{label}"
    if parcel not in df.columns:
        continue
    percent_dict[parcel] = {}
    tstat_dict[parcel] = {}

    for low_label, high_label in comparisons:
        low_group = df.loc[df["age_group"] == low_label, parcel].dropna()
        high_group = df.loc[df["age_group"] == high_label, parcel].dropna()

        if len(low_group) > 1 and len(high_group) > 1:
            t_stat, p_val = ttest_ind(high_group, low_group, equal_var=False)
            if p_val < 0.05:
                percent_change = (high_group.mean() - low_group.mean()) / low_group.mean() * 100
                percent_dict[parcel][f"{low_label}_vs_{high_label}"] = percent_change
                tstat_dict[parcel][f"{low_label}_vs_{high_label}"] = t_stat

# --- Prepare for plotting ---
comparisons_labels = sorted({comp for d in percent_dict.values() for comp in d.keys()})
nuclei = list(percent_dict.keys())
num_nuclei = len(nuclei)
bar_width = 0.06
x = np.arange(len(comparisons_labels))
cmap = cm.get_cmap("tab20", num_nuclei)
colors = [cmap(i) for i in range(num_nuclei)]

# --- Plot 1: Percent Change ---
plt.figure(figsize=(14, 6))
for j, parcel in enumerate(nuclei):
    vals = [percent_dict[parcel].get(comp, np.nan) for comp in comparisons_labels]
    mask = ~np.isnan(vals)
    plt.bar(
        x[mask] + j * bar_width - (num_nuclei * bar_width / 2),
        np.array(vals)[mask],
        width=bar_width,
        color=colors[j],
        label=parcel
    )
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xticks(x, comparisons_labels, rotation=45)
plt.ylabel("Percent Change in Volume")
plt.title("Thalamic nuclei volume percent change across decades (p < 0.05)")
plt.legend(title="Nucleus", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- Plot 2: T-statistics ---
plt.figure(figsize=(14, 6))
for j, parcel in enumerate(nuclei):
    vals = [tstat_dict[parcel].get(comp, np.nan) for comp in comparisons_labels]
    mask = ~np.isnan(vals)
    plt.bar(
        x[mask] + j * bar_width - (num_nuclei * bar_width / 2),
        np.array(vals)[mask],
        width=bar_width,
        color=colors[j],
        label=parcel
    )
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xticks(x, comparisons_labels, rotation=45)
plt.ylabel("T-statistic")
plt.title("Thalamic nuclei t-statistics across decades (p < 0.05)")
plt.legend(title="Nucleus", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% ##PERCENT CHANGE IN THALAMUS VOLUME ACROSS DECADES VISUALIZATION ##
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import nibabel as nib
from nilearn import datasets, plotting  

merged = pd.read_csv("/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/thalamus/thomas_volumes_raw.csv")
parcel_cols = [c for c in df.columns if c not in ["subject_id", "age"]]
 
# --- Identify nucleus labels ---
volume_cols = [c for c in df.columns if c.startswith(("L_", "R_"))]
labels = sorted(set(int(c.split("_")[1]) for c in volume_cols))

# Compute bilateral volumes
for label in labels:
    left = f"L_{label}"
    right = f"R_{label}"
    if left in df.columns and right in df.columns:
        df[f"{label}"] = df[left] + df[right]
parcel_cols = [c for c in df.columns if not c.startswith(("L_", "R_")) and c not in ["subject_id", "age"]]

target_ages = [30, 40, 50, 60, 70, 80]
percent_change_dict = {}

for parcel in parcel_cols:
    young_mean = merged[merged["age"] <= 20][parcel].mean()
    percent_change_dict[parcel] = {}
    for t in target_ages:
        old_mean = merged[merged["age"] >= t][parcel].mean()
        percent_change_dict[parcel][str(t)] = ((old_mean - young_mean) / young_mean) * 100

print("Computed percent change for all parcels.")

# --- Choose hemisphere ---
hemi = "rh"  # or "lh" depending on which annot you want to apply to
annot_path = os.path.expandvars(
    f"/mnt/DataDrive1/data_preproc/human_mri/CamCAN/sourcedata/fastsurfer/Recon/fsaverage/label/{hemi}.Schaefer2018_400Parcels_7Networks_order.annot"
)

# --- Load annotation file (fsaverage parcellation) ---
labels, ctab, names = nib.freesurfer.read_annot(annot_path)
region_names = [n.decode("utf-8") for n in names]

# --- Create a mapping from region name to percent change ---
# match parcel columns to annotation region names
parcel_to_value = {}

print("\n--- Matching parcels from CSV to annotation regions ---")
n_matched = 0

for region in region_names:
    # ignore the first 3 chars (e.g., 'lh_' or 'rh_') in CSV column names
    match = [p for p in percent_change_dict.keys() if region.lower() in p[3:].lower()]
    for target_age in target_ages:
        if match:
            matched_parcel = match[0]
            val = percent_change_dict[matched_parcel][str(target_age)]
            parcel_to_value[region] = val
            n_matched += 1
            print(f"✓ {region:<40} ← {matched_parcel:<40} ({val:.2f})")
        else:
            parcel_to_value[region] = np.nan
            print(f"✗ {region:<40} — no match found")

print(f"\nMatched {n_matched}/{len(region_names)} regions")

# --- Assign percent change values to vertices on fsaverage surface ---
percent_change_map = np.zeros_like(labels, dtype=float)
for idx, region in enumerate(region_names):
    percent_change_map[labels == idx] = parcel_to_value[region]
print(f"Percent change values assigned to {np.sum(~np.isnan(percent_change_map))} vertices.")
print(f"Range of percent_change_map: {np.nanmin(percent_change_map)} to {np.nanmax(percent_change_map)}")

for i, region in enumerate(region_names[:10]):
    matched_value = parcel_to_value[region]
    print(f"{i}: {region} → {matched_value}")

plotting.view_surf(
    surf_mesh=fsavg[f"infl_right" if hemi == "rh" else "infl_left"],
    surf_map=percent_change_map,
    colorbar=True,
    title=f"{hemi.upper()} Percent Change in volumes",
    symmetric_cmap=False,
    cmap="coolwarm"
)