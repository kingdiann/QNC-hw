#%%
import pandas as pd 

data = pd.read_csv('/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/thalamus/thomas_volumes_VIF.csv')

print(data.head())

#%% TEST
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = data[['age', 'CSFvol']]

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
print("VIF collinearity")
print(vif_data)
# %% TEST
import pandas as pd
import pingouin as pg

# 1. Load your CSV  
df = pd.read_csv('/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/thalamus/thomas_volumes_VIF.csv')  
 
print("Mediation Model:")
# 3. Run mediation analysis  
results = pg.mediation_analysis(data = df,
                                x    = 'age',
                                m    = 'CSFvol',
                                y    = 'PULVvol',
                                alpha = 0.05,
                                n_boot = 10000,      # you may increase for better precision
                                seed   = 42)       # for reproducibility
print(results)

# Partial correlation
print("Partial Correlation:")
pcorr = pg.partial_corr(data=df, x='age', y='PULVvol', covar='CSFvol')
print(pcorr)

# %% # MAKE A RESIDUALIZED THALAMUS DATASET 
import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv("/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/thalamus/thomas_volumes_raw_edit.csv")

# Identify columns
x_col = "age"
m_col = "CSFvol"
y_cols = [col for col in df.columns if col not in ["subject_id", "CSFvol", "age"]]

# --- Start new DataFrame with real (unresidualized) age ---
resid_df = pd.DataFrame()
resid_df[x_col] = df[x_col]  # keep *real* age

# --- Residualize age (x) against CSFvol (m) ---
x_model = sm.OLS(df[x_col], sm.add_constant(df[m_col])).fit()
resid_df[f"{x_col}_resid"] = x_model.resid

# --- Residualize each Y against CSFvol ---
for y in y_cols:
    model = sm.OLS(df[y], sm.add_constant(df[m_col])).fit()
    resid_df[f"{y}"] = model.resid

# --- Save only residuals + real age ---
out_path = "/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/thalamus/thomas_volumes_residualized.csv"
resid_df.to_csv(out_path, index=False)
print(f"Residualized dataset saved to: {out_path}")

# %% # MAKE A RESIDUALIZED CORTEX DATASET
import pandas as pd
import statsmodels.api as sm

measure = "volume"

# Load data
df = pd.read_csv(f"/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/cortical/17schaefer_{measure}_stats.csv")
age_df = pd.read_csv("/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/subject_ages.csv")
CSF_df = pd.read_csv("/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/CSF_volumes.csv")

# --- Merge age and GM measure ---
merged = df.merge(age_df, on="subject_id", how="inner")
final = merged.merge(CSF_df, on="subject_id", how="inner")
print(f"Merged dataset has {len(final)} subjects")
print(final.columns.tolist())

# Identify columns
x_col = "age"
m_col = "CSFvol"
y_cols = [col for col in final.columns if col not in ["subject_id", "CSFvol", "age"]]

# --- Start new DataFrame with real (unresidualized) age ---
resid_final = pd.DataFrame()
resid_final[x_col] = final[x_col]  # keep *real* age

# --- Residualize age (x) against CSFvol (m) ---
x_model = sm.OLS(final[x_col], sm.add_constant(final[m_col])).fit()
resid_final[f"{x_col}_resid"] = x_model.resid

# --- Residualize each Y against CSFvol ---
for y in y_cols:
    model = sm.OLS(final[y], sm.add_constant(final[m_col])).fit()
    resid_final[f"{y}"] = model.resid

# --- Save only residuals + real age ---
out_path = f"/mnt/DataDrive1/data_preproc/human_mri/CamCAN/derivatives/structural_measures/cortical/17Schaefer_{measure}_residualized.csv"
resid_final.to_csv(out_path, index=False)
print(f"Residualized dataset saved to: {out_path}")
# %%
