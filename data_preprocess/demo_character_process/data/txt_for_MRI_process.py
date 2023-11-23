import pandas as pd

data = pd.read_csv('./ADNI.csv')
data["filename"] = data["filename"].str.replace(".npy", ".nii").astype("str")

data['full_path'] = data['SavePath'] + data['filename']
data = data.loc[:, 'full_path']
data.to_csv('../../MRI_process/test_data/mri_for_process.txt', header=None, index=0)