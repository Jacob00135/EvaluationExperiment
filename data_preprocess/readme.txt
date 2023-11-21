this CrossValid folder will contain 5 fold cross validation split information

step1, run MRI_analysis.py to find MRI data that meets the 1.5 T, T1-weighted MRI scans criteria

step2, run the combine.py to combine all subjects from all cohorts and output results in ADNI.csv

step3, run the split.py to split the ADNI into train, valid, test, the data split results will be stored in folder cross0/ cross1/ etc

step4, run appendNonImage.py to fill up non-imaging information by table joining

