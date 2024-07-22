# DATASETS

## Link to download the datasets
All data used in this study should be available free of charge upon request from the specific cohorts.
1. NACC data can be downloaded at <a href="https://naccdata.org">naccdata.org</a>.
2. Data from ADNI, AIBL, NIFD, PPMI and 4RTNI can be downloaded from the LONI website at <a href="https://ida.loni.usc.edu">ida.loni.usc.edu</a>.
3. OASIS data can be downloaded at <a href="https://sites.wustl.edu/oasisbrains/">sites.wustl.edu/oasisbrains/</a>.
4. Data from FHS (<a href="https://www.framinghamheartstudy.org/fhs-for-researchers/data-available-overview/">www.framinghamheartstudy.org/fhs-for-researchers/data-available-overview/</a>) and LBDSU can be obtained upon request, subject to institutional approval.
5. We used the Montreal Neuroimaging Institute MNI152 template for image processing purposes, and the template can be downloaded at <a href="http://www.bic.mni.mcgillca/ServicesAtlases/ICBM152NLin2009">http://www.bic.mni.mcgillca/ServicesAtlases/ICBM152NLin2009</a>.

We have provided the patient IDs used for this study to reproduce our results.

## Steps to use the pre-trained model checkpoint with the downloaded datasets
To maintain data consistency, we have converted all the datasets to the [Uniform Data Set (UDS)](https://github.com/vkola-lab/nmed2024/tree/main/data/datasets/example_conversion_scripts/UDS_v3.pdf) format. Example scripts are provided to help you convert the downloaded datasets to this required format.

Please refer to [FHS conversion script](https://github.com/vkola-lab/nmed2024/tree/main/data/datasets/example_conversion_scripts/fhs_to_uds.ipynb) or [ADNI conversion script](https://github.com/vkola-lab/nmed2024/tree/main/data/datasets/example_conversion_scripts/adni_to_uds.py) for converting the datasets to UDS format. Before testing the pre-trained model checkpoint, please use the [uds_to_model_input.pkl](https://github.com/vkola-lab/nmed2024/tree/main/data/datasets/example_conversion_scripts/uds_to_model_input.pkl) dictionary to convert the dataset to the required format.

We have also open sourced our final model checkpoint on [Hugging Face](https://huggingface.co/spaces/vkola-lab/nmed2024). 