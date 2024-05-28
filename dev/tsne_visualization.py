#%%
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import argparse
from tqdm import tqdm
import json
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#%%
# load data
dat_file = '/home/skowshik/ADRD_repo/pipeline_v1/adrd_tool/data/training_cohorts/train_vld_test_split_updated/nacc_test_with_np_cli.csv'   
df = pd.read_csv(dat_file)

# load the embeddings
path = '/data_1/skowshik/DenseNet_config2_3way/'
data = []
labels = []
# print(df)
for mri in os.listdir(path):
    data.append(np.load(path + mri, mmap_mode='r')[0])
    # print(mri)
    # print(mri.split('@')[0][:-2] + '.zip')
    label = df.loc[df['mri_zip'] == mri.split('@')[0][:-2] + '.zip'][['NC', 'MCI', 'DE']]
    labels.append(np.array(label)[0])
    # print(labels)
    # break
data = np.array(data)
labels = np.array(labels)
print(labels.shape)
print(data.shape)
# print(data[0].shape)

# %%

scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)
tsne = TSNE(n_components=2, perplexity=30, n_iter=500, random_state=0) 
embeddings_2d = tsne.fit_transform(standardized_data)

#%%
embeddings_2d.shape

#%%
# embeddings_2d[:, 1].shape
%matplotlib inline

# Create a color mapping for your labels
color_mapping = {
    (1, 0, 0): 'red',   # nc=1, mic=0, de=0
    (0, 1, 0): 'green', # nc=0, mic=1, de=0
    (0, 0, 1): 'blue' # nc=0, mic=0, de=1
}

# Assign colors based on the labels
colors = [color_mapping[tuple(label)] for label in labels]

plt.figure(figsize=(5, 4))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, marker='o', s=10)
plt.title('t-SNE Visualization of Image Embeddings')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

# %%