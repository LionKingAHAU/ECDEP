import os
import gzip
import numpy as np
import pandas as pd
import io
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

# You can change the path to your own configs
inPath = "result/"
outPath = "../data/Dynamic Network Demo/Input Data/"

# 1. Using all observed communities as initial features
all_communities = []
print("---------------STEP4---------------")
print("----Handling communities......")
for root, dirs, files in os.walk(inPath):
    for file_name in files:
        if file_name.startswith("strong") and file_name.endswith(".gz"):
            with gzip.open(os.path.join(root, file_name), 'rb') as f:
                file_data = f.read()
                if len(file_data) == 0:
                    print(f"Skipping empty file: {file_name}")
                    continue
                # Decompress the gzipped file and create a file-like object
                with io.BytesIO(file_data) as buffer:
                    community = pd.read_csv(buffer, sep="\t", header=None, encoding='utf-8')
                community_fea = np.zeros((len(np.load(outPath + "label.npy")), len(community)))
                for line in range(len(community)):
                    nodes = community.iloc[line][1][1:][:-1].split(', ')
                    for node in nodes:
                        community_fea[int(node)][line] = 1
                all_communities.append(community_fea)

concatenated_communities = np.concatenate(all_communities, axis=-1)
np.save("../data/Dynamic Network Demo/Input Data/raw_community.npy", concatenated_communities)

print("----Selecting communities......")
# 2. Use SVM-RFE to obtain better feature subset
class Data(object):
    def __init__(self):
        communities = concatenated_communities
        label = np.load(outPath + 'label.npy')
        self.num = len(label)
        label = label.reshape((self.num, 1))
        shuffle_index = np.random.permutation(self.num)
        self.X = communities[shuffle_index]
        self.y = label[shuffle_index].flatten()


data = Data()
X = data.X
y = data.y
nFeatures = 64

model = SVC(kernel='linear')
rfe = RFE(estimator=model, n_features_to_select=nFeatures)
X_selected = rfe.fit_transform(X, y)
np.save(outPath + "community_selected.npy", concatenated_communities[:, rfe.get_support(indices=True)])

# Now we get the subset community features, please head to the final part to predict essential proteins.
