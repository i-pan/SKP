import glob
import numpy as np
import os

from tqdm import tqdm


data_dir = "/home/ian/datasets/totalsegmentator/extracted_embeddings_for_organ_classification_with_augs_v2b0/fold0/"
features = glob.glob(data_dir + "*_features.npy")

for f in tqdm(features):
    features = np.load(f)
    for i in range(len(features)):
        fp = f.replace("_features.npy", f"_{i:03d}_features.npy")
        np.save(fp, features[i])
    os.system(f"rm {f}")
