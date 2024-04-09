import h5py
import tqdm
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from numba import cuda

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print('Using %s' % device)
torch.cuda.empty_cache()

# get source ids from XP/APOGEE cross-match
APOGEE_XP_XMATCH_PATH = '../data/xp_apogee_cat.h5'
f = h5py.File(APOGEE_XP_XMATCH_PATH,'r')['__astropy_table__']
source_ids = f['ids']

print('Number of source ids: %s' % len(source_ids))

# only keep test indices from the above
from sklearn.model_selection import train_test_split
validation_split = 0.1 # default value

# split up xp data to match training (see xp_vae.model.fit), by index
idx = np.arange(len(source_ids))
idx_train,idx_val = train_test_split(idx,test_size=validation_split,random_state=12345)

source_ids_subset = source_ids[idx_val] # to cross-match with ZGR23 catalogs

print('Number of source ids in test subset: %s\n' % len(source_ids_subset))

stellar_params_subset = np.zeros((len(source_ids_subset),5))
quality_flags_subset = np.zeros(len(source_ids_subset))

for j in range(10):
    subcat_path = '/yngve_data/zhang23/stellar_params_catalog_0%s.h5' % j
    g = h5py.File(subcat_path, 'r')
    zgr23_source_ids = g['gdr3_source_id']
    stellar_params_est = np.array(g['stellar_params_est'])
    quality_flags = g['quality_flags']

    cross_idxs = np.argwhere(np.isin(zgr23_source_ids,source_ids_subset))
    print('Number of cross-matches:',len(cross_idxs))

    for i in tqdm.tqdm(range(len(cross_idxs))):
        test_idx = np.argwhere(source_ids_subset==zgr23_source_ids[cross_idxs[i]])[0]
        stellar_params_subset[test_idx] = stellar_params_est[cross_idxs[i]][0]
        quality_flags_subset[test_idx] = quality_flags[cross_idxs[i]][0]

    print('Done %s\n' % subcat_path)

np.savez('/geir_data/scr/alaroche/xp_vae/data/zhang_stellar_params_xmatch.npz',
        source_ids=source_ids_subset,
        stellar_params=stellar_params_subset,
        quality_flags=quality_flags_subset)