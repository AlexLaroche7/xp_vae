import h5py
from astropy.io import votable
import requests
import io
from tqdm import tqdm
import numpy as np

from joblib import Parallel, delayed
import multiprocessing

APOGEE_XP_XMATCH_PATH = 'data/xp_apogee_cat.h5'
f = h5py.File(APOGEE_XP_XMATCH_PATH,'r')['__astropy_table__']

from sklearn.model_selection import train_test_split
validation_split = 0.1 # default value

# split up xp data to match training (see xp_vae.model.fit), by index
ids = f['ids']
idx = np.arange(len(ids))
idx_train,idx_val = train_test_split(idx,test_size=validation_split,random_state=12345)

ids = ids[idx_val]

print('Downloading covariances for %s sources in validation...' % len(ids))

bp_covs = []
rp_covs = []

def get_corrs(gdr3_source_id):

    url = f"https://gea.esac.esa.int/data-server/data?ID=Gaia+DR3+{gdr3_source_id}&RETRIEVAL_TYPE=XP_CONTINUOUS"
    temp_data = requests.get(url).content
    votable_data = votable.parse(io.BytesIO(temp_data))
    gaia_xp_data = votable_data.get_first_table().array

    bp_corr_i = gaia_xp_data['bp_coefficient_correlations'][0].data
    rp_corr_i = gaia_xp_data['rp_coefficient_correlations'][0].data

    return bp_corr_i,rp_corr_i

num_cores = multiprocessing.cpu_count() - 1
results = Parallel(n_jobs=num_cores)(delayed(get_corrs)(gdr3_source_id) for gdr3_source_id in tqdm(ids))

bp_corrs = [result[0] for result in results]
rp_corrs = [result[1] for result in results]

np.savez('data/xp_corrs.npz',bp_corrs=bp_corrs,rp_corrs=rp_corrs)

print('Done!')