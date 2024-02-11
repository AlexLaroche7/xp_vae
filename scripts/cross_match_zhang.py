import h5py
import tqdm
import numpy as np
from joblib import Parallel, delayed

# load XP/APOGEE cross-match
APOGEE_XP_XMATCH_PATH = '../data/xp_apogee_cat.h5'
f = h5py.File(APOGEE_XP_XMATCH_PATH,'r')['__astropy_table__']

source_ids = f['GAIAEDR3_SOURCE_ID'] # to cross-match with ZGR23 catalogs

for i in range(1,10):

    subcat_i = i
    # you will need to download the stellar parameters catalog of ZGR23 to run this script
    subcat_path = '../../zhang23/stellar_params_catalog_0%s.h5' % subcat_i
    g = h5py.File(subcat_path, 'r')
    zgr23_source_ids = g['gdr3_source_id']

    def cross_match_bool(i):
        if zgr23_source_ids[i] in source_ids:
            return True
        else:
            return False

    stellar_params_mask = np.array(Parallel(n_jobs=-1,prefer="threads")(delayed(cross_match_bool)(i) for i in tqdm.tqdm(range(len(zgr23_source_ids)))))
    zgr23_source_ids_to_keep = zgr23_source_ids[stellar_params_mask]
    stellar_params = g['gdr3_source_id'][stellar_params_mask]

    print('\nFound %s matches\n for %s' % (len(stellar_params),subcat_path))
    np.savez('../data/zgr23_cross_match/stellar_params_catalog_0%s_xmatch.npz' % subcat_i,
            stellar_params=stellar_params,
            source_ids=source_ids)