import numpy as np
import h5py
import torch
import sys

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Using %s' % device)
torch.cuda.empty_cache()

APOGEE_XP_XMATCH_PATH = '../data/xp_apogee_cat.h5'
f = h5py.File(APOGEE_XP_XMATCH_PATH,'r')['__astropy_table__']

# flux normalization
g_flux = f['phot_g_mean_flux'][:,np.newaxis]
xp = f['coeffs']/g_flux
xp_err = f['coeff_errs']/g_flux

# standard normalization
norm = np.load('../data/apogee_norm.npz')
xp_norm = (xp - norm['mu']) / norm['sig']
xp_err_norm = xp_err / norm['sig']

from sklearn.model_selection import train_test_split
validation_split = 0.1 # default value

# split up xp data to match training (see xp_vae.model.fit), by index
idx = np.arange(len(xp))
idx_train,idx_val = train_test_split(idx,test_size=validation_split,random_state=12345)

sys.path.append('..')
from xp_vae.model import ScatterVAE

model = ScatterVAE().to(device)
weights = torch.load('../models/APOGEE_MODEL/weight.pt',map_location=device)
model.load_state_dict(state_dict=weights['model_state_dict'])
model.eval()

in1 = torch.from_numpy(xp_norm).cpu().to(device)
in2 = torch.from_numpy(xp_err_norm).to(device)

_,_,mu,log_var,est,s = model.forward(in1,in2)

est = est.detach().cpu().numpy()*norm['sig']+norm['mu']

import tensorflow as tf

zhang_nn_model = tf.saved_model.load("../../stellar_flux_model")
zhang_sample_wavelengths = zhang_nn_model._sample_wavelengths

from gaiaxpy.core.config import _load_xpmerge_from_xml, _load_xpsampling_from_xml
xp_sampling_grid, xp_merge = _load_xpmerge_from_xml()
xp_design_matrices = _load_xpsampling_from_xml()

wavelength_mask_xp = np.array([lam in zhang_sample_wavelengths.numpy() for lam in xp_sampling_grid]) 

xp_wavelength_space = np.zeros((len(xp),sum(wavelength_mask_xp)))
vae_wavelength_space = np.zeros((len(xp),sum(wavelength_mask_xp)))
xp_err_wavelength_space = np.zeros((len(xp),sum(wavelength_mask_xp)))

import tqdm 
from joblib import Parallel, delayed

from numpy.typing import NDArray
# from astroNN_stars_foundation.utils.gaia_utils
def xp_coeffs_phys(bprp_coeffs: dict) -> NDArray[np.float64]:
    """
    Turn the coefficients into physical spectra
    """
    bp_spec = bprp_coeffs["bp"].dot(xp_design_matrices["bp"])
    rp_spec = bprp_coeffs["rp"].dot(xp_design_matrices["rp"])
    return np.add(
        np.multiply(bp_spec, xp_merge["bp"]), np.multiply(rp_spec, xp_merge["rp"])
    )

def xp_to_lam(i):
    xp_i = xp[i] * g_flux[i]
    dict_ = {'bp':xp_i[:55],'rp':xp_i[55:]}
    return xp_coeffs_phys(dict_)[wavelength_mask_xp]

xp_wavelength_space = np.array(Parallel(n_jobs=-1)(delayed(xp_to_lam)(i) for i in tqdm.tqdm(range(len(xp)))))
np.save('../data/xp_wavelength_space.npy',xp_wavelength_space)

def vae_to_lam(i):
    xp_i = est[i] * g_flux[i]
    dict_ = {'bp':xp_i[:55],'rp':xp_i[55:]}
    return xp_coeffs_phys(dict_)[wavelength_mask_xp]

vae_wavelength_space = np.array(Parallel(n_jobs=-1)(delayed(vae_to_lam)(i) for i in tqdm.tqdm(idx_val)))
np.save('../data/vae_wavelength_space.npy',vae_wavelength_space)

def err_to_lam(i):
    err_i = xp_err[i] * g_flux[i]
    dict_ = {'bp':err_i[:55],'rp':err_i[55:]}
    return xp_coeffs_phys(dict_)[wavelength_mask_xp]

err_wavelength_space = np.array(Parallel(n_jobs=-1)(delayed(err_to_lam)(i) for i in tqdm.tqdm(idx_val)))
np.save('../data/err_wavelength_space.npy',err_wavelength_space)
