import torch
import numpy as np
import h5py
import sys
from astropy.table import Table
from sklearn.model_selection import train_test_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using %s' % device)

ANDRAE23_RED_GIANTS_XP = '/geir_data/scr/alaroche/CEMP_DATA/xp_andrae23.h5'
f_xp = h5py.File(ANDRAE23_RED_GIANTS_XP,'r')

ANDRAE23_RED_GIANTS_ABUNDANCES = '/geir_data/scr/alaroche/CEMP_DATA/table_2_catwise.fits'
f_ab = Table.read(ANDRAE23_RED_GIANTS_ABUNDANCES)

# Convert mag to flux
# Reference: https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photProc/cu5pho_ssec_photCal.html
# https://www.cosmos.esa.int/web/gaia/dr3-passbands
phot_g_mean_flux = 10**( (25.6873668671 / 2.5) - f_ab['phot_g_mean_mag'])[:,np.newaxis]
mh = f_ab['mh_xgboost']
# also create zero array for carbon
cfe = np.zeros_like(mh)

xp = f_xp['xp']/phot_g_mean_flux
xp_err = f_xp['xp_err']/phot_g_mean_flux

nan_mask = (np.sum(~np.isnan(xp),axis=-1).astype('bool'))*(np.sum(~np.isnan(xp_err),axis=-1).astype('bool'))
xp = xp[nan_mask]
xp_err = xp_err[nan_mask]
mh = mh[nan_mask]
cfe = cfe[nan_mask]
# combine carbon and metallicity
y = np.stack([mh,cfe],axis=-1)
y_err = np.zeros_like(y)

# Split data into same training and validation set that will be used
RANDOM_STATE = 12345
validation_split = 0.1
xp_train,_,_,_ = train_test_split(xp,xp_err,test_size=validation_split,random_state=RANDOM_STATE)
y_train,_,_,_ = train_test_split(y,y_err,test_size=validation_split,random_state=RANDOM_STATE)

# Compute mean and std of training set
xp_mu = np.mean(xp_train,axis=0)
xp_sig = np.std(xp_train,axis=0)
y_mu = np.mean(y_train,axis=0)
y_sig = np.std(y_train,axis=0)

# Save means and stds
np.savez('../data/andrae23_norm.npz',mu=xp_mu,sig=xp_sig,mu_y=y_mu,sig_y=y_sig)
