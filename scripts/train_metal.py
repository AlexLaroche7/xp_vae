import torch
import numpy as np
import h5py
import sys
from astropy.table import Table

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using %s' % device)

lr = 1e-4
lr_min = 1e-10
batch_size = 1024
epochs = 5000
cos_anneal_t0 = 500
check_every_n_epochs = 500
num_data = int(1e7)

sys.path.append('..')
from xp_vae.abundance_model import ScatterVAEwithAbundancePrediction
model = ScatterVAEwithAbundancePrediction().to(device)
model.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

ANDRAE23_RED_GIANTS_XP = '/geir_data/scr/alaroche/CEMP_DATA/xp_andrae23.h5'
f_xp = h5py.File(ANDRAE23_RED_GIANTS_XP,'r')

ANDRAE23_RED_GIANTS_ABUNDANCES = '/geir_data/scr/alaroche/CEMP_DATA/table_2_catwise.fits'
f_ab = Table.read(ANDRAE23_RED_GIANTS_ABUNDANCES)

# for now, just use the first num_data for a quick train
num_data = int(1e7)

# Convert mag to flux
# Reference: https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photProc/cu5pho_ssec_photCal.html
# https://www.cosmos.esa.int/web/gaia/dr3-passbands
phot_g_mean_flux = 10**( (25.6873668671 / 2.5) - f_ab['phot_g_mean_mag'].value[:num_data])[:,np.newaxis]
mh = f_ab['mh_xgboost'].value[:num_data]
# also create zero array for carbon
cfe = np.zeros_like(mh)
# combine carbon and metallicity
y = np.stack([mh,cfe],axis=-1)
y_err = np.zeros_like(y)

xp = f_xp['xp'][:num_data]/phot_g_mean_flux
xp_err = f_xp['xp_err'][:num_data]/phot_g_mean_flux

nan_mask = (np.sum(~np.isnan(xp),axis=-1).astype('bool'))*(np.sum(~np.isnan(xp_err),axis=-1).astype('bool'))
xp = xp[nan_mask]
xp_err = xp_err[nan_mask]
mh = mh[nan_mask]
cfe = cfe[nan_mask]

norm = np.load('../data/andrae23_norm.npz')
xp = (xp - norm['mu']) / norm['sig']
xp_err = xp_err / norm['sig']
y = (y - norm['mu_y']) / norm['sig_y']
y_err = y_err / norm['sig_y']

lr_scheduler = lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                                      T_0=cos_anneal_t0,
                                                                                      T_mult=1,
                                                                                      eta_min=lr_min,
                                                                                      last_epoch=-1,
                                                                                    )

model.fit(xp,xp_err,
          y,y_err,
          epochs=epochs,
          lr_scheduler=lr_scheduler,
          batch_size=batch_size,
          checkpoint_every_n_epochs=check_every_n_epochs,
          output_direc='../models/METAL_MODEL')

model.save('../models/METAL_MODEL')