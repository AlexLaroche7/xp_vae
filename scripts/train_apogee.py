import torch
import numpy as np
import h5py
import sys

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
from xp_vae.model import ScatterVAE
model = ScatterVAE().to(device)
model.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

APOGEE_XP_XMATCH_PATH = '../data/xp_apogee_cat.h5'
f = h5py.File(APOGEE_XP_XMATCH_PATH,'r')['__astropy_table__']

g_flux = f['phot_g_mean_flux'][:,np.newaxis]
xp = f['coeffs']/g_flux
xp_err = f['coeff_errs']/g_flux

nan_mask = (np.sum(~np.isnan(xp),axis=-1).astype('bool'))*(np.sum(~np.isnan(xp_err),axis=-1).astype('bool'))
xp = xp[nan_mask]
xp_err = xp_err[nan_mask]

norm = np.load('../data/apogee_norm.npz')
xp = (xp - norm['mu']) / norm['sig']
xp_err = xp_err / norm['sig']

lr_scheduler = lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                                      T_0=cos_anneal_t0,
                                                                                      T_mult=1,
                                                                                      eta_min=lr_min,
                                                                                      last_epoch=-1,
                                                                                    )

model.fit(xp,xp_err,
          epochs=epochs,
          lr_scheduler=lr_scheduler,
          batch_size=batch_size,
          checkpoint_every_n_epochs=check_every_n_epochs,
          output_direc='../models/APOGEE_MODEL')

model.save('../models/APOGEE_MODEL')