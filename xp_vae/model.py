import os
import json
import tqdm
import torch
import pathlib
import numpy as np
from torch import nn
from datetime import timedelta
from numpy.typing import NDArray
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split

from xp_vae.utils import DataGenerator

class ScatterVAE(nn.Module):
    """
    A variational auto-encoder which generates both reconstructions and intrinsic scatter estimates for Gaia XP spectra
    """

    def __init__(self,
                 input_dim: int = 110,
                 latent_dim: int = 6,
                 intermediate_layers: List = [90,70,50,30,10],
                 device: str = 'cuda',
                 mixed_precision: bool = False
        ):
        super(ScatterVAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.intermediate_layers = intermediate_layers
        self.device_type = device
        self.mixed_precision = mixed_precision

        ### MODEL ARCHITECTURE ###

        # encoder
        encoder_layers = []
        curr_input_dim = self.input_dim

        for output_dim in self.intermediate_layers:
            encoder_layers.append(
                nn.Sequential(
                    nn.Linear(curr_input_dim, output_dim),
                    nn.BatchNorm1d(output_dim),
                    nn.GELU()
                )
            )
            curr_input_dim = output_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(self.intermediate_layers[-1], self.latent_dim)
        self.log_var = nn.Linear(self.intermediate_layers[-1], self.latent_dim)
    
        # decoder
        decoder_layers = []
        decoder_in = nn.Linear(self.latent_dim, self.intermediate_layers[-1])
        flip_layers = self.intermediate_layers[::-1]

        for i in range(len(flip_layers)-1):
            decoder_layers.append(
                nn.Sequential(
                    nn.Linear(flip_layers[i], flip_layers[i+1]),
                    nn.BatchNorm1d(flip_layers[i+1]),
                    nn.GELU()
                )
            )
        decode_out = nn.Sequential(
                nn.Linear(flip_layers[-1], self.input_dim)
        )
        self.decoder = nn.Sequential(*([decoder_in]+decoder_layers+[decode_out]))
        
        # scatter
        scatter_layers = []
        scatter_in = nn.Linear(self.latent_dim, self.intermediate_layers[-1])

        for i in range(len(flip_layers)-1):
            scatter_layers.append(
                nn.Sequential(
                    nn.Linear(flip_layers[i], flip_layers[i+1]),
                    nn.BatchNorm1d(flip_layers[i+1]),
                    nn.GELU()
                )
            )

        scatter_out = nn.Sequential(
            nn.Linear(flip_layers[-1], self.input_dim),
            nn.Sigmoid()
        )
        self.scatter = nn.Sequential(*([scatter_in]+scatter_layers+[scatter_out]))

    ### ANCILLARY FUNCTIONS ###

    def encode(self, input):
        encoding = self.encoder(input)
        return self.mu(encoding), self.log_var(encoding)
    
    def decode(self, latent):
        return self.decoder(latent)
    
    def intrinsic_scatter(self, latent):
        return self.scatter(latent)
    
    def reparam(self, mu, log_var):
        sig = torch.exp(0.5*log_var)
        eps = torch.randn_like(sig)
        return eps*sig+mu

    def forward(self, x, x_err):
        mu, log_var = self.encode(x)
        latent = self.reparam(mu, log_var)
        xhat = self.decode(latent)
        shat = self.intrinsic_scatter(latent)
        return x, x_err, mu, log_var, xhat, shat

    ### LOSS FUNCTIONS ###

    def get_loss(self, *forward):
        x, x_err, mu, log_var, xhat, shat = forward
        wmse = self.get_weighted_mse_loss(x, x_err, xhat, shat)
        kld = self.get_kld(mu, log_var)
        loss = wmse+kld
        return loss, wmse, kld

    def get_weighted_mse_loss(self, x, x_err, xhat, shat):
        weight = x_err**2. + shat**2.
        return torch.mean(torch.sum(0.5*(x-xhat)**2./weight + 0.5*torch.log(weight), dim=1), dim=0)
    
    def get_kld(self, mu, log_var):
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu**2. - log_var.exp(), dim=1), dim=0)
    
    ### TRAINING FUNCTION ###

    def fit(self,
            xp: NDArray,
            xp_err: NDArray,
            batch_size: int = 512,
            val_batchsize_factor: int = 5,
            epochs: int = 64,
            validation_split: float = 0.1,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
            checkpoint_every_n_epochs: int = 0, # no checkpoints if 0
            terminate_on_nan: bool = True,
            output_direc: str = '/geir_data/scr/alaroche/xp_vae',
        ) -> None:
        
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.device_type == 'cuda')
        self.epochs = epochs
        self.output_direc = output_direc
        
        pathlib.Path(f'%s/checkpoints' % self.output_direc).mkdir(parents=True,exist_ok=True)

        train_log = open(f'%s/train.log' % self.output_direc, 'w')
        train_log.write(f'Batch Size: %s\n' % batch_size)
        train_log.write("====================================\n")

        train_metrics = open(f'%s/train_metrics.csv' % self.output_direc,'w')
        train_metrics.write('time,loss,mse_loss,kld_loss,val_loss,val_mse_loss,val_kld_loss,lr\n')
        
        xp_train,xp_val,xp_err_train,xp_err_val = train_test_split(xp,xp_err,test_size=validation_split,random_state=12345) # for reproducibility of train/test split

        train_gen = DataGenerator(batch_size=batch_size,xp=xp_train,xp_err=xp_err_train)
        val_gen = DataGenerator(batch_size=batch_size*val_batchsize_factor,xp=xp_val,xp_err=xp_err_val)

        scheduler = lr_scheduler(self.optimizer)

        ### TRAINING ###
        t=0
        with tqdm.tqdm(range(epochs),unit='epoch') as pbar:
            for epoch in pbar:
                self.epoch = epoch+1
                train_log.write(f'Epoch %s/%s:' % (self.epoch,self.epochs))

                self.train()
                running_loss,running_mse,running_kld=0.,0.,0.
                
                for batch_num, (xp,xp_err) in enumerate(train_gen):
                    self.optimizer.zero_grad()
                    with torch.autocast(device_type=self.device_type,enabled=self.mixed_precision):
                        forward = self.forward(torch.from_numpy(xp).to(self.device_type),torch.from_numpy(xp_err).to(self.device_type))
                        loss,mse,kld = self.get_loss(*forward)

                        grad_scaler.scale(loss).backward()
                        grad_scaler.step(self.optimizer)
                        grad_scaler.update()

                        running_loss += loss.item()
                        running_mse += mse.item()
                        running_kld += kld.item()
                
                last_loss = running_loss/(batch_num+1)
                last_mse = running_mse/(batch_num+1)
                last_kld = running_kld/(batch_num+1)
                train_gen.epoch_end()
                
                # epoch validation 
                self.eval()
                running_val_loss,running_val_mse,running_val_kld=0.,0.,0.
                with torch.inference_mode():
                    for batch_num, (xp,xp_err) in enumerate(val_gen):
                        forward = self.forward(torch.from_numpy(xp).to(self.device_type),torch.from_numpy(xp_err).to(self.device_type))
                        vloss,vmse,vkld = self.get_loss(*forward)

                        running_val_loss += vloss.item()
                        running_val_mse += vmse.item()
                        running_val_kld += vkld.item()

                avg_val_loss = running_val_loss/(batch_num+1)
                avg_val_mse = running_val_mse/(batch_num+1)
                avg_val_kld = running_val_kld/(batch_num+1)

                self.loss = last_loss
                self.mse = last_mse
                self.kld = last_kld
                self.val_loss = avg_val_loss
                self.val_mse = avg_val_mse
                self.val_kld = avg_val_kld
                self.learning_rate = self.optimizer.param_groups[-1]['lr']
                val_gen.epoch_end()

                # end of epoch
                scheduler.step()

                # record epoch results in files
                lr_fmt = np.format_float_scientific(self.learning_rate, precision=4, unique=False)
                curr_t = pbar.format_dict['elapsed'] - t
                t = pbar.format_dict['elapsed']

                train_log.write(f' elapsed: %s s - rate: %.2f s - loss: %.4f - mse: %.4f - kld: %.4f - val_loss: %.4f - val_mse: %.4f - val_kld: %.4f - lr: %s\n'
                                % (str(timedelta(seconds=t)), curr_t, last_loss, last_mse, last_kld, avg_val_loss, avg_val_mse, avg_val_kld, lr_fmt))
                train_log.flush()

                train_metrics.write(f'%s,%s,%s,%s,%s,%s,%s,%s' % (curr_t, last_loss, last_mse, last_kld, avg_val_loss, avg_val_mse, avg_val_kld, lr_fmt))
                train_metrics.flush()

                if terminate_on_nan and np.isnan(last_loss):
                    raise ValueError('Nan loss, training terminated!')
                
                if checkpoint_every_n_epochs > 0:
                    if self.epoch % checkpoint_every_n_epochs == 0 or self.epoch == 1:
                        self.save(model_direc=f'%s/checkpoints/epoch_%s' % (self.output_direc,self.epoch))
        
        train_log.close()
        train_metrics.close()

    ### SAVING MODEL ###

    def save(self, model_direc: str = 'model') -> None:
        pathlib.Path(model_direc).mkdir(parents=True,exist_ok=True)
        json_path = f'%s/config.json' % model_direc
        if not os.path.exists(json_path):
            with open(json_path,'w') as f:
                json.dump(self.get_config(), f, indent=4)
            self.save_model(model_direc=model_direc)
        else:
            raise FileExistsError('Input direc already has model in it! change model direc')
    
    def get_config(self) -> Dict[str, Any]:

        nn_config = {'latent_dim': self.latent_dim,
                     'input_dim': self.input_dim,
                     'intermediate_layers': self.intermediate_layers,
                     'device': self.device_type,
                     'mixed_precision':self.mixed_precision}
        
        return {'nn_config': nn_config}
    
    def save_model(self, model_direc: str):

        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'optimizer': self.optimizer.__class__.__name__,
                    'epoch': self.epoch},
                    f'%s/weight.pt' % model_direc)