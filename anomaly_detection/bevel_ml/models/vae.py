# https://github.com/AntixK/PyTorch-VAE/blob/master/experiment.py を参照

import os
from typing import Callable, List, TypeVar, override

import torch
import torchvision.utils as vutils
from torch import nn, optim
from torch.nn import functional as F

from bevel_ml.models.base import BaseDetector

Tensor = TypeVar('torch.tensor')

# from bevel_ml.models.vae import BaseVAE

class VanillaVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: List = None,
        data_size: int = None,
        **kwargs) -> None:
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.last_size = data_size

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
            
        for _ in range(len(hidden_dims)):
            self.last_size = self.last_size // 2

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels=h_dim,
                        kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*self.last_size**2, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*self.last_size**2, latent_dim)


        # Build Decoder
        modules = []
        
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.last_size**2)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride = 2,
                        padding=1,
                        output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(
                                hidden_dims[-1],
                                hidden_dims[-1],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(
                                hidden_dims[-1], out_channels= 3,
                                kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, self.last_size, self.last_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(
        self,
        *args,
        **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(
        self,
        num_samples:int,
        current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class VAEDetector(BaseDetector):
    def __init__(
        self,
        in_channels: int=3,
        latent_dim: int=128,
        kld_weight: float=None,
        lr: float=None,
        weight_decay: float=None,
        LR_2: float=None,
        submodel=None, 
        scheduler_gamma: float=None,
        scheduler_gamma_2: float=None,
        transform: Callable | None = None,
        data_size: int = None,
        **kwargs) -> None:
        super().__init__(transform=transform)
        
        self.model = VanillaVAE(
            in_channels=in_channels,
            latent_dim=latent_dim,
            data_size=data_size
        )
        self.kld_weight = kld_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.LR_2 = LR_2
        self.submodel = submodel
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_gamma_2 = scheduler_gamma_2
        # self.curr_device = None
    
    # def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
    #     return self.model(input, **kwargs)
    
    @override
    def training_step(self, batch, batch_idx):
        # real_img, labels = batch
        
        # results = self.forward(real_img, labels=labels)
        results = self.forward(batch)
        train_loss = self.model.loss_function(
            *results,
            M_N=self.kld_weight)
        
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        
        return train_loss['loss']
    
    @override
    def validation_step(self, batch, batch_idx):
        # real_img, labels = batch
        
        # results = self.forward(real_img, labels=labels)
        results = self.forward(batch)
        val_loss = self.model.loss_function(
            *results,
            M_N=1.0)
        
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        
    @override
    def predict_step(self, batch, batch_idx):
        x, y = batch
        
        recons, real_image, _, _ = self.model(x)
        recons_loss = F.mse_loss(recons, real_image, reduction="none")
        recons_loss = torch.mean(recons_loss, dim=(1, 2, 3)).cpu().detach()
        
        outputs = {
            "loss": [None],
            "pred_scores": recons_loss,
            "label": y,
        }
        
        return outputs
        
    # def on_validation_end(self) -> None:
    #     self.sample_images()
        
    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay)
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.LR_2 is not None:
                optimizer2 = optim.Adam(
                    getattr(self.model,self.submodel).parameters(),
                    lr=self.LR_2)
                optims.append(optimizer2)
        except Exception:
            pass
        try:
            if self.scheduler_gamma is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(
                    optims[0],
                    gamma = self.scheduler_gamma)
                scheds.append(scheduler)
                # Check if another scheduler is required for the second optimizer
                try:
                    if self.scheduler_gamma_2 is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(
                            optims[1],
                            gamma = self.scheduler_gamma_2)
                        scheds.append(scheduler2)
                except Exception:
                    pass
                return optims, scheds
        except Exception:
            return optims
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.val_dataloader()))
        test_input = test_input.to("cuda")
        test_label = test_label.to("cuda")

#         test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(
            recons.data,
            os.path.join(
                self.logger.log_dir , 
                "Reconstructions", 
                f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
            normalize=True,
            nrow=12)
        try:
            samples = self.model.sample(
                144,
                "cuda",
                labels = test_label)
            vutils.save_image(
                samples.cpu().data,
                os.path.join(
                    self.logger.log_dir , 
                    "Samples",      
                    f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                normalize=True,
                nrow=12)
        except Warning:
            pass