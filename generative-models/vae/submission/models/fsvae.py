import torch
import torch.utils.data
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

if 'solution' in script_directory:
    from solution import utils as ut
    from solution.models import nns
else:
    from submission import utils as ut
    from submission.models import nns

from torch import nn

class FSVAE(nn.Module):
    def __init__(self, nn='v2', name='fsvae'):
        super().__init__()
        self.name = name
        self.z_dim = 10
        self.y_dim = 10
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.y_dim) # from ./nns/v2.py
        self.dec = nn.Decoder(self.z_dim, self.y_dim) # from ./nns/v2.py

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x, y):
        """
            Computes the Evidence Lower Bound, KL and, Reconstruction costs

            Args:
                x: tensor: (batch, dim): Observations
                y: tensor: (batch, y_dim): Labels

            Returns:
                nelbo: tensor: (): Negative evidence lower bound
                kl_z: tensor: (): ELBO KL divergence to prior for latent variable z
                rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that we are interested in the ELBO of ln p(x | y)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   nelbo, kl_z, rec
        ################################################################################
        ### START CODE HERE ###
        # Compute the approximate posterior parameters q(z|x,y)
        z_mu, z_var = self.enc(x, y)  # Mean and log-variance of q(z|x,y)
        z_logvar = torch.log(z_var)

        # Compute KL divergence between q(z|x,y) and p(z)
        kl_per_sample = 0.5 * torch.sum((z_var) + z_mu**2 - 1 - z_logvar, dim=1)
        kl_z = kl_per_sample.sum()  # Sum over the batch to get a scalar

        # Sample z from q(z|x,y) using the reparameterization trick
        z = ut.sample_gaussian(z_mu, z_var)  # Reparameterized sample

        # Compute the decoder output μθ(y, z)
        x_recon = self.dec(y, z)  # Reconstructed x

        # Compute the reconstruction loss using the fixed variance σ² = 1/10
        rec_per_sample = 5 * torch.sum((x - x_recon) ** 2, dim=1)  # 5 = 1/(2*(1/10))
        rec = rec_per_sample.sum()  # Sum over the batch to get a scalar

        # Compute the Negative ELBO
        nelbo = kl_z + rec  # NELBO = KL divergence + Reconstruction loss

        return nelbo, kl_z, rec
        ### END CODE HERE ###

        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

    def loss(self, x, y):
        nelbo, kl_z, rec = self.negative_elbo_bound(x, y)
        loss = nelbo

        summaries = dict((
            ('train/loss', loss),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def compute_mean_given(self, z, y):
        return self.dec(z, y)

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                  self.z_prior[1].expand(batch, self.z_dim))
