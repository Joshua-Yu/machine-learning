import torch
from torch import nn
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

if 'solution' in script_directory:
    from solution import utils as ut
    from solution.models import nns
else:
    from submission import utils as ut
    from submission.models import nns

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim) # from ./nns/v1.py
        self.dec = nn.Decoder(self.z_dim) # from ./nns/v1.py

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   nelbo, kl, rec
        ################################################################################
        ### START CODE HERE ###

        # Forward pass through the encoder to get the latent mean and variance
        mean, var = self.enc(x)

        # Reparameterization trick: Sample latent vector z from N(0, 1) using the
        # mean and log variance output from the encoder
        z = ut.sample_gaussian(mean, var)

        # Forward pass through the decoder to reconstruct the input from the latent variable.
        # Because x_reconstructed is in [0, 1], a sigmoid is applied.
        x_reconstructed = torch.sigmoid(self.dec(z))
        #x_reconstructed = self.dec(z)

        # Reconstruction loss (Binary Cross-Entropy/BCE if x is binary, MSE otherwise)

        rec = nn.functional.binary_cross_entropy(x_reconstructed, x, reduction='none').sum(-1)#/ x.size(0)
        #rec_mse = -1 * nn.functional.mse_loss(x_reconstructed, x, reduction="sum") / x.size(0) #-1 * ut.log_bernoulli_with_logits(x_reconstructed, x).mean() / x.size(0)
        #rec = -1 * ut.log_bernoulli_with_logits(x_reconstructed, x)

        # KL Divergence between q(z|x) and the prior p(z) ~ N(0, I)
        kl = ut.kl_normal(mean, var, self.z_prior_m, self.z_prior_v)
        #kl = kl.mean()

        #kl = torch.mean(-0.5 * torch.sum(1 + torch.log(var) - mean ** 2 - var, dim = 1), dim = 0)

        # Negative ELBO is the sum of KL and reconstruction loss
        nelbo = rec + kl
        nelbo = nelbo.mean()

        return nelbo, kl.mean(), rec.mean()

        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   niwae, kl, rec
        #
        # HINT: The summation over m may seem to prevent us from 
        # splitting the ELBO into the KL and reconstruction terms, but instead consider 
        # calculating log_normal w.r.t prior and q
        ################################################################################
        ### START CODE HERE ###

        # Duplicate inputs for importance weighting (replicate observations iw times)
        x_rep = ut.duplicate(x, iw)
        #x_rep = x.unsqueeze(1).expand(x.size(0), iw, x.size(1))

        # Prior parameters (mean, std) for latent space
        prior_mean, prior_logvar = self.z_prior

        # Get the variational distribution parameters (mu, logvar)
        qz_mean, qz_var = self.enc(x)  # Outputs mean and logvar of q(z|x)

        qz_mean = ut.duplicate(qz_mean, iw)
        qz_var = ut.duplicate(qz_var, iw)

        # Sample z from the variational posterior q(z|x) using reparameterization trick
        z_samples = ut.sample_gaussian(qz_mean, qz_var) # z_samples shape: (iw, batch, latent_dim)


        #z_samples = ut.duplicate(z_samples, iw)

        # Compute log likelihood p(x|z) and log prior p(z) for each z sample
        x_rep_recon = torch.sigmoid(self.dec(z_samples))  # (iw * batch)
        log_px_given_z = nn.functional.binary_cross_entropy(x_rep_recon, x_rep, reduction='none').sum(-1) 

        kl = ut.kl_normal(qz_mean, qz_var, self.z_prior_m, self.z_prior_v) 

        log_pz = ut.log_normal(z_samples, prior_mean, prior_logvar)  # (iw * batch)
        
        # Compute log variational posterior q(z|x)
        log_qz_given_x = ut.log_normal(z_samples, qz_mean, qz_var)  # (iw * batch)
        
        # Compute the importance weights: log p(x, z) - log q(z | x)
        log_weights = -(log_pz + log_px_given_z - log_qz_given_x)  # (iw * batch)
        log_weights = log_px_given_z + kl
        
        # Reshape log weights to (iw, batch)
        log_weights = log_weights.view(iw, x.size(0))  # (iw, batch)
        log_weights = torch.transpose(log_weights, 0, 1)    # (batch, iw)
        
        # Compute log mean of importance weights for IWAE bound (numerically stable)
        log_iw_mean = ut.log_mean_exp(-log_weights, dim=1)  # (batch)
        
        # Negative IWAE bound
        niwae = -torch.mean(log_iw_mean)
        #niwae = log_iw_mean

        # ELBO Decomposition: KL 
        kl = kl.view(iw, x.size(0))  # (iw, batch)
        kl = torch.transpose(kl, 0, 1)    # (batch, iw)
        kl = ut.log_mean_exp(kl, dim=1).mean()
        
        # Reconstruction term
        rec = log_px_given_z.view(iw, x.size(0))  
        rec = torch.transpose(rec, 0, 1)    # (batch, iw)
        rec = ut.log_mean_exp(rec, dim=1).mean()

        return niwae, kl, rec

        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

    def negative_iwae_bound1(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   niwae, kl, rec
        #
        # HINT: The summation over m may seem to prevent us from 
        # splitting the ELBO into the KL and reconstruction terms, but instead consider 
        # calculating log_normal w.r.t prior and q
        ################################################################################
        ### START CODE HERE ###

        # Duplicate inputs for importance weighting (replicate observations iw times)
        x_rep = ut.duplicate(x, iw)
        #x_rep = x.unsqueeze(1).expand(x.size(0), iw, x.size(1))

        # Get the variational distribution parameters (mu, logvar)
        qz_mean, qz_var = self.enc(x)  # Outputs mean and logvar of q(z|x)

        qz_mean = ut.duplicate(qz_mean, iw)
        qz_var = ut.duplicate(qz_var, iw)

        # Sample z from the variational posterior q(z|x) using reparameterization trick
        z_samples = ut.sample_gaussian(qz_mean, qz_var) # z_samples shape: (iw, batch, latent_dim)

        # Compute log likelihood p(x|z) and log prior p(z) for each z sample
        x_rep_recon = torch.sigmoid(self.dec(z_samples))  # (iw * batch)
        log_px_given_z = x_rep_recon.sum(1) #nn.functional.binary_cross_entropy(x_rep_recon, x_rep, reduction='none').sum(-1) 

        log_pz = -0.5 * torch.sum(z_samples ** 2, dim=-1) #ut.log_normal(z_samples, prior_mean, prior_logvar)  # (iw * batch)
        
        # Compute log variational posterior q(z|x)
        sigma = torch.exp(0.5 * qz_var)
        log_qz_given_x = -0.5 * torch.sum(((z_samples - qz_mean) / sigma) ** 2 + qz_var, dim=-1)  # [batch, m]
        
        # Compute the importance weights: log p(x, z) - log q(z | x)
        log_weights = log_pz + log_px_given_z - log_qz_given_x  # (iw * batch)
        
        # Reshape log weights to (iw, batch)
        log_weights = log_weights.view(iw, x.size(0))  # (iw, batch)
        log_weights = torch.transpose(log_weights, 0, 1)    # (batch, iw)
        
        # Compute log mean of importance weights for IWAE bound (numerically stable)
        log_iw_mean = ut.log_mean_exp(log_weights, dim=1)  # (batch)
        
        # Negative IWAE bound
        niwae = torch.mean(log_iw_mean)
        #niwae = log_iw_mean

        # ELBO Decomposition: KL 
        kl = (-log_qz_given_x + log_pz).view(iw, x.size(0))  # (iw, batch)
        kl = torch.transpose(kl, 0, 1)    # (batch, iw)
        kl = ut.log_mean_exp(kl, dim=1).mean()
        
        # Reconstruction term
        rec = log_px_given_z.view(iw, x.size(0))  
        rec = torch.transpose(rec, 0, 1)    # (batch, iw)
        rec = ut.log_mean_exp(rec, dim=1).mean()

        return niwae, kl, rec

        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError    

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
