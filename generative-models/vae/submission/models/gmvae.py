import numpy as np
import torch
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

if 'solution' in script_directory:
    from solution import utils as ut
    from solution.models import nns
else:
    from submission import utils as ut
    from submission.models import nns

from torch import nn

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim) # from ./nns/v1.py
        self.dec = nn.Decoder(self.z_dim) # from ./nns/v1.py

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

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
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   nelbo, kl_z, rec
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        ### START CODE HERE ###
        m_mixture, v_mixture = prior

        # Step 1: Encode x into latent space to get q(z|x) parameters
        q_mu, q_var = self.enc(x)  # Posterior mean and log variance

        # Step 2: Sample z from the posterior q(z|x) using the reparameterization trick
        z = ut.sample_gaussian(q_mu, q_var)

        # Step 3: Compute the Reconstruction term (log p(x | z))
        x_reconstructed = torch.sigmoid(self.dec(z))  # Reconstructed x
        #rec_loss = ut.log_bernoulli_with_logits(x_reconstructed, x).sum() / x.size(0)
        # Here, if we use binary_cross_entropy both x and x_reconstructed should be in [0,1] range
        # because they are supposed to be probabilities, i.e. after applying sigmoid.
        rec_loss = nn.functional.binary_cross_entropy(x_reconstructed, x, reduction='none').sum(-1) # / x.size(0)

        # Step 4: Compute KL divergence between q(z|x) and p(z) (Mixture of Gaussians prior)
        
        # Compute log q(z|x) using log_normal (log of posterior)
        log_q_z_given_x = ut.log_normal(z, q_mu, q_var)

        # Compute log p(z) using log_normal_mixture (log of mixture of Gaussians prior)
        # Expand m_mixture to match the batch size of z
        m_mixture = m_mixture.expand(z.size(0), -1, -1)
        v_mixture = v_mixture.expand(z.size(0), -1, -1)

        log_p_z = ut.log_normal_mixture(z, m_mixture, v_mixture)

        # KL divergence (Monte Carlo approximation)
        kl_z = (log_q_z_given_x - log_p_z)

        # Step 5: Compute NELBO (Negative Evidence Lower Bound)
        #kl_z = kl_z.mean() #ut.log_mean_exp(kl_z,0)
        
        nelbo = rec_loss + kl_z
        #nelbo_bce = kl_z + rec_loss_bce

        return nelbo.mean(), kl_z.mean(), rec_loss.mean()
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
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        ### START CODE HERE ###

        # Duplicate inputs for importance weighting (replicate observations iw times)
        x_rep = ut.duplicate(x, iw)

        m_mixture, v_mixture = prior

        # Step 1: Encode x into latent space to get q(z|x) parameters
        q_mu, q_var = self.enc(x)  # Posterior mean and log variance

        q_mu = ut.duplicate(q_mu, iw)
        q_var = ut.duplicate(q_var, iw)

        # Step 2: Sample z from the posterior q(z|x) using the reparameterization trick
        z = ut.sample_gaussian(q_mu, q_var)

        # Step 3: Compute the Reconstruction term (log p(x | z))
        x_reconstructed = torch.sigmoid(self.dec(z))  # Reconstructed x
        #rec_loss = ut.log_bernoulli_with_logits(x_reconstructed, x).sum() / x.size(0)
        # Here, if we use binary_cross_entropy both x and x_reconstructed should be in [0,1] range
        # because they are supposed to be probabilities, i.e. after applying sigmoid

        # Here we exp
        
        #z = ut.duplicate(z, iw)
        #x_reconstructed = ut.duplicate(x_reconstructed, iw) 

        log_px_given_z = nn.functional.binary_cross_entropy(x_reconstructed, x_rep, reduction='none').sum(-1)
        
        # Step 4: Compute KL divergence between q(z|x) and p(z) (Mixture of Gaussians prior)
        
        # Compute log q(z|x) using log_normal (log of posterior)
        log_qz_given_x = ut.log_normal(z, q_mu, q_var)

        # Compute log p(z) using log_normal_mixture (log of mixture of Gaussians prior)
        # Expand m_mixture to match the batch size of z
        m_mixture = m_mixture.expand(z.size(0), -1, -1)
        v_mixture = v_mixture.expand(z.size(0), -1, -1)

        log_p_z = ut.log_normal_mixture(z, m_mixture, v_mixture)

        # KL divergence (Monte Carlo approximation)
        kl = (log_qz_given_x - log_p_z)

        # Compute the importance weights: log p(x, z) - log q(z | x)
        log_weights = log_p_z + log_px_given_z - log_qz_given_x  # (iw * batch)
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
        kl = kl.mean() #ut.log_mean_exp(kl, dim=1).mean()
        
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
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
