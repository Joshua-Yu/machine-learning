import numpy as np
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
from torch.nn import functional as F

class SSVAE(nn.Module):
    def __init__(self, nn='v1', name='ssvae', gen_weight=1, class_weight=100):
        super().__init__()
        self.name = name
        self.z_dim = 64
        self.y_dim = 10
        self.gen_weight = gen_weight
        self.class_weight = class_weight
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.y_dim) # from ./nns/v1.py
        self.dec = nn.Decoder(self.z_dim, self.y_dim) # from ./nns/v1.py
        self.cls = nn.Classifier(self.y_dim)

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
            kl_z: tensor: (): ELBO KL divergence to prior for latent variable z
            kl_y: tensor: (): ELBO KL divergence to prior for labels y
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL_Z, KL_Y and Rec decomposition
        #
        # To assist you in the vectorization of the summation over y, we have
        # the computation of q(y | x) = `y_prob` and some tensor tiling code for you.
        #
        # Note that nelbo = kl_z + kl_y + rec
        #
        # Outputs should all be tensor scalars

        # Return:
        #   nelbo, kl_z, kl_y, rec

        # HINT 1: The function `kl_normal` and `kl_cat` will be useful from `utils.py`

        # HINT 2: Start by computing KL_Y, KL_Z, and the rec terms. Remember that for
        # KL_Z and rec we have the additional y_dim necessary for the "expectation" w/r
        # to the values of y.

        # HINT 3: When computing the expectation w/r to the distribution q(y | x),
        # KL_Z and rec have values **grouped by y_dim**, so you should reshape
        # to (y_dim,batch) NOT (batch, y_dim). You can take the transpose of y_prob
        # when calculating the expectation then.

        # HINT 4: Try making use of log_bernoulli_with_logits in your calculation of rec
        # from utils.py
        ################################################################################
        y_logits = self.cls(x)
        #y_logprob = F.log_softmax(y_logits, dim=1)
        #y_prob = torch.softmax(y_logprob, dim=1) # (batch, y_dim)
        y_prob = F.softmax(y_logits, dim=1)     # Convert logits to probabilities
        log_q_y = F.log_softmax(y_logits, dim=1)

        # Duplicate y based on x's batch size. Then duplicate x
        # This enumerates all possible combination of x with labels (0, 1, ..., 9)
        y = np.repeat(np.arange(self.y_dim), x.size(0))
        y = x.new(np.eye(self.y_dim)[y])
        x = ut.duplicate(x, self.y_dim)

        ### START CODE HERE ###
        # Assume a uniform prior over y (log_p = log(1 / y_dim) = -log(y_dim))
        log_p_y = torch.full_like(log_q_y, -torch.log(torch.tensor(self.y_dim, dtype=torch.float)))

        # Compute KL divergence for labels (KL_Y)
        kl_y = ut.kl_cat(q=y_prob, log_q=log_q_y, log_p=log_p_y)  # KL divergence for labels (batch,)

        # Compute the KL divergence for the latent variable z
        # q(z | x, y) is parameterized as a Gaussian distribution
        z_mu, z_logvar = self.enc(x, y)
        z_var = z_logvar #torch.exp(z_logvar)
        z = ut.sample_gaussian(z_mu, z_var)
        prior_mu = torch.zeros_like(z_mu)  # prior mean
        prior_var = torch.ones_like(z_var)  # prior variance
        kl_z = ut.kl_normal(qm=z_mu, qv=z_var, pm=prior_mu, pv=prior_var)  # (y_dim, batch)

        # Compute the reconstruction term
        x_logits = torch.sigmoid(self.dec(z, y))  # Reconstructed x logits
        rec = ut.log_bernoulli_with_logits(x, x_logits)  # (y_dim, batch)
        rec = -nn.functional.binary_cross_entropy(x_logits, x, reduction='none').sum(-1)#/ x.size(0)

        # Reshape for expectation over q(y | x)
        kl_z = kl_z.view(self.y_dim, -1)  # (y_dim, batch)
        rec = rec.view(self.y_dim, -1)  # (y_dim, batch)

        # Compute the expectation with respect to q(y | x) using y_prob
        exp_kl_z = torch.sum(kl_z * y_prob.T, dim=0)  # (batch,)
        exp_rec = -torch.sum(rec * y_prob.T, dim=0)  # (batch,)

        # Sum over all components to get negative ELBO
        nelbo = exp_kl_z + kl_y + exp_rec
        return nelbo.mean(), exp_kl_z.mean(), kl_y.mean(), exp_rec.mean()
        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

    def classification_cross_entropy(self, x, y):
        y_logits = self.cls(x)
        return F.cross_entropy(y_logits, y.argmax(1))

    def loss(self, x, xl, yl):
        if self.gen_weight > 0:
            nelbo, kl_z, kl_y, rec = self.negative_elbo_bound(x)
        else:
            nelbo, kl_z, kl_y, rec = [0] * 4
        ce = self.classification_cross_entropy(xl, yl)
        loss = self.gen_weight * nelbo + self.class_weight * ce

        summaries = dict((
            ('train/loss', loss),
            ('class/ce', ce),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/kl_y', kl_y),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def compute_sigmoid_given(self, z, y):
        logits = self.dec(z, y)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                  self.z_prior[1].expand(batch, self.z_dim))

    def sample_x_given(self, z, y):
        return torch.bernoulli(self.compute_sigmoid_given(z, y))
