import numpy as np
import torch
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
if 'solution' in script_directory:
    from solution import utils as ut
else:
    from submission import utils as ut
from torch import optim

def train(model, train_loader, labeled_subset, device, tqdm, writer,
          iter_max=np.inf, iter_save=np.inf,
          model_name='model', y_status='none', reinitialize=False):
    # Optimization
    if reinitialize:
        model.apply(ut.reset_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    i = 0
    with tqdm(total=iter_max) as pbar:
        while True:
            for batch_idx, (xu, yu) in enumerate(train_loader):
                i += 1 # i is num of gradient steps taken by end of loop iteration
                
                # In PyTorch, optimizer.zero_grad() is a crucial step when training a model using backpropagation. 
                # It resets the gradients of all model parameters to zero before performing the backward pass. 
                
                # clear the previous gradients of all optimized variables
                optimizer.zero_grad()

                if y_status == 'none':
                    # reshape xu to be 2D i.e. (batch_size, 1 * 28 * 28)
                    # then generates random samples from a Bernoulli distribution for each element of xu.
                    xu = torch.bernoulli(xu.to(device).reshape(xu.size(0), -1))

                    # From yu / labels of 0-9, create diagonal matrix of one-hot vectors 10x10
                    # For unsupervised training, this step is unnecessary.
                    yu = yu.new(np.eye(10)[yu]).to(device).float()

                    loss, summaries = model.loss(xu)

                    # ADDED: calculate loss for the batch
                    #loss = -1 * loss.sum() / xu.size(0)

                elif y_status == 'semisup':
                    xu = torch.bernoulli(xu.to(device).reshape(xu.size(0), -1))
                    yu = yu.new(np.eye(10)[yu]).to(device).float()
                    # xl and yl already preprocessed
                    xl, yl = labeled_subset
                    xl = torch.bernoulli(xl)
                    loss, summaries = model.loss(xu, xl, yl)

                    # Add training accuracy computation
                    pred = model.cls(xu).argmax(1)
                    true = yu.argmax(1)
                    acc = (pred == true).float().mean()
                    summaries['class/acc'] = acc

                elif y_status == 'fullsup':
                    # Janky code: fullsup is only for SVHN
                    # xu is not bernoulli for SVHN
                    xu = xu.to(device).reshape(xu.size(0), -1)
                    yu = yu.new(np.eye(10)[yu]).to(device).float()
                    loss, summaries = model.loss(xu, yu)

                # Backward pass
                loss.backward()
                # Update parameters
                optimizer.step()

                # Feel free to modify the progress bar
                if y_status == 'none':
                    pbar.set_postfix(
                        loss='{:.2e}'.format(loss))
                elif y_status == 'semisup':
                    pbar.set_postfix(
                        loss='{:.2e}'.format(loss),
                        acc='{:.2e}'.format(acc))
                elif y_status == 'fullsup':
                    pbar.set_postfix(
                        loss='{:.2e}'.format(loss),
                        kl='{:.2e}'.format(summaries['gen/kl_z']))
                pbar.update(1)

                # Log summaries
                if i % 50 == 0: ut.log_summaries(writer, summaries, i)

                # Save model
                if i % iter_save == 0:
                    ut.save_model_by_name(model, i)

                if i == iter_max:
                    return
