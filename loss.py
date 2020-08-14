import math
import numpy as np

import torch
import torch.nn.functional as F


def loss_function(data, targets, px_logit, variational_params, latent_samples):

    nent = torch.sum(variational_params['qy'] * torch.nn.LogSoftmax(1)(variational_params['qy_logit']), 1)  ###

    losses = [None]*10
    for i in range(10):
        losses[i] = labeled_loss(data, px_logit[i], latent_samples['z'][i], variational_params['zm'][i], torch.exp(variational_params['zv'][i]), variational_params['zm_prior'][i], torch.exp(variational_params['zv_prior'][i]))

    loss = torch.stack([nent] + [variational_params['qy'][:, i] * losses[i] for i in range(10)]).sum(0)

    loss_dict = {
        'nent': nent.sum(),
        'optimization_loss': loss.sum(),     
    }

    return loss_dict

def labeled_loss(x, px_logit, z, zm, zv, zm_prior, zv_prior):
    xy_loss = -log_bernoulli_with_logits(x, px_logit)
    xy_loss += log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)
    return xy_loss - np.log(0.1)

def log_bernoulli_with_logits(x, logits, eps=0.0, axis=-1):
    if eps > 0.0:
        max_val = np.log(1.0 - eps) - np.log(eps)
        logits = torch.clamp(logits, -max_val, max_val)
    return -torch.sum(
        F.binary_cross_entropy(logits, x, reduction="none"), axis)

def log_normal(x, mu, var, eps=0.0, axis=-1):
    if eps > 0.0:
        var = torch.add(var, eps, name='clipped_var')
    return -0.5 * torch.sum(
        np.log(2 * math.pi) + torch.log(var) + torch.square(x - mu) / var, axis)

def test_acc(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        data = test_loader.dataset.data.view(-1, 784).to(device)/255.0
        labels = test_loader.dataset.targets.to(device)
        qy_logit, _ = model.qy_graph(data)
        cat_pred = qy_logit.argmax(1)
        real_pred = np.zeros_like(cat_pred)
        for cat in range(qy_logit.shape[1]):
            idx = cat_pred == cat
            lab = labels[idx]
            if len(lab) == 0:
                continue
            real_pred[cat_pred == cat] = lab.mode()[0] 
    acc = np.mean(real_pred == test_loader.dataset.targets.numpy())    
    return(acc)
