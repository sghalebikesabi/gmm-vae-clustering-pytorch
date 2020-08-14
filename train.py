import numpy as np
import torch
from torchvision.utils import save_image

from loss import loss_function, test_acc
from utils import stream_print


def train(epoch, model, train_loader, test_loader, optimizer, device, f):
    model.train()
    train_loss = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device).view(-1, 784)
        optimizer.zero_grad()
        px_logit, variational_params, latent_samples = model(data)
        loss = loss_function(data, targets, px_logit, variational_params, latent_samples)
        loss['optimization_loss'].backward()
        train_loss += loss['optimization_loss'].item()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        
        data_eval = train_loader.dataset.data.view(-1, 784)[np.random.choice(50000, 10000)].to(device)/255.0
        recon_batch_eval, var_eval, lat_eval = model(data_eval) 
        loss_eval = loss_function(data_eval, targets, recon_batch_eval, var_eval, lat_eval)
        
        data_test = test_loader.dataset.data.view(-1, 784).to(device)/255.0
        recon_batch_test, var_test, lat_test = model(data_test) 
        loss_test = loss_function(data_test, targets, recon_batch_test, var_test, lat_test)

        a,b,c,d = -loss_eval['nent']/len(data_eval), loss_eval['optimization_loss']/len(data_eval), -loss_test['nent']/len(data_test), loss_test['optimization_loss']/len(data_test)
        e = test_acc(model, test_loader, device)
        string = ('{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s}'
                    .format('tr_ent', 'tr_loss', 't_ent', 't_loss', 't_acc', 'epoch'))
        stream_print(f, string, epoch==1)
        string = ('{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10d}'
                    .format(a, b, c, d, e, epoch))
        stream_print(f, string)
    
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    n = min(data.size(0), 8)
    for i in range(10):
        comparison = torch.cat([data.view(-1, 1, 28, 28)[:n],
                                px_logit[i][:n].view(-1, 1, 28, 28)])
        save_image(comparison.cpu(),
                    'logs/reconstruction_' + str(epoch) + '_' + str(i) + '.png', nrow=n)
