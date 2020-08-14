import torch
from torch import nn
from torch.nn import functional as F


class VAE(torch.nn.Module):
    """VAE with GMM prior."""
    
    def __init__(self, model_params_dict):
        super(VAE, self).__init__()

        # input params
        self.cuda = model_params_dict['cuda']
        self.input_dim = model_params_dict['input_dim']
        self.r_cat_dim = model_params_dict['r_cat_dim']
        self.z_dim = model_params_dict['z_dim']
        self.h_dim = model_params_dict['h_dim']

        # q(y|x)
        self.fc_x_h = torch.nn.Linear(self.input_dim, self.h_dim)
        self.fc_hx_h = torch.nn.Linear(self.h_dim, self.h_dim)
        self.fc_h_qyl = torch.nn.Linear(self.h_dim, self.r_cat_dim)
        self.fc_qyl_qy = torch.nn.Softmax(1)

        # q(z|x, y)
        self.fc_xy_h = torch.nn.Linear(self.input_dim + self.r_cat_dim, self.h_dim)
        self.fc_hxy_h = torch.nn.Linear(self.h_dim, self.h_dim)
        self.fc_h_z = torch.nn.Linear(self.h_dim, self.z_dim*2)
        
        # p(z|y)
        self.fc_y_z = torch.nn.Linear(self.r_cat_dim, self.z_dim*2)
        
        # p(x|z)
        self.fc_z_h = torch.nn.Linear(self.z_dim, self.h_dim)
        self.fc_hz_h = torch.nn.Linear(self.h_dim, self.h_dim)
        self.fc_h_xl = torch.nn.Linear(self.h_dim, self.input_dim)

    def qy_graph(self, x):
        # q(y|x)
        hx = F.relu(self.fc_x_h(x))
        h = F.relu(self.fc_hx_h(hx))
        qy_logit = self.fc_h_qyl(h)
        qy = self.fc_qyl_qy(qy_logit)
        return qy_logit, qy

    def qz_graph(self, x, y):
        # q(z|x, y)
        xy = torch.cat([x, y], 1)

        hxy = F.relu(self.fc_xy_h(xy))
        h1 = F.relu(self.fc_hxy_h(hxy))
        z_post = self.fc_h_z(h1)
        z_mu_post, z_logvar_post = torch.split(z_post, self.z_dim, dim=1) 
        z_std_post = torch.sqrt(torch.exp(z_logvar_post))

        eps = torch.randn_like(z_std_post)
        z = z_mu_post + eps*z_std_post

        return z, z_mu_post, z_logvar_post 
        
    def decoder(self, z, y):
        # p(z)
        z_prior = self.fc_y_z(y)
        z_mu_prior, z_logvar_prior = torch.split(z_prior, self.z_dim, dim=1) 

        # p(x|z)
        hz = F.relu(self.fc_z_h(z))
        h2 = F.relu(self.fc_hz_h(hz))
        x_logit = self.fc_h_xl(h2)
                
        return z_mu_prior, z_logvar_prior, torch.sigmoid(x_logit)

    def forward(self, x):
        xb = x
        y_ = torch.zeros([x.shape[0], 10])
        qy_logit, qy = self.qy_graph(xb)
        z, zm, zv, zm_prior, zv_prior, px = [[None] * 10 for i in range(6)]
        for i in range(10):
            y = y_ + torch.eye(10)[i]
            z[i], zm[i], zv[i] = self.qz_graph(xb, y)
            zm_prior[i], zv_prior[i], px[i] = self.decoder(z[i], y)
        
        latent_samples = {'z': z}
        variational_params = {
            'zm': zm,
            'zv': zv, 
            'zm_prior': zm_prior, 
            'zv_prior': zv_prior,
            'qy_logit': qy_logit,
            'qy': qy,
        }

        return px, variational_params, latent_samples
