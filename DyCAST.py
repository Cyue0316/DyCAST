import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import geotorch
from torchdiffeq import odeint_adjoint as odeint

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
    

class Func(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims, dims):
        super(Func, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.dims = dims
        
        self.linear = nn.Sequential(nn.Linear(input_dims, 2*input_dims, bias=True),
                                    
                                    nn.Tanh(),
                                    nn.Linear(input_dims*2, input_dims, bias=True),
                                )
        
        self.decoder = nn.Sequential(
                                    nn.Linear(input_dims+1, 512),
                                    nn.Linear(512, 512),
                                    nn.SiLU(),
                                    nn.Linear(512, dims*dims),
                                    )
        
    
    def constraint(self, z, t):
        
        def _h_fun(z, s=1.0, t=1.0):
            z = F.tanh(z)
            t = torch.tensor(t).unsqueeze(0).unsqueeze(1).to(z)
            z = torch.cat((z, t), dim=1)
            w_est = self.decoder(z).reshape(self.dims, self.dims)
            h = torch.trace(torch.matrix_exp(w_est ** 2)) - self.dims
            return h
        z = z.detach().requires_grad_(True)
        h = _h_fun(z, t=t)
        jacobian = torch.autograd.functional.jacobian(_h_fun, z, create_graph=True)
        # F_ = jacobian.t().to(z.device)
        F_ = jacobian.t() @ torch.linalg.pinv(jacobian @ jacobian.t()).to(z.device)
        return - 1 * F_ * h
    
    def forward(self, t, z):
        x = self.linear(z)
        constrain = self.constraint(x, t)
        
        return x + constrain.t().reshape(1, -1)
    

class LowRankLinear(nn.Module):
    def __init__(self, in_features, k):
        super(LowRankLinear, self).__init__()
        self.d = in_features
        self.k = k
        self.u = nn.Conv1d(self.d, self.k, kernel_size=1, bias=False)
        self.v = nn.Conv1d(self.k, self.d, kernel_size=1, bias=False)
        

    def forward(self, x):
        self.E_est = self.u.weight.squeeze(2)
        self.A_est = self.v.weight.squeeze(2)
        w_est = (self.E_est.transpose(0, 1) @ self.A_est.transpose(0, 1)) 
        out = self.u(x)  # 第一层变换
        out = self.v(out)  # 第二层变换
        return out, w_est

class encoder(nn.Module):
    def __init__(self, in_features, k):
        super(encoder, self).__init__()
        self.line1 = nn.Linear(in_features, k, bias=True)
        

    def forward(self, x):
        x = self.line1(x)
        x = F.relu(x)
        return x


class CausalODE(nn.Module):
    def __init__(self, dims=5, k_dims=4, lag=2):
        super(CausalODE, self).__init__()
        self.d = dims
        self.k = k_dims
        self.lag = lag
        
        self.encoder = encoder(2*dims, k_dims)
        self.func = Func(hidden_dims=k_dims, input_dims=dims * k_dims, dims=dims)
        # self.inter_func = inter_Func(input_dims=dims*dims, hidden_dims=k_dims, dims=dims)

        self.init_intra_t = nn.Parameter(torch.randn(dims, k_dims), requires_grad=True)
        self.init_intra_s = nn.Parameter(torch.randn(k_dims, dims), requires_grad=True)
        self.init_intra_t = nn.init.kaiming_uniform_(self.init_intra_t)
        self.init_intra_s = nn.init.kaiming_uniform_(self.init_intra_s)

        # self.inter = nn.Parameter(torch.randn(dims, dims), requires_grad=True)

        self.layers = nn.ModuleList([LowRankLinear(self.d, self.k)
                                    for _ in range(self.lag)])
        
    def terminal_value_CDE(self, w0, length):
        t = torch.linspace(1, length, length).to(w0.device)
        intra_t = odeint(self.func, w0, t, method="rk4", atol=1e-9, rtol=1e-7).permute(1, 0 ,2)
        return intra_t
    
    def h_func(self, s=1.0):
        h = 0
        for t in range(self.west_t.size(0)):
            h += torch.trace(torch.matrix_exp(self.west_t[t, ...] * self.west_t[t, ...])) - self.d
        return h
    
    def l1_reg(self):
        west_t= self.west_t.permute(2, 1, 0)
        loss = torch.norm(self.p_est, p=1, dim=(0, 1)).sum() + torch.norm(west_t, p=1, dim=(0, 1)).sum()
        return loss
    
    def diag_zero(self):
        diag_loss = 0
        for i in range(self.west_t.size(0)):
            diag_loss += torch.trace(self.west_t[i, ...] * self.west_t[i, ...])
        return diag_loss
    
    def laplacian_loss(self):
        loss = 0.0
        # 在时间维度（第3维）上计算相邻矩阵之间的差异
        for t in range(1, self.west_t.size(0)-1):
            laplacian = self.west_t[t-1, :, :] + self.west_t[t+1, :, :] - 2*self.west_t[t, :, :]
            loss += torch.sum(laplacian ** 2)
        return loss
    


    def forward(self, x):
        # x:tensor -> B T D
        x_t = x # B T D
        length = x.size(1)
        self.init_intra = torch.matmul(self.init_intra_t, self.init_intra_s)

        patchs = [torch.cat((self.init_intra[i, :], self.init_intra[:, i]), dim=0) for i in range(self.d)]
        patchs = torch.stack(patchs, dim=0)
        t = torch.linspace(1, length, length).unsqueeze(0).unsqueeze(2).to(x.device)
        
        # z0 = self.encoder(self.init_intra).reshape(1, -1)
        z0 = self.encoder(patchs).reshape(1, -1)
        # self.west_h = (self.terminal_value_CDE(z0, length))
        self.west_h = F.tanh(self.terminal_value_CDE(z0, length))
        self.west_h = torch.cat((self.west_h, t), dim=2)
        self.west_t = self.func.decoder(self.west_h).reshape(length, self.d, self.d)
        intra_output = torch.einsum('btd, tdk -> btk', x_t, self.west_t)
        
        output = []
        causal = []
        # static inter
        for i, layer in enumerate(self.layers.to(x.device)):
            out, A_est = layer(x[:, :-(i+1), :].permute(0, 2, 1))
            output.append(out)
            causal.append(A_est)
        output = torch.stack(output, dim=2)
        self.p_est = torch.stack(causal, dim=2)
        output = torch.sum(output, dim=2).permute(0, 2, 1)
        
        output = torch.cat([torch.zeros_like(output[:, :1, :]), output], dim=1).to(x.device)
        out = intra_output + output

        # dynamic inter
        # Y = x[:, :-1, :]
        # h_0 = self.inter.reshape(1, -1)
        # self.p_est = odeint(self.inter_func, h_0, torch.linspace(1, length-1, length-1).to(x.device), method="rk4", atol=1e-9, rtol=1e-7).reshape(length-1, self.d, self.d)
        # inter_output = torch.einsum('btd, tdk -> btk', Y, self.p_est)
        # inter_output = torch.cat([torch.zeros_like(inter_output[:, :1, :]), inter_output], dim=1).to(x.device)
        # out = intra_output + inter_output
        return out

if __name__ == '__main__':
    x = torch.rand(64, 7, 5)
    model = CausalODE(dims=5, k_dims=64, lag=1)
    output = model(x)
    print(output.shape)

