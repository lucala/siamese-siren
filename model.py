# Code from https://github.com/YannickStruempler/inr_based_compression

import numpy as np
import torch
from torch import nn


class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''

    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True, num_frequencies=None, scale=2):
        super().__init__()

        self.in_features = in_features
        self.scale = scale
        self.sidelength = sidelength
        if num_frequencies == None:
            if self.in_features == 3:
                self.num_frequencies = 10
            elif self.in_features == 2:
                assert sidelength is not None
                if isinstance(sidelength, int):
                    sidelength = (sidelength, sidelength)
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
            elif self.in_features == 1:
                assert fn_samples is not None
                self.num_frequencies = 4
                if use_nyquist:
                    self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)
        else:
            self.num_frequencies = num_frequencies
        # self.frequencies_per_axis = (num_frequencies * np.array(sidelength)) // max(sidelength)
        self.out_dim = in_features + in_features * 2 * self.num_frequencies  # (sum(self.frequencies_per_axis))

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(np.floor(np.log2(nyquist_rate)))

    def forward(self, coords):
        coords_pos_enc = coords
        for i in range(self.num_frequencies):

            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((self.scale ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((self.scale ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc
    
class SineLayer(nn.Module):
    # refer to SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, act_fn=torch.sin):
        super().__init__()
        
        self.omega_0 = omega_0
        self.is_first = is_first
        self.act_fn = act_fn
        
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
        return self.act_fn(self.omega_0 * self.linear(input))
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30., num_frq=64, scale=2.0, act_fn=torch.sin):
        super().__init__()
        
        
        if num_frq is not None:
            pos_out_dim = 2 * num_frq + 1
            is_first = False
            self.positional_encoding = PosEncodingNeRF(in_features=in_features,
                                                        sidelength=None,
                                                        fn_samples=None,
                                                        use_nyquist=True,
                                                        num_frequencies=num_frq,
                                                        scale=scale)
        else:
            self.positional_encoding = nn.Identity()
            pos_out_dim = in_features
            is_first = True
        
        self.net = []
        self.net.append(SineLayer(in_features=pos_out_dim, out_features=hidden_features[0],
                                  is_first=is_first, omega_0=first_omega_0, act_fn=act_fn))

        for i in range(len(hidden_features)-1):
            self.net.append(SineLayer(hidden_features[i], hidden_features[i+1],
                                      is_first=False, omega_0=hidden_omega_0, act_fn=act_fn))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features[-1], out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features[-1]) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features[-1]) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features[-1], out_features, is_first=False,
                                      omega_0=hidden_omega_0, act_fn=act_fn))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        pos_coords = self.positional_encoding(coords)
        output = self.net(pos_coords)
        return output, coords
    
class SiameseSiren(nn.Module):
    def __init__(self, in_features, hidden_features, siam_features, out_features, outermost_linear=False, first_omega_0=30, 
                 hidden_omega_0=30., num_frq=64, scale=2.0, act_fn=torch.sin, separate_last_layer=False):
        super().__init__()
        
        
        if num_frq is not None:
            pos_out_dim = 2 * num_frq + 1
            is_first = False
            self.positional_encoding = PosEncodingNeRF(in_features=in_features,
                                                        sidelength=None,
                                                        fn_samples=None,
                                                        use_nyquist=True,
                                                        num_frequencies=num_frq,
                                                        scale=scale)
        else:
            self.positional_encoding = nn.Identity()
            pos_out_dim = in_features
            is_first = True
        
        self.net = []
        if len(hidden_features) > 0:
            self.net.append(SineLayer(in_features=pos_out_dim, out_features=hidden_features[0],
                                    is_first=is_first, omega_0=first_omega_0, act_fn=act_fn))

        for i in range(len(hidden_features)-1):
            self.net.append(SineLayer(hidden_features[i], hidden_features[i+1],
                                      is_first=False, omega_0=hidden_omega_0, act_fn=act_fn))
            
        self.left_siam = []
        self.right_siam = []
        if len(siam_features) > 0:
            is_first = is_first and len(hidden_features) == 0
            pre_siam_features = hidden_features[-1] if len(hidden_features) > 0 else pos_out_dim
            self.left_siam.append(SineLayer(pre_siam_features, siam_features[0],
                                        is_first=is_first, omega_0=hidden_omega_0, act_fn=act_fn))
            self.right_siam.append(SineLayer(pre_siam_features, siam_features[0],
                                        is_first=is_first, omega_0=hidden_omega_0, act_fn=act_fn))

        for i in range(len(siam_features)-1):
            self.left_siam.append(SineLayer(siam_features[i], siam_features[i+1],
                                      is_first=False, omega_0=hidden_omega_0, act_fn=act_fn))
            self.right_siam.append(SineLayer(siam_features[i], siam_features[i+1],
                                      is_first=False, omega_0=hidden_omega_0, act_fn=act_fn))

        last_features = siam_features[-1] if len(siam_features) > 0 else hidden_features[-1]
        last_linear_layer = nn.Linear(last_features, out_features)
        final_linear_left = last_linear_layer
        if len(siam_features) > 0 or separate_last_layer: # hack to make sure last layer is not shared
            last_linear_layer = nn.Linear(last_features, out_features)
        final_linear_right = last_linear_layer
        
        with torch.no_grad():
            final_linear_left.weight.uniform_(-np.sqrt(6 / last_features) / hidden_omega_0, 
                                            np.sqrt(6 / last_features) / hidden_omega_0)
            final_linear_right.weight.uniform_(-np.sqrt(6 / last_features) / hidden_omega_0, 
                                            np.sqrt(6 / last_features) / hidden_omega_0)
            
        self.left_siam.append(final_linear_left)
        self.right_siam.append(final_linear_right)
        
        self.net = nn.Sequential(*self.net)
        self.left_siam = nn.Sequential(*self.left_siam)
        self.right_siam = nn.Sequential(*self.right_siam)
        self.left_siam = nn.Sequential(self.net, self.left_siam)
        self.right_siam = nn.Sequential(self.net, self.right_siam)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        pos_coords = self.positional_encoding(coords)
        left = self.left_siam(pos_coords)
        right = self.right_siam(pos_coords)
        output = torch.cat((left, right), dim=-1)
        return output, coords