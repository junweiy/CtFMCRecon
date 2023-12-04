import numpy as np

import torch
import torch.nn as nn
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

############ Input Positional Encoding ############
class Positional_Encoder():
    def __init__(self, params):
        if params['embedding'] == 'gauss':
            self.B = torch.randn((params['embedding_size'], params['coordinates_size'])) * params['scale']
            self.B = self.B.to(device)
        else:
            raise NotImplementedError

    def embedding(self, x):
        x_embedding = (2. * np.pi * x) @ self.B.t()
        x_embedding = torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        return x_embedding


class Positional_Encoder_Output():
    def __init__(self, size=4):
            # self.B = torch.ones((size, 2))
            self.B = torch.stack((torch.arange(1., size + 1), torch.arange(1., size + 1)), dim=-1)
            self.B = self.B.cuda()

    def embedding(self, x):
        x_embedding = (2. * np.pi * x) @ self.B.t()
        x_embedding = torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        return x_embedding

############ Fourier Feature Network ############
class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class FFN(nn.Module):
    def __init__(self, params):
        super(FFN, self).__init__()

        num_layers = params['network_depth']
        hidden_dim = params['network_width']
        input_dim = params['network_input_size']
        output_dim = params['network_output_size']

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for i in range(1, num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out



############ SIREN Network ############
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / \
            self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


class SIREN(nn.Module):
    def __init__(self, params):
        super(SIREN, self).__init__()

        num_layers = params['network_depth']
        hidden_dim = params['network_width']
        input_dim = params['network_input_size']
        output_dim = 4 if params['multi_contrast'] else 2
        # output_dim = 2

        layers = [SirenLayer(input_dim, hidden_dim, is_first=True)]
        for i in range(1, num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim))
            # layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.Dropout(0.1))
        layers.append(SirenLayer(hidden_dim, output_dim, is_last=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)

        return out


class SIREN_BI(nn.Module):
    def __init__(self, params):
        super(SIREN_BI, self).__init__()

        num_layers = params['network_depth']
        hidden_dim = params['network_width']
        input_dim = params['network_input_size']
        output_dim = 2

        self.layers = nn.ModuleList()
        self.layers.append(SirenLayer(input_dim, hidden_dim, is_first=True))
        for i in range(1, num_layers - 1):
            self.layers.append(SirenLayer(hidden_dim, hidden_dim))
            # layers.append(torch.nn.BatchNorm1d(hidden_dim))
            self.layers.append(torch.nn.Dropout(0.1))

        self.snd_last_t1 = SirenLayer(hidden_dim, hidden_dim)
        self.snd_last_q = SirenLayer(hidden_dim, hidden_dim)
        self.last_t1 = SirenLayer(hidden_dim, output_dim, is_last=True)
        self.last_q = SirenLayer(hidden_dim, output_dim, is_last=True)


    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        t1 = self.snd_last_t1(x)
        t1 = self.last_t1(t1)
        q = self.snd_last_q(x)
        q = self.last_q(q)

        return torch.concat((t1, q), dim=-1)


class Double_SIREN(nn.Module):

    def __init__(self, params):
        super(Double_SIREN, self).__init__()

        num_layers = params['network_depth']
        hidden_dim = params['network_width']
        input_dim = params['network_input_size']
        output_dim = 2

        self.layers = nn.ModuleList()
        self.layers.append(SirenLayer(input_dim, hidden_dim, is_first=True))
        for i in range(1, num_layers - 1):
            self.layers.append(SirenLayer(hidden_dim, hidden_dim))
            # layers.append(torch.nn.BatchNorm1d(hidden_dim))
            self.layers.append(torch.nn.Dropout(0.1))

        self.snd_last_t1 = SirenLayer(hidden_dim, hidden_dim)
        self.snd_last_q = SirenLayer(hidden_dim, hidden_dim)
        self.last_t1 = SirenLayer(hidden_dim, output_dim, is_last=True)
        self.last_q = SirenLayer(hidden_dim, output_dim, is_last=True)

        self.layers_low = nn.ModuleList()
        self.layers_low.append(SirenLayer(input_dim, hidden_dim, is_first=True))
        for i in range(1, num_layers - 1):
            self.layers_low.append(SirenLayer(hidden_dim, hidden_dim))
            # layers.append(torch.nn.BatchNorm1d(hidden_dim))
            self.layers_low.append(torch.nn.Dropout(0.1))

        self.snd_last_t1_low = SirenLayer(hidden_dim, hidden_dim)
        self.snd_last_q_low = SirenLayer(hidden_dim, hidden_dim)
        self.last_t1_low = SirenLayer(hidden_dim, output_dim, is_last=True)
        self.last_q_low = SirenLayer(hidden_dim, output_dim, is_last=True)

        self.last = SirenLayer(output_dim * 4, output_dim * 2, is_last=True)


    def forward(self, x):
        x_low = x.clone()
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        t1 = self.snd_last_t1(x)
        t1 = self.last_t1(t1)
        q = self.snd_last_q(x)
        q = self.last_q(q)
        output = torch.concat((t1, q), dim=-1)

        for i in range(len(self.layers)):
            x_low = self.layers_low[i](x_low)
        t1_low = self.snd_last_t1_low(x_low)
        t1_low = self.last_t1_low(t1_low)
        q_low = self.snd_last_q_low(x_low)
        q_low = self.last_q_low(q_low)
        output_low = torch.concat((t1_low, q_low), dim=-1)

        final = self.last(torch.concat((output, output_low), dim=-1))

        return output, output_low, final
