import torch
import torch.nn as nn
import params
import os


def save_model(models, optimizers, fname='model.pth'):
    path = os.path.abspath('') + '/models/' + fname
    if params.control_affine or params.linear_state_space:
        if params.linear_state_space_offset:
            ae, fdyn = models
            fdyn_drift, fdyn_cntrl = fdyn
            fdyn_drift, fdyn_offset = fdyn_drift
            ae_opt, fdyn_opt = optimizers
            fdyn_drift_opt ,fdyn_cntrl_opt = fdyn_opt
            fdyn_drift_opt, fdyn_offset_opt = fdyn_drift_opt
            torch.save({'ae_state_dict': ae.state_dict(),
                        'fdyn_drift_state_dict': fdyn_drift.state_dict(),
                        'fdyn_offset_state_dict': fdyn_offset.state_dict(),
                        'fdyn_cntrl_state_dict': fdyn_cntrl.state_dict(),
                        'ae_opt_state_dict': ae_opt.state_dict(),
                        'fdyn_drift_opt_state_dict': fdyn_drift_opt.state_dict(),
                        'fdyn_offset_opt_state_dict': fdyn_offset_opt.state_dict(),
                        'fdyn_cntrl_opt_state_dict': fdyn_cntrl_opt.state_dict()},
                        path)
        else:
            ae, fdyn = models 
            fdyn_drift, fdyn_cntrl = fdyn
            ae_opt, fdyn_opt = optimizers
            fdyn_drift_opt, fdyn_cntrl_opt = fdyn_opt
            torch.save({'ae_state_dict': ae.state_dict(),
                        'fdyn_drift_state_dict': fdyn_drift.state_dict(),
                        'fdyn_cntrl_state_dict': fdyn_cntrl.state_dict(),
                        'ae_opt_state_dict': ae_opt.state_dict(),
                        'fdyn_drift_opt_state_dict': fdyn_drift_opt.state_dict(),
                        'fdyn_cntrl_opt_state_dict': fdyn_cntrl_opt.state_dict()},
                        path)
    else:
        ae, fdyn = models
        ae_opt, fdyn_opt = optimizers 
        torch.save({'ae_state_dict': ae.state_dict(),
                    'fdyn_state_dict': fdyn.state_dict(),
                    'ae_opt_state_dict': ae_opt.state_dict(),
                    'fdyn_opt_state_dict': fdyn_opt.state_dict()},
                    path)


def load_model(fname='model.pth', abcrown=False):
    if not abcrown:
        path = os.path.abspath('') + '/models/' + fname
    else:
        path = os.path.abspath('') + '/' + fname
    ckpt = torch.load(path, weights_only=True) # strict=False

    if not params.neural_dynamics:
        if params.vae:
            ae = VAE(params.enc_layers, params.dec_layers,
                     residual=params.residual_ae, res_bias=params.res_bias_ae,
                     init=params.init_ae)
        else:
            ae = AE(params.enc_layers, params.dec_layers,
                    residual=params.residual_ae, res_bias=params.res_bias_ae,
                    init=params.init_ae)
    else:
        ae = AE(params.enc_layers, params.dec_layers,
                residual=params.residual_ae, res_bias=params.res_bias_ae,
                init=params.init_ae,
                iden=True) # iden=True is the only thing that matters here

    ae_opt = params.ae_opt(ae.parameters(), lr=params.lr)
    ae.load_state_dict(ckpt['ae_state_dict'])
    ae_opt.load_state_dict(ckpt['ae_opt_state_dict'])

    if params.control_affine or params.linear_state_space:
        fdyn_drift = FF(params.fdyn_drift_layers,
                        residual=params.residual_fdyn_drift, res_bias=params.res_bias_fdyn,
                        init=params.init_fdyn_drift)
        fdyn_cntrl = FF(params.fdyn_cntrl_layers,
                        residual=params.residual_fdyn_cntrl, res_bias=params.res_bias_fdyn,
                        init=params.init_fdyn_cntrl)
        fdyn_drift.load_state_dict(ckpt['fdyn_drift_state_dict'])
        fdyn_cntrl.load_state_dict(ckpt['fdyn_cntrl_state_dict'])
        fdyn_drift_opt = params.fdyn_drift_opt(fdyn_drift.parameters(), lr=params.lr)
        fdyn_cntrl_opt = params.fdyn_cntrl_opt(fdyn_cntrl.parameters(), lr=params.lr)
        fdyn_drift_opt.load_state_dict(ckpt['fdyn_drift_opt_state_dict'])
        fdyn_cntrl_opt.load_state_dict(ckpt['fdyn_cntrl_opt_state_dict'])
        if params.linear_state_space_offset:
            fdyn_offset = FF(params.fdyn_offset_layers,
                            residual=params.residual_fdyn_offset, res_bias=params.res_bias_fdyn,
                            init=params.init_fdyn_offset)
            fdyn_offset.load_state_dict(ckpt['fdyn_offset_opt_state_dict'])
            fdyn_offset_opt = params.fdyn_offset_opt(fdyn_offset.parameters(), lr=params.lr)
            fdyn_offset_opt.load_state_dict(ckpt['fdyn_offset_opt_state_dict'])
            fdyn_drift = (fdyn_drift, fdyn_offset)
            fdyn_drift_opt = (fdyn_drift_opt, fdyn_offset_opt)
        fdyn = (fdyn_drift, fdyn_cntrl)
        fdyn_opt = (fdyn_drift_opt, fdyn_cntrl_opt)
    else:
        fdyn = FF(params.fdyn_layers,
                  residual=params.residual_fdyn, res_bias=params.res_bias_fdyn,
                  init=params.init_fdyn)
        fdyn_opt = params.fdyn_opt(fdyn.parameters(), lr=params.lr)
        fdyn.load_state_dict(ckpt['fdyn_state_dict'])
        fdyn_opt.load_state_dict(ckpt['fdyn_opt_state_dict'])

    return ae, fdyn, ae_opt, fdyn_opt


class I(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1, 1)
    
    def forward(self, x):
        return x


class FF(nn.Module):
    def __init__(self, layers, residual=False, init=None, init_bias=None, res_bias=False):
        super().__init__()
        self.features = layers
        self.layers = nn.ModuleList()
        self.residual = residual
        self.res_bias = res_bias
        for j, (out_f, in_f) in enumerate(layers):
            print("\n")
            print("### Layer {} ###".format(j))
            if residual and (out_f != in_f):
                print("Residual, fin != fout")
                g = nn.Linear(in_f, out_f, bias=self.res_bias)
                p = nn.Linear(in_f, out_f, bias=False)
                if self.res_bias:
                    if init_bias is not None:
                        print("Bias used in residual layer is", init_bias)
                        init_bias(g.bias)
                    else:
                        print("Bias used in residual layer is", nn.init.zeros_)
                        nn.init.zeros_(g.bias)
                if init is not None:
                    print("g init is", init)
                    print("p init is n.init.eye_")
                    init(g.weight)
                    nn.init.eye_(p.weight)
                if params.batch_norm:
                    print("Using batch_norm")
                    bn = nn.BatchNorm1d(in_f)
                    layer = nn.ModuleList([bn, g, p])
                else:
                    layer = nn.ModuleList([g, p])
                self.layers.append(layer)
            elif residual:
                print("Residual, fin == fout")
                g = nn.Linear(in_f, out_f, bias=self.res_bias)
                if self.res_bias:
                    if init_bias is not None:
                        print("Bias used in residual layer is", init.bias)
                        init_bias(g.bias)
                    else:
                        print("Bias used in residual layer is", nn.init.zeros_)
                        nn.init.zeros_(g.bias)
                if init is not None:
                    print("g init:", init)
                    init(g.weight)
                if params.batch_norm:
                    print("Using batch norm")
                    bn = nn.BatchNorm1d(in_f)
                    layer = nn.ModuleList([bn, g])
                else:
                    layer = nn.ModuleList([g])
                self.layers.append(layer)
            else:
                print("Not residual")
                g = nn.Linear(in_f, out_f)
                if init is not None:
                    print("g init is", init)
                    init(g.weight)
                if init_bias is not None:
                    print("g bias init is", init_bias)
                    init_bias(g.bias)
                else: 
                    print("g bias init is nn.init.zeros")
                    nn.init.zeros_(g.bias)
                if params.batch_norm:
                    print("Using batch norm")
                    bn = nn.BatchNorm1d(in_f)
                    layer = nn.ModuleList([bn, g])
                else:
                    layer = nn.ModuleList([g])
                self.layers.append(layer)
        print("\n")


    def forward(self, x):
        for i, l in enumerate(self.layers):
            off = int(params.batch_norm)
            if params.batch_norm:
                bn = l[0]
                x = bn(x)
            x_prev = x
            if self.residual:
                if self.features[i][0] != self.features[i][1]:
                    g, p = (l[off], l[off+1])
                    s = g(x)
                    x = params.act(s) + p(x)
                else:
                    s = l[off](x)
                    x = s + x
            else:
                s = l[off](x)
                x = params.act(s)
        if params.final_act:
            return x
        elif self.residual:
            if self.features[-1][0] != self.features[i][1]:
                l = self.layers[-1]
                g, p = (l[off], l[off+1])
                return s + p(x_prev)
            else:
                return s + x_prev
        else:
            return s


class AE(nn.Module):
    def __init__(self, enc_layers, dec_layers, residual=False, res_bias=False,
                 iden=False, init=None, init_bias=None):
        super().__init__()
        if iden:
            self.encoder = I()
            self.decoder = I()
        else:
            print("ENCODER")
            self.encoder = FF(enc_layers, residual=residual, res_bias=res_bias, init=init, init_bias=init_bias)
            print("DECODER")
            self.decoder = FF(dec_layers, residual=residual, res_bias=res_bias, init=init, init_bias=init_bias)
        self.vae = False
    
    def forward(self, x):
        xhat = self.decoder(self.encoder(x))
        return xhat

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        xhat = self.decoder(z)
        return xhat


class VAE(nn.Module):
    def __init__(self, enc_layers, dec_layers, residual=False,
                 init=None):
        super().__init__()
        self.encoder = FF(enc_layers, residual=residual, init=init)
        self.decoder = FF(dec_layers, residual=residual, init=init)
        self.vae = True

    def get_latent_params(self, x):
        # acts on single element
        enc_out = self.encoder(x)
        assert len(enc_out.shape) == 1
        d_z = int(enc_out.shape[0] // 2)
        mu, logvar = (enc_out[:d_z], enc_out[d_z:])
        return mu, logvar

    def encode(self, x):
        # acts on  a batch
        mu, logvar = torch.vmap(self.get_latent_params)(x)
        assert (len(mu.shape) == 2) and (len(logvar.shape) == 2)
        std = torch.exp(logvar/2)
        eps = torch.randn(size=std.shape)
        return mu + eps*std

    def decode(self, z):
        xhat = self.decoder(z)
        return xhat

    def forward(self, x):
        return self.decode(self.encode(x))
