import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from lop.algos.gnt import GnT


class BackpropKL(object):
    def __init__(self, net, step_size=0.001, loss='mse', opt='sgd', beta_1=0.9, beta_2=0.999, weight_decay=0.0,
                 to_perturb=False, perturb_scale=0.1,
            kl_div_scale = 0,
            power_law_alpha = -1.5,
            replacement_rate=0.001,
            decay_rate=0.9,
            device='cpu',
            maturity_threshold=100,
            util_type='contribution',
            init='kaiming',
            accumulate=False, momentum=0):
        self.net = net
        self.to_perturb = to_perturb
        self.perturb_scale = perturb_scale
        self.device = device
        self.util = None
        self.kl_div_scale = kl_div_scale
        self.power_law_alpha = power_law_alpha
        self.bins = 100

        # define the optimizer
        if opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)
        elif opt == 'adam':
            self.opt = optim.Adam(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),
                                  weight_decay=weight_decay)
        elif opt == 'adamW':
            self.opt = optim.AdamW(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),
                                   weight_decay=weight_decay)

        # define the loss function
        self.loss = loss
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]

        # Placeholder
        self.previous_features = None

        self.target_power_law = self.compute_target_power_law()

        # define the generate-and-test object for the given network
        self.gnt = None
        self.gnt = GnT(
            net=self.net.layers,
            hidden_activation=self.net.act_type,
            opt=self.opt,
            replacement_rate=replacement_rate,
            decay_rate=decay_rate,
            maturity_threshold=maturity_threshold,
            util_type=util_type,
            device=device,
            loss_func=self.loss_func,
            init=init,
            accumulate=accumulate,
        )

    def compute_target_power_law(self):
        # histogram bin centers
        points = torch.linspace(
           0.0,
           1.0,
            self.bins + 1,
            device=self.device
        )

        centers = (points[:-1] + points[1:]) / 2

        # target power law P
        # TODO: look into computing the integral over the bin
        p = centers ** (self.power_law_alpha)
        p = p / p.sum()

        return p

    def copy_util_score(self, array_of_torch_tensors):
        return torch.stack(array_of_torch_tensors)

    def utility_powerlaw_kl(self, utilities,  eps=1e-8):

        # flatten utilities
        u = utilities.flatten()

        # avoid division by 0
        u += eps

        # scale-free
        u = u / u.sum()

        # empirical distribution Q
        hist = torch.histc(
            u,
            bins=self.bins,
            min=0.0,
            max=1.0
        )

        q = hist / hist.sum()
        p = self.target_power_law

        # Forward `KL(P || Q)
        kl = torch.sum(p * (torch.log(p+eps) - torch.log(q+eps)))

        #print("kl: ", kl)
        return kl

    def learn(self, x, target):
        """
        Learn using one step of gradient-descent
        :param x: input
        :param target: desired output
        :return: loss
        """

        output, features = self.net.predict(x=x)
        loss = self.loss_func(output, target)
        #print("loss: ", loss)
        self.previous_features = features


        # Update util functions
        if type(self.gnt) is GnT:
            self.gnt.update_utility_for_logging(features=self.previous_features)

            self.util = self.copy_util_score(self.gnt.util)

            # Add KL div loss
            if self.kl_div_scale > 0:
                util_kl = self.utility_powerlaw_kl(self.util)
                loss = loss + self.kl_div_scale * util_kl

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.to_perturb:
            self.perturb()
        if self.loss == 'nll':
            return loss.detach(), output.detach()
        return loss.detach()

    def perturb(self):
        with torch.no_grad():
            for i in range(int(len(self.net.layers)/2)+1):
                self.net.layers[i * 2].bias +=\
                    torch.empty(self.net.layers[i * 2].bias.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)
                self.net.layers[i * 2].weight +=\
                    torch.empty(self.net.layers[i * 2].weight.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)
