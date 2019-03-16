
from RealNVP import RealNVP
import numpy as np
import torch
from torch import distributions
from torch import nn


class SampleGenerator():

    def __init__(self):

        # create the network
        # set the parameters
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # enable gpu processing
        # Masks
        masks = torch.from_numpy(np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]] * 4).astype(np.float32)).to(device)
        # network dimensions
        num_neurons = 40
        in_dim = 5  # in dim and out dim are the same

        nets = lambda: nn.Sequential(nn.BatchNorm1d(in_dim), nn.Linear(in_dim, num_neurons),
                                     nn.BatchNorm1d(num_neurons), nn.ReLU(),  # input block
                                     nn.Linear(num_neurons, num_neurons), nn.BatchNorm1d(num_neurons), nn.ReLU(),
                                     # input block
                                     nn.Linear(num_neurons, num_neurons), nn.BatchNorm1d(num_neurons), nn.ReLU(),
                                     # residual block
                                     nn.Linear(num_neurons, num_neurons), nn.BatchNorm1d(num_neurons), nn.ReLU(),
                                     # residual block
                                     nn.BatchNorm1d(num_neurons), nn.ReLU(), nn.Linear(num_neurons, in_dim), nn.Tanh()
                                     # end block
                                     )

        nett = lambda: nn.Sequential(nn.BatchNorm1d(in_dim), nn.Linear(in_dim, num_neurons),
                                     nn.BatchNorm1d(num_neurons), nn.ReLU(),  # input block
                                     nn.Linear(num_neurons, num_neurons), nn.BatchNorm1d(num_neurons), nn.ReLU(),
                                     # input block
                                     nn.Linear(num_neurons, num_neurons), nn.BatchNorm1d(num_neurons), nn.ReLU(),
                                     # residual block
                                     nn.Linear(num_neurons, num_neurons), nn.BatchNorm1d(num_neurons), nn.ReLU(),
                                     # residual block
                                     nn.BatchNorm1d(num_neurons), nn.ReLU(), nn.Linear(num_neurons, in_dim)  # end block
                                     )
        # Gaussian prior
        prior = distributions.MultivariateNormal(torch.zeros(in_dim).to(device), torch.eye(in_dim).to(device))

        self.net = RealNVP(nets, nett, masks, prior)
        self.net.load_state_dict(torch.load("net.pt"))
        self.net.eval()
        self.net.to(device)

    def sample(self):
        x = self.net.sample()
        log_prob = self.net.log_prob(x)
        prob = torch.exp(log_prob)
        x = x.cpu().detach().numpy()
        prob = prob.cpu().detach().numpy()
        return x,prob

cdef public object createSampleGenerator():
    sampleGen = SampleGenerator()
    return sampleGen

cdef public void sample(object p, float sampleArray[5], float* pdf):
    x, prob = p.sample()
    pdf[0] = float(prob) # pdf[0] = *pdf

    # copy data into array
    for ind,val in enumerate(x):
        sampleArray[ind]=val

