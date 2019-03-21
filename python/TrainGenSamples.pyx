import numpy as np
import torch
from torch import distributions
from torch import nn

class RealNVP(nn.Module):
    def __init__(self, net_s, net_t, masks, prior):
        super(RealNVP, self).__init__()
        self.prior = prior
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # enable gpu processing
        self.masks = nn.Parameter(masks, requires_grad=False).to(self.device)
        self.t = torch.nn.ModuleList([net_t() for _ in range(len(masks))]).to(self.device)
        self.s = torch.nn.ModuleList([net_s() for _ in range(len(masks))]).to(self.device)

        self.vec_dim = len(masks)

    def g(self, z):  # inverse mapping (latent space to primary space)
        x = z
        for i in reversed(range(len(self.t))):
            x_ = x * self.masks[i]
            s = self.s[i](x_) * (1 - self.masks[i])
            t = self.t[i](x_) * (1 - self.masks[i])
            x = x_ + (1 - self.masks[i]) * ((x - t) * torch.exp(-s))
        return x  # sample in primary space

    def f(self, x):  # forward mapping from primary space to latent space
        y = x
        log_det_J = x.new_zeros(x.shape[0])  # create array of zeros with the number of rows of x

        for i in range(len(self.t)):
            y_ = self.masks[i] * y  # apply mask i to input vector, preventing these parameters being sent to NN
            s = self.s[i](y_) * (
                        1 - self.masks[i])  # compute the scale factor for this layer and select only the active part

            t = self.t[i](y_) * (1 - self.masks[i])  # compute the translation factor
            y = y_ + (1 - self.masks[i]) * (y * torch.exp(
                s) + t)  # apply exp scale and translation to input vector and selected unmasked parts
            log_det_J += s.sum(dim=1)

        return y, log_det_J  # y is in latent space, log_det_J is the log of the Jacobian of the transformation

    def log_prob(self, x):
        z, logp = self.f(x)
        return logp + self.prior.log_prob(z)

    def sample(self, batchSize=1):
        z = torch.Tensor(self.prior.sample((batchSize,)))  # draw sample in latent space
        x = self.g(z) # transform to primary space
        return x


class SampleGenerator(object):

    def __init__(self):
        print("Constructing NN...")

        # create the network
        # set the parameters
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # enable gpu processing
        # Masks
        masks = torch.from_numpy(np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]] * 4).astype(np.float32))
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
        prior = distributions.MultivariateNormal(torch.zeros(in_dim), torch.eye(in_dim))

        self.net = RealNVP(nets, nett, masks, prior)
        self.net.load_state_dict(torch.load("net.pt", map_location='cpu'))
        self.net.eval()

        print("Done")

    def sample(self):
        x = self.net.sample()
        log_prob = self.net.log_prob(x)
        prob = torch.exp(log_prob)
        x = x.squeeze().clamp(0,1).cpu().detach().numpy()
        prob = prob.cpu().detach().numpy()
        return x,prob


cdef public object createSampleGenerator():
    print("Calling sampler generator constructor...")
    return SampleGenerator()

cdef public void sample(object p, float sampleArray[5], float* pdf):
    x, prob = p.sample()
    pdf[0] = float(prob) # pdf[0] = *pdf
    # copy data into array
    i = 0
    for val in x:
        sampleArray[i]=val
        i+=1

cdef public void samplerTestPrint():
    print("sampler test print worked")


cdef public object createDummyObject():
    return object