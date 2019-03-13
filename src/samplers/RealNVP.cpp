#pragma once

#include "RealNVP.h"
#include "torch/torch.h"



// implement the NN
using namespace torch::nn;
RealNVP::RealNVP(int _num_dim) : torch::nn::Module() {
    int num_dim = _num_dim;
    prior = new MultivariateNormal(num_dim);

    torch::Tensor mask_even = torch::tensor({1.f, 0.f, 1.f, 0.f, 1.f});
    torch::Tensor mask_odd = torch::tensor({0.f, 1.f, 0.f, 1.f, 0.f});

    // construct the different layers
    // each is a seperate NN

    /*torch::nn::Sequential seq(torch::nn::Linear(3, 4), torch::nn::BatchNorm(4),
                              torch::nn::Dropout(0.5));*/

    Sequential test = Sequential();
    auto layer1 = BatchNorm(5);
    test->push_back(layer1);
    test->push_back(Linear(5, 40));

    for (int i = 0; i < num_layers; ++i) {
        // scale NN
        Sequential net_s(
            BatchNorm(num_dim), Linear(num_dim, num_neurons),
            BatchNorm(num_neurons), Functional(torch::relu),  // input block
            Linear(num_neurons, num_neurons), BatchNorm(num_neurons),
            Functional(torch::relu),  // residual block 1
            Linear(num_neurons, num_neurons), BatchNorm(num_neurons),
            Functional(torch::relu),  // residual block 2
            BatchNorm(num_neurons), Functional(torch::relu),
            Linear(num_neurons, num_dim),
            Functional(torch::tanh));  // end block

        // translate NN
        Sequential net_t(
            BatchNorm(num_dim), Linear(num_dim, num_neurons),
            BatchNorm(num_neurons), Functional(torch::relu),  // input block
            Linear(num_dim, num_neurons), BatchNorm(num_neurons),
            Functional(torch::relu),  // residual block 1
            Linear(num_dim, num_neurons), BatchNorm(num_neurons),
            Functional(torch::relu),  // residual block 2
            BatchNorm(num_neurons), Linear(num_dim, num_neurons),
            BatchNorm(num_neurons), Linear(num_neurons, num_dim));  // end block

        s.emplace_back(net_s);
        t.emplace_back(net_t);

        if (i % 2 == 0) {
            masks.emplace_back(mask_even);
        } else {
            masks.emplace_back(mask_odd);
        }
    }
}

torch::Tensor RealNVP::f(
    torch::Tensor x,
    torch::Tensor *jacobian) {  // map from primary to latent space
    auto log_det_J = torch::zeros(x.size(0));

    auto z = x;
    for (int i = 0; i < num_layers; ++i) {
        auto z_ = masks[i] *
                  z;  // apply mask i to input vector, selecting parameters
                      // being sent to NN (part is unchanged after this layer)
        auto s_ = s[i]->forward(z_) *
                  (1 - masks[i]);  // compute the scale factor for this layer
                                   // and select only the active part
        auto t_ = t[i]->forward(z_) *
                  (1 - masks[i]);  // compute the translation factor
        z = z_ + (1 - masks[i]) * (z * torch::exp(s_) + t_);
        log_det_J += s_.sum({1});  // sum along dim 1
    }
    *jacobian = log_det_J;
    return z;
}

torch::Tensor RealNVP::g(torch::Tensor z) {  // map from latent to primary space
    auto x = z;
    // iterate over layers backwards
    for (int i = num_layers - 1; i >= 0; --i) {
        auto x_ = masks[i] * x;
        auto s_ = s[i]->forward(x_) * (1 - masks[i]);
        auto t_ = t[i]->forward(x_) * (1 - masks[i]);
        x = x_ + (1 - masks[i]) * ((x - t_) * torch::exp(-s_));
    }
    return x;
}

torch::Tensor RealNVP::logProb(torch::Tensor x) {
    // compute the log prob of a sample in primary space based on prior dist and
    // learned parameters
    torch::Tensor *log_det_jac;
    auto z = f(x, log_det_jac);
    torch::Tensor prior_log_prob = prior->logProb(z);
    return (*log_det_jac) + prior_log_prob;
}

torch::Tensor RealNVP::sample(int batchsize) {
    torch::Tensor z = prior->sample(batchsize);
    torch::Tensor logp = prior->logProb(z);
    torch::Tensor x = g(z);
    return x;
}