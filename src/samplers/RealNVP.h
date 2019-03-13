#pragma once

#include "torch/torch.h"
#include "MultivariateNormal.h"


class RealNVP : public torch::nn::Module {
public:
	RealNVP(int num_dim);
	torch::Tensor f(torch::Tensor x, torch::Tensor *jacobian);
	torch::Tensor g(torch::Tensor z);
	torch::Tensor logProb(torch::Tensor x);
	torch::Tensor sample(int batchsize);

private:
	std::vector<torch::nn::Sequential> s;
	std::vector<torch::nn::Sequential> t;
	std::vector<torch::Tensor> masks;
	MultivariateNormal* prior;
	int num_layers = 8;
	int num_neurons = 40;
};
