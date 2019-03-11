#include "MultivariateNormal.h"

MultivariateNormal::MultivariateNormal(int _dim) { dim = _dim; }

torch::Tensor MultivariateNormal::sample(int batchSize) {
    // assumes identity covariance matrix and 0 mean
    // unfortunatly it doesn't seem like the distributions pytorch package
    // is in C++
    torch::Tensor z = torch::randn({batchSize, dim});
    return z;
}

torch::Tensor MultivariateNormal::logProb(torch::Tensor z) {
    // calculate the log prob of a gaussian with
    // identity covariance matrix and 0 mean
    torch::Tensor prob = torch::zeros({z.size(0), 1});
    prob += LOG_SQRT_INV_2PI * dim;
    torch::Tensor _rows = z.narrow(0, 0, z.size(0));
    prob += -_rows * _rows / 2;
    return prob;
}

