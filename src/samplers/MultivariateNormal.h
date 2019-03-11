#pragma once

#include "torch/torch.h"
class MultivariateNormal {  // wrote this class in here b/c I am lazy and this
                            // isn't too big
  public:
    MultivariateNormal(int _dim);
    torch::Tensor sample(int batchSize);
    torch::Tensor logProb(torch::Tensor z);   

  private:
    int dim;
    float const stdDev = 1.f;
    float const mean = 0.f;
    float const SQRT_INV_2PI = 0.3989422804;
    float const LOG_SQRT_INV_2PI = -0.39908993417;
};

