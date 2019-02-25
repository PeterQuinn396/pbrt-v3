
#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_SAMPLERS_LEARNED_H
#define PBRT_SAMPLERS_LEARNED_H

// samplers/random.h*
#include <random>  // for gaussian
#include "rng.h"
#include "sampler.h"
#include "torch/torch.h"

namespace pbrt {

class LearnedSampler : public Sampler {
  public:
    LearnedSampler(int ns, int maxDepth, int seed = 0);
    void StartPixel(const Point2i &);
    void GenerateSample(float *pdf);
    Float Get1D();
    Point2f Get2D();
    Point2f GetRand2D();

    void train();
    void saveSample();

    std::unique_ptr<Sampler> Clone(int seed);
    bool trainMode = true;

  private:
    RNG rng;
    int sampleNum = 0;
    int maxDepth;
    std::vector<Point2f> samples2D;
    Float sample1D;
};

class RealNVP : public torch::nn::Module {
  public:
    RealNVP(net_s, net_t, masks, prior);
    torch::Tensor f(torch::Tensor x);
    torch::Tensor g(torch::Tensor z);
    float logProb(torch::Tensor x);

  private:
};

class MultivariateNormal {  // wrote this class in here b/c I am lazy and this
                            // isn't too big
  public:
    MultivariateNormal(int _dim) {
        dim = _dim;
        // should probably add some error checks on the matrix sizes
    }

    torch::Tensor sample(int batchSize = 1) {
        // assumes identity covariance matrix and 0 mean
		// unfortunatly it doesn't seem like the distributions pytorch package is in C++
        torch::Tensor z = torch::randn({batchSize, dim});
        return z;
    }

    torch::Tensor logProb(torch::Tensor z) { 		
		// calculate the log prob of a gaussian with 
		// identity covariance matrix and 0 mean
		torch::Tensor prob = torch::zeros({z.size[0], 1});
        prob += LOG_SQRT_INV_2PI * dim;
        torch::Tensor _rows = z.narrow(0, 0, z.size[0]);
        prob += -_rows * _rows / 2;
        return prob;	
	}

  private:
    int dim;
    float stdDev = 1.f;
    float mean = 0.f;
    float SQRT_INV_2PI = 0.3989422804;
    float LOG_SQRT_INV_2PI = -0.39908993417
};

Sampler *CreateLearnedSampler(const ParamSet &params);

}  // namespace pbrt

#endif  // PBRT_SAMPLERS_RANDOM_H
