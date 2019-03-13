
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
#include "RealNVP.h"


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
    void saveSample(float Li);
    void setEval();

    std::unique_ptr<Sampler> Clone(int seed);

  private:
    RNG rng;
    int sampleNum = 0;
    int maxDepth;
    std::vector<Point2f> samples2D;
    Float sample1D;
    int num_features;
    bool eval = false;

    torch::Tensor savedData_tensor;
    std::vector<std::vector<float>> savedData_vec;
    RealNVP net;
};

Sampler *CreateLearnedSampler(const ParamSet &params);
}  // namespace pbrt

#endif  // PBRT_SAMPLERS_RANDOM_H
