
#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_SAMPLERS_LEARNED_H
#define PBRT_SAMPLERS_LEARNED_H

// samplers/random.h*

#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

#include "rng.h"
#include "sampler.h"


namespace pbrt {

class LearnedSampler : public Sampler {
  public:
    LearnedSampler(int ns, int maxDepth, int seed = 0);
    ~LearnedSampler();
    void StartPixel(const Point2i &);
    void GenerateSample(float *pdf);
    Float Get1D();
    Point2f Get2D();
    Point2f GetRand2D();

    // void train();
    // void saveSample(float Li);  // replace with a write out to .csv

    std::vector<float> getSampleValues();
    // void storeSample(std::vector<float> sample);
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
	
    int sample_index = 0;
    int total_samples = 0;
    bool in_lightdist_sampling = false;

    // csv stuff
    bool usecsv = true;
    int lines_in_csv = 100000;
    int current_line = lines_in_csv;
    int current_file = 1;
    std::vector<std::vector<float>> loaded_data;

    // neural network stuff
    PyObject *net;
    float net_samples[5];
};

Sampler *CreateLearnedSampler(const ParamSet &params);
}  // namespace pbrt

#endif  // PBRT_SAMPLERS_RANDOM_H
