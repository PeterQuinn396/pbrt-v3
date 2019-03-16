// samplers/random.cpp*


#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif
#include "TrainGenSamples.h"
#include "samplers/learned.h"

#include "paramset.h"
#include "sampling.h"
#include "stats.h"

namespace pbrt {

LearnedSampler::LearnedSampler(int ns, int maxDepth, int seed)
    : Sampler(ns),
      maxDepth(maxDepth),
      rng(seed),
      num_features(2 * (maxDepth) + 1){};

LearnedSampler::~LearnedSampler() {
    Py_Finalize();
};  // kill the python interpreter

Float LearnedSampler::Get1D() {
    // commented out whatever this was
    // ProfilePhase _(Prof::GetSample);
    // CHECK_LT(currentPixelSampleIndex, samplesPerPixel);
    CHECK_EQ(sampleNum,
             maxDepth);  // these should be equal when this sample gets used
    in_lightdist_sampling = true;
    return sample1D;
}

Point2f LearnedSampler::Get2D() {
    ProfilePhase _(Prof::GetSample);
    // CHECK_LT(currentPixelSampleIndex, samplesPerPixel);

    if (in_lightdist_sampling) {
        return {rng.UniformFloat(), rng.UniformFloat()};
    } else {
        CHECK_LT(
            sampleNum,
            maxDepth);  // error if we try to grab more samples than generated
        return samples2D[sampleNum++];  // get the current sample pair and
                                        // increment
    }                                   // sample count
}

Point2f LearnedSampler::GetRand2D() {
    ProfilePhase _(Prof::GetSample);
    CHECK_LT(currentPixelSampleIndex, samplesPerPixel);
    return {rng.UniformFloat(), rng.UniformFloat()};
}

void LearnedSampler::GenerateSample(float *pdf) {
    // reset sample num and sample array
    sampleNum = 0;
    in_lightdist_sampling = false;
    samples2D.clear();
    // generate (k+1) pairs of samples, (total of 2(k+1) samples)
    // k is number of segments in path, +1 for choosing the initial point to
    // trace from in pixel need to add +1 to maxdepth again because pbrt counts
    // the number of bounce points, not segments

    if (!eval) {
        for (int i = 0; i < maxDepth; ++i) {
            Point2f _sample = {
                rng.UniformFloat(),
                rng.UniformFloat()};  // generate a random point for now
            samples2D.emplace_back(_sample);
        }

        // set the sample that will choose which light source to use for
        // direct illumination
        sample1D = rng.UniformFloat();

        *pdf = 1.f;  // uniform
    } else {         // network is trained and ready for use

        // use the python code to generate a sample
        sample(net, net_samples, pdf);  // pdf gets set here

        // sort the samples into pairs of points
        int i = 0;
        for (; i < maxDepth; i++) {
            Point2f _sample = {net_samples[2 * i], net_samples[2 * i + 1]};
            samples2D.emplace_back(_sample);
        }

        // grab the last sample
        sample1D = net_samples[2 * i];
    }
}

// return all the values in sample as a vector of floats
// used for exporting the training data to a csv file
std::vector<float> LearnedSampler::getSampleValues() {
    std::vector<float> data;
    for (auto &i : samples2D) {
        data.emplace_back(i.x);
        data.emplace_back(i.y);
    }
    data.emplace_back(sample1D);
    return data;
}

void LearnedSampler::setEval() {
    eval = true;
    Py_Initialize();
    initTrainGenSamples();
    net = createSampleGenerator();
}

std::unique_ptr<Sampler> LearnedSampler::Clone(int seed) {
    LearnedSampler *rs = new LearnedSampler(*this);
    rs->rng.SetSequence(seed);
    return std::unique_ptr<Sampler>(rs);
}

void LearnedSampler::StartPixel(const Point2i &p) {
    ProfilePhase _(Prof::StartPixel);
    for (size_t i = 0; i < sampleArray1D.size(); ++i)
        for (size_t j = 0; j < sampleArray1D[i].size(); ++j)
            sampleArray1D[i][j] = rng.UniformFloat();

    for (size_t i = 0; i < sampleArray2D.size(); ++i)
        for (size_t j = 0; j < sampleArray2D[i].size(); ++j)
            sampleArray2D[i][j] = {rng.UniformFloat(), rng.UniformFloat()};
    Sampler::StartPixel(p);
}

// not going to use
Sampler *CreateLearnedSampler(const ParamSet &params) {
    int ns = params.FindOneInt("pixelsamples", 4);
    int maxDepth = params.FindOneInt("maxdepth", 1);  // maxDepth
    return new LearnedSampler(ns, maxDepth);
}

}  // namespace pbrt