// samplers/random.cpp*
#include "samplers/learned.h"
#include "paramset.h"
#include "sampling.h"
#include "stats.h"
#include "torch/torch.h"


namespace pbrt {

LearnedSampler::LearnedSampler(int ns, int maxDepth, int seed)
    : Sampler(ns), maxDepth(maxDepth), rng(seed) {

	// initialize NN stuff here


}

Float LearnedSampler::Get1D() {
    // commented out whatever this was
    // ProfilePhase _(Prof::GetSample);
    // CHECK_LT(currentPixelSampleIndex, samplesPerPixel);
    return sample1D;
}

Point2f LearnedSampler::Get2D() {
    ProfilePhase _(Prof::GetSample);
    // CHECK_LT(currentPixelSampleIndex, samplesPerPixel);
    CHECK_LT(
        sampleNum,
        maxDepth + 2);  // error if we try to grab more samples than generated
    return samples2D[sampleNum++];  // get the current sample pair and increment
                                    // sample count
}

Point2f LearnedSampler::GetRand2D() {
    ProfilePhase _(Prof::GetSample);
    CHECK_LT(currentPixelSampleIndex, samplesPerPixel);
    return {rng.UniformFloat(), rng.UniformFloat()};
}

void LearnedSampler::GenerateSample(float *pdf) {
    // reset sample num and sample array
    sampleNum = 0;
    samples2D.clear();
    // generate (k+1) pairs of samples, (total of 2(k+1) samples)
    // k is number of segments in path, +1 for choosing the initial point to
    // trace from in pixel need to add +1 to maxdepth again because pbrt counts
    // the number of bounce points, not segments
    for (int i = 0; i < maxDepth + 2; ++i) {
        Point2f _sample = {
            rng.UniformFloat(),
            rng.UniformFloat()};  // generate a random point for now
        samples2D.emplace_back(_sample);
    }

    // set the sample that will choose which light source to use for
    // direct illumination
    sample1D = rng.UniformFloat();

    *pdf = 1.f;  // to be changed later to the pdf corresponding to the pss warp
}

void LearnedSampler::saveSample() {  // save a data sample in a tensor
	
	return;
}

void LearnedSampler::train() { // process the saved data

	// do the resampling procedure 


	// train the network
	return; 
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




// implement the NN
}  // namespace pbrt