

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_INTEGRATORS_PSS_H
#define PBRT_INTEGRATORS_PSS_H

// integrators/path.h*
#include "integrator.h"
#include "lightdistrib.h"
#include "pbrt.h"
using namespace std;
namespace pbrt {

// PathIntegrator Declarations
class PSSIntegrator : public SamplerIntegrator {
  public:
    // PSSIntegrator Public Methods
    PSSIntegrator(int maxDepth, std::shared_ptr<const Camera> camera,
                   std::shared_ptr<Sampler> sampler,
                   const Bounds2i &pixelBounds, Float rrThreshold = 1,
                   const std::string &lightSampleStrategy = "spatial");

    void Preprocess(const Scene &scene, Sampler &sampler);
    Spectrum Li(const RayDifferential &ray, const Scene &scene,
                Sampler &sampler, MemoryArena &arena, int depth) const;

  private:
    // PSSIntegrator Private Data
	vector<Point2f> samples;
    int sampleNum;

    const int maxDepth;
    const Float rrThreshold; //to remove
    const std::string lightSampleStrategy;
    std::unique_ptr<LightDistribution> lightDistribution;
   

	//private methods
    //void generateSamples(Sampler &sampler);
    Point2f getSample(int num) const;
    vector<Point2f> generateSamples(std::shared_ptr<Sampler> sampler,
                                    int maxDepth);
};

PSSIntegrator *CreatePSSIntegrator(const ParamSet &params,
                                     std::shared_ptr<Sampler> sampler,
                                     std::shared_ptr<const Camera> camera);

}  // namespace pbrt

#endif  // PBRT_INTEGRATORS_PSS_H