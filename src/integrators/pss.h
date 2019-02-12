

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
#include "samplers/learned.h"
using namespace std;
namespace pbrt {

// PathIntegrator Declarations
class PSSIntegrator : public Integrator{
  public:
    // PSSIntegrator Public Methods
    PSSIntegrator(int maxDepth, std::shared_ptr<const Camera> camera,
                  std::shared_ptr<Sampler> randSampler,
			      std::shared_ptr<LearnedSampler> learnedSampler, 
				  const Bounds2i &pixelBounds,
                  Float rrThreshold = 1,
                  const std::string &lightSampleStrategy = "spatial",
                  const std::string &pathSampleStrategy = "bsdf",
                  const bool usenee = true);

    void Preprocess(const Scene &scene, Sampler &sampler);

    Spectrum Li(const RayDifferential &ray, const Scene &scene,
                Sampler &randSampler, LearnedSampler &learnedSampler, MemoryArena &arena, int depth) const;

	Spectrum Li_standardPath(const RayDifferential &r,
                                            const Scene &scene,
                                            Sampler &sampler,
                                            MemoryArena &arena,
                                            int depth) const;

    void Render(const Scene &scene);

protected:
    // SamplerIntegrator Protected Data
    std::shared_ptr<const Camera> camera;

  private:
    // PSSIntegrator Private Data
    vector<Point2f> samples;
    int sampleNum;   
    const int maxDepth;
    const Float rrThreshold;  // to remove
    const std::string lightSampleStrategy;
    const std::string pathSampleStrategy;
    const bool usenee;
    std::unique_ptr<LightDistribution> lightDistribution;
    Bounds2i pixelBounds;

	std::shared_ptr<LearnedSampler> learnedSampler;    
    std::shared_ptr<Sampler> randSampler;
        
};

PSSIntegrator *CreatePSSIntegrator(const ParamSet &params,                                  
                                   std::shared_ptr<const Camera> camera);

}  // namespace pbrt

#endif  // PBRT_INTEGRATORS_PSS_H