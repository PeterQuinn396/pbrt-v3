// integrators/path.cpp*
#include "integrators/pss.h"
#include "bssrdf.h"
#include "camera.h"
#include "film.h"
#include "interaction.h"
#include "paramset.h"
#include "scene.h"
#include "stats.h"

namespace pbrt {
STAT_PERCENT("Integrator/Zero-radiance paths", zeroRadiancePaths, totalPaths);
STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);

// PathIntegrator Method Definitions
PSSIntegrator::PSSIntegrator(int maxDepth, std::shared_ptr<const Camera> camera,
                             std::shared_ptr<Sampler> sampler, //we will just expect a random sampler
                             const Bounds2i &pixelBounds, Float rrThreshold,
                             const std::string &lightSampleStrategy)
    : SamplerIntegrator(camera, sampler, pixelBounds),
      maxDepth(maxDepth),
      rrThreshold(rrThreshold),
      lightSampleStrategy(lightSampleStrategy) {
	//constructor 

	// generate samples here in the initialization of the object so Li can be const as it is in the path integrator
    samples = generateSamples(sampler, maxDepth);

}

void PSSIntegrator::Preprocess(const Scene &scene, Sampler &sampler) {
    lightDistribution =
        CreateLightSampleDistribution(lightSampleStrategy, scene);
}

// generates a vector containing pairs of numbers to use to determine the next direction to bounce
// will be replaced by a learning algorithm later
vector<Vector2f> generateSamples(std::shared_ptr<Sampler> sampler,
                                 int maxDepth) {
	// generate a random pair of points for each time we will need to bounce
	// canonical random numbers
	// Should modify to be maxDepth + 1 and generate the initial ray in this file 
	std::vector<Vector2f> vec;
	for (int i = 0; i < maxDepth; i++) {
		vec.emplace_back(sampler->Get2D); 
    }	
	return vec;
}

// gets the next sample pair from the list generated at the start 
// replaces a random v2f sampling 
Vector2f PSSIntegrator::getSample(int num) const { //has to be const so it can work in const Li
    return samples[num];		
}

Spectrum PSSIntegrator::Li(const RayDifferential &r, const Scene &scene,
                           Sampler &sampler, MemoryArena &arena,
                           int depth) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L(0.f), beta(1.f);
    RayDifferential ray(r); //this will need to be change to be constructed from a PSS sample
    bool specularBounce = false;
    int bounces=0;
	Float etaScale = 1;

	Vector2f sampleA = getSample(bounces);
		
	for (bounces = 0;; ++bounces) {
        // Find next path vertex and accumulate contribution
        VLOG(2) << "Path tracer bounce " << bounces << ", current L = " << L
                << ", beta = " << beta;

		//Intersect ray with scene and store interaction data
        SurfaceInteraction i;
        bool foundIntersection = scene.Intersect(ray, &i);

		//Possibly
	
	}


}

}  // namespace pbrt
