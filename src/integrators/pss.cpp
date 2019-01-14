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
PSSIntegrator::PSSIntegrator(
    int maxDepth, std::shared_ptr<const Camera> camera,
    std::shared_ptr<Sampler> sampler,  // we will just expect a random sampler
    const Bounds2i &pixelBounds, Float rrThreshold,
    const std::string &lightSampleStrategy)
    : SamplerIntegrator(camera, sampler, pixelBounds),
      maxDepth(maxDepth),
      rrThreshold(rrThreshold),
      lightSampleStrategy(lightSampleStrategy) {
    // constructor

    // generate samples here in the initialization of the object so Li can be
    // const as it is in the path integrator
    samples = generateSamples(sampler, maxDepth);
}

void PSSIntegrator::Preprocess(const Scene &scene, Sampler &sampler) {
    lightDistribution =
        CreateLightSampleDistribution(lightSampleStrategy, scene);
}

// generates a vector containing pairs of numbers to use to determine the next
// direction to bounce will be replaced by a learning algorithm later
vector<Point2f> PSSIntegrator::generateSamples(std::shared_ptr<Sampler> sampler,
                                               int maxDepth) {
    // generate a random pair of points for each time we will need to bounce
    // canonical random numbers
    // Should modify to be maxDepth + 1 and generate the initial ray in this
    // file
    std::vector<Point2f> vec;
    for (int i = 0; i < maxDepth; i++) {
        vec.emplace_back(sampler->Get2D());
    }
    return vec;
}

// gets the next sample pair from the list generated at the start
// replaces a random v2f sampling
// has to be const so it can work in const Li
Point2f PSSIntegrator::getSample(int num) const {  	
    return samples[num];
}

Spectrum PSSIntegrator::Li(const RayDifferential &r, const Scene &scene,
                           Sampler &sampler, MemoryArena &arena,
                           int depth) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L(0.f), beta(1.f);
    RayDifferential ray(
        r);  // this will need to be change to be constructed from a PSS sample
    bool specularBounce = false;
    int bounces = 0;
    Float etaScale = 1;

    Point2f sampleA = getSample(bounces);

    for (bounces = 0;bounces>maxDepth; ++bounces) {
        // Find next path vertex and accumulate contribution
        VLOG(2) << "Path tracer bounce " << bounces << ", current L = " << L
                << ", beta = " << beta;

        // Intersect ray with scene and store interaction data
        SurfaceInteraction i;
        bool foundIntersection = scene.Intersect(ray, &i);

        // Possibly add emitted light at intersection

        if (bounces == 0 || specularBounce) {
            // Add emitted light at path vertex or from environment
            if (foundIntersection) {
                L += beta * i.Le(-ray.d);
                VLOG(2) << "Added Le -> L = " << L;
            } else {
                for (const auto &light : scene.infiniteLights)
                    L += beta * light->Le(ray);
                VLOG(2) << "Added infinite area lights -> L = " << L;
            }
        }

        // Terminate path if ray escaped or maxDepth exceeded (not sure why this hasn't been put in the loop declaration)
        if (!foundIntersection || bounces > maxDepth) break;

        // Compute scattering functions and skip over medium boundaries
        i.ComputeScatteringFunctions(ray, arena, true);
        if (!i.bsdf) {
            VLOG(2) << "Skipping intersection due to null bsdf";
            ray = i.SpawnRay(ray.d);
            bounces--;
            continue;
        }

        const Distribution1D *distrib = lightDistribution->Lookup(i.p);

        // Sample illumination from lights to find path contribution/
        // But skip for perfectly specfular BSDFs
		// modified to only do this sampling if we at the end of our path
        if (i.bsdf->NumComponents(BxDFType(BSDF_ALL & ~BSDF_SPECULAR)) > 0 && bounces == maxDepth) {
            ++totalPaths;
            Spectrum Ld = beta * UniformSampleOneLight(i, scene, arena, sampler,
                                                       false, distrib);

            VLOG(2) << "Sampled direct lighting Ld = " << Ld;
            if (Ld.IsBlack()) ++zeroRadiancePaths;
            CHECK_GE(Ld.y(), 0.f);
            L += Ld;
        }

        // Sample BSDF to get new path direction
        // Change to pull from the generated vector sample

        Vector3f wo = -ray.d, wi;
        // wi = UniformSampleHemisphere(getSample(bounces));
        // sample a uniform hemisphere instead?
        // warp to uniform hemisphere
        Float pdf;
        BxDFType flags;
        Spectrum f = i.bsdf->Sample_f(wo, &wi, sampler.Get2D(), &pdf, BSDF_ALL,
                                      &flags);  // remove pdf from here?

        VLOG(2) << "Sampled BSDF, f = " << f << ", pdf = " << pdf;
        if (f.IsBlack() || pdf == 0.f) break;
        beta *= f * AbsDot(wi, i.shading.n) / pdf;
        VLOG(2) << "Updated beta = " << beta;
        CHECK_GE(beta.y(), 0.f);
        DCHECK(!std::isinf(beta.y()));
        specularBounce =
            (flags & BSDF_SPECULAR) != 0;  // Check for a specular bounce
        if ((flags & BSDF_SPECULAR) && (flags & BSDF_TRANSMISSION)) {
            Float eta = i.bsdf->eta;
            // Update the term that tracks radiance scaling for refraction
            // depending on whether the ray is entering or leaving the medium
            etaScale *= (Dot(wo, i.n) > 0) ? (eta * eta) : 1 / (eta * eta);
        }

        ray = i.SpawnRay(wi);

        // Account for subsurface scattering if applicable
        // This should be removed for us, we don't want SSS
		// ---------------------------------------
        if (i.bssrdf && (flags * BSDF_TRANSMISSION)) {
            // Importance sample the BSSRDF
            SurfaceInteraction pi;
            Spectrum S = i.bssrdf->Sample_S(scene, sampler.Get1D(),
                                            sampler.Get2D(), arena, &pi, &pdf);
            DCHECK(!std::isinf(beta.y()));
            if (S.IsBlack() || pdf == 0) break;
            beta *= S / pdf;

            // Account for the direct subsurface scattering component

            L += beta * UniformSampleOneLight(pi, scene, arena, sampler, false,
                                              lightDistribution->Lookup(pi.p));

            // Account for the indirect subsurface scattering component

            Spectrum f = pi.bsdf->Sample_f(pi.wo, &wi, sampler.Get2D(), &pdf,
                                           BSDF_ALL, &flags);

            if (f.IsBlack() || pdf == 0) break;
            beta *= f * AbsDot(wi, pi.shading.n) / pdf;
            DCHECK(!std::isinf(beta.y()));
            specularBounce = (flags & BSDF_SPECULAR) != 0;
            ray = pi.SpawnRay(wi);
        }

		// -------------------------------- end SSS

        // Possibly terminate path with RR
        // Factor out radiance scaling due to refraction in rrBeta
        // We should remove this, we want our paths to be all of the same length
        // (determined by the size of our PSS)

        Spectrum rrBeta = beta * etaScale;
        // check if we have done a min num of bounces and
        // that our leftover radiance is getting quite small
        if (rrBeta.MaxComponentValue() < rrThreshold && bounces > 3) {
            Float q = std::max((Float).05, 1 - rrBeta.MaxComponentValue());
            if (sampler.Get1D() < q) break;
            beta /= 1 - q;
            DCHECK(!std::isinf(beta.y()));
        }
    }

    ReportValue(pathLength, bounces);
    return L; //will have to divide out by the proper pdf we get from our PSS sampling strategy
}

PSSIntegrator *CreatePSSIntegrator(const ParamSet &params,
                                   std::shared_ptr<Sampler> sampler,
                                   std::shared_ptr<const Camera> camera) {
    int maxDepth = params.FindOneInt("maxdepth", 1); // maxDepth
    int np;
    const int *pb = params.FindInt("pixelbounds", &np);
    Bounds2i pixBounds = camera->film->GetSampleBounds();
    if (pb) {
        if (np != 4)
            Error("Expected four values for \"pixelbounds\" parameter. Got %d",
                  np);
        else {
            pixBounds =
                Intersect(pixBounds, Bounds2i{{pb[0], pb[2]}, {pb[1], pb[3]}});
            if (pixBounds.Area() == 0)
                Error("Degenerate \"pixelbounds\" specified");
        }
    }
    Float rrThreshold = params.FindOneFloat("rrthreshold", 1.f);
    std::string lightStrategy =
        params.FindOneString("lightsamplestrategy", "spatial");
    return new PSSIntegrator(maxDepth, camera, sampler, pixBounds, rrThreshold,
                             lightStrategy);
}

}  // namespace pbrt
