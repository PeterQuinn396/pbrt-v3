// integrators/path.cpp*
#include "integrators/pss.h"
#include "bssrdf.h"
#include "camera.h"
#include "film.h"
#include "integrator.h"
#include "interaction.h"
#include "parallel.h"
#include "paramset.h"
#include "progressreporter.h"
#include "samplers/learned.h"
#include "samplers/random.h"
#include "scene.h"
#include "stats.h"

namespace pbrt {
STAT_PERCENT("Integrator/Zero-radiance paths", zeroRadiancePaths, totalPaths);
STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);
STAT_COUNTER("Integrator/Camera rays traced", nCameraRays);

// PSSIntegrator Method Definitions

PSSIntegrator::PSSIntegrator(int maxDepth, std::shared_ptr<const Camera> camera,
                             std::shared_ptr<Sampler> randSampler,
                             LearnedSampler &learnedSampler,
                             const Bounds2i &pixelBounds, Float rrThreshold,
                             const std::string &lightSampleStrategy,
                             const std::string &pathSampleStrategy,
                             const bool usenee)
    : camera(camera),
      randSampler(randSampler),
      learnedSampler(learnedSampler),
      pixelBounds(pixelBounds),
      maxDepth(maxDepth),
      rrThreshold(rrThreshold),
      lightSampleStrategy(lightSampleStrategy),
      pathSampleStrategy(pathSampleStrategy),
      usenee(usenee) {}

void PSSIntegrator::Preprocess(const Scene &scene, Sampler &sampler) {
    lightDistribution =
        CreateLightSampleDistribution(lightSampleStrategy, scene);
}

Spectrum PSSIntegrator::Li(const RayDifferential &r, const Scene &scene,
                           Sampler &randSampler, LearnedSampler &learnedSampler,
                           MemoryArena &arena, int depth) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L(0.f), beta(1.f);
    RayDifferential ray(r);  
    bool specularBounce = false;
    int bounces = 0;
    Float etaScale = 1;

    for (bounces = 0;; ++bounces) {
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

        // Terminate path if ray escaped or maxDepth exceeded (not sure why this
        // hasn't been put in the loop declaration)
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
        if (i.bsdf->NumComponents(BxDFType(BSDF_ALL & ~BSDF_SPECULAR)) > 0 &&
            (bounces == maxDepth || usenee)) {  // <--- changed
            ++totalPaths;
            Spectrum Ld = beta * UniformSampleOneLight(
                                     i, scene, arena, randSampler,  // to change
                                     false, distrib);

            VLOG(2) << "Sampled direct lighting Ld = " << Ld;
            if (Ld.IsBlack()) ++zeroRadiancePaths;
            CHECK_GE(Ld.y(), 0.f);
            L += Ld;
        }

		if (bounces == maxDepth) break; // break when we hit the maxDepth - 1
		
        // Sample BSDF to get new path direction
        // Change to pull from the generated vector sample

        Vector3f wo = -ray.d, wi;  // these should be in world
        // wi = UniformSampleHemisphere(getSample(bounces));
        // sample a uniform hemisphere instead?
        // warp to uniform hemisphere
        Float pdf;
        BxDFType flags;

        Spectrum f;
        Point2f rand2D = learnedSampler.Get2D();

        // -------------------------------------- Can use to compare different
        // sampling strategies for example, is it better to learn to
        // sample/select paths from a unifrom hemisphere, or do we
        // preweight/bias it by cosine/bsdf?
        if (pathSampleStrategy == "uniform") {
            wi = i.bsdf->LocalToWorld(UniformSampleHemisphere(rand2D));
            pdf = UniformHemispherePdf();
            f = i.bsdf->f(wo, wi);
        } else if (pathSampleStrategy == "cosine") {
            wi = CosineSampleHemisphere(rand2D);
            pdf = CosineHemispherePdf(wi.z);  // local
            wi = i.bsdf->LocalToWorld(wi);
            f = i.bsdf->f(wo, wi);
        } else if (pathSampleStrategy == "bsdf") {
            f = i.bsdf->Sample_f(wo, &wi, rand2D, &pdf, BSDF_ALL,
                                 &flags);  // remove pdf from here?
        } else {
            Error("Invalid path construction parameter specified: \"%s\"",
                  pathSampleStrategy);
            exit(2);
        }

        // Spectrum f = i.bsdf->Sample_f(wo, &wi, sampler.Get2D(), &pdf,
        // BSDF_ALL,
        //                              &flags);  // remove pdf from here?

        // ----------------------------------------

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
            Spectrum S =
                i.bssrdf->Sample_S(scene, randSampler.Get1D(),
                                   randSampler.Get2D(), arena, &pi, &pdf);
            DCHECK(!std::isinf(beta.y()));
            if (S.IsBlack() || pdf == 0) break;
            beta *= S / pdf;

            // Account for the direct subsurface scattering component

            L += beta * UniformSampleOneLight(pi, scene, arena, randSampler,
                                              false,
                                              lightDistribution->Lookup(pi.p));

            // Account for the indirect subsurface scattering component

            Spectrum f = pi.bsdf->Sample_f(pi.wo, &wi, randSampler.Get2D(),
                                           &pdf, BSDF_ALL, &flags);

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
            if (randSampler.Get1D() < q) break;
            beta /= 1 - q;
            DCHECK(!std::isinf(beta.y()));
        }
    }

    ReportValue(pathLength, bounces);
    return L;  // will have to divide out by the proper pdf we get from our PSS
               // sampling strategy
}

void PSSIntegrator::Render(const Scene &scene) {  // generate samples here

    Preprocess(scene, *randSampler);

    // Compute number of tiles, _nTiles_, to use for parallel rendering
    Bounds2i sampleBounds = camera->film->GetSampleBounds();
    Vector2i sampleExtent = sampleBounds.Diagonal();
    const int tileSize = 16;  // to change
    Point2i nTiles((sampleExtent.x + tileSize - 1) / tileSize,
                   (sampleExtent.y + tileSize - 1) / tileSize);
    ProgressReporter reporter(
        camera->film->fullResolution.x * camera->film->fullResolution.y,
        "Rendering");

    // rewritten tracing loop
    // single thread

    for (Point2i pixel : sampleBounds) {  // for each pixel

        MemoryArena arena;

		// initialize/ reset sample counting paramters
        learnedSampler.StartPixel(pixel);
        randSampler->StartPixel(pixel);

        // Do this check after the StartPixel() call; this keeps
        // the usage of RNG values from (most) Samplers that use
        // RNGs consistent, which improves reproducability /
        // debugging.
        if (!InsideExclusive(pixel, pixelBounds)) continue;
        int seed = pixel.x * pixel.y + pixel.x;  // whatever

        // std::unique_ptr<Sampler> tileSampler = sampler->Clone(seed);

        do {
            // Initialize _CameraSample_ for current sample
            // CameraSample cameraSample = sampler->GetCameraSample(pixel);

            // generate the set of samples that will be used to construct the
            // path save the pdf/jacobian of the warping process
            float warp_pdf = 1.f;
            
            learnedSampler.GenerateSample(&warp_pdf);

            CameraSample cameraSample;
            cameraSample.pFilm =
                (Point2f)pixel +
                learnedSampler
                    .Get2D();  // choose the coords in the pixel to start from

			cameraSample.pLens = learnedSampler.Get2D();

            // I don't think these are used for anything
            cameraSample.time = randSampler->Get1D();
            

            // Generate camera ray for current sample
            RayDifferential ray;
            Float rayWeight =
                camera->GenerateRayDifferential(cameraSample, &ray);

            // not exactly sure what this scale factor is doing
			// makes some difference in the noise pattern
            ray.ScaleDifferentials(1 / std::sqrt((Float)learnedSampler.samplesPerPixel));
            ++nCameraRays;

            // Evaluate radiance along camera ray
            Spectrum L(0.f);
            if (rayWeight > 0)
                 L = Li(ray, scene, *randSampler, learnedSampler, arena, maxDepth);
               // L = Li_standardPath(ray, scene, *randSampler, arena, maxDepth); // works perfectly
            // Issue warning if unexpected radiance value returned
            if (L.HasNaNs()) {
                LOG(ERROR) << StringPrintf(
                    "Not-a-number radiance value returned "
                    "for pixel (%d, %d), sample %d. Setting to "
                    "black.",
                    pixel.x, pixel.y,
                    (int)learnedSampler.CurrentSampleNumber());
                L = Spectrum(0.f);
            } else if (L.y() < -1e-5) {
                LOG(ERROR) << StringPrintf(
                    "Negative luminance value, %f, returned "
                    "for pixel (%d, %d), sample %d. Setting to "
                    "black.",
                    L.y(), pixel.x, pixel.y,
                    (int)learnedSampler.CurrentSampleNumber());
                L = Spectrum(0.f);
            } else if (std::isinf(L.y())) {
                LOG(ERROR) << StringPrintf(
                    "Infinite luminance value returned "
                    "for pixel (%d, %d), sample %d. Setting to "
                    "black.",
                    pixel.x, pixel.y,
                    (int)learnedSampler.CurrentSampleNumber());
                L = Spectrum(0.f);
            }
            VLOG(1) << "Camera sample: " << cameraSample << " -> ray: " << ray
                    << " -> L = " << L;

            // Add camera ray's contribution to image
            camera->film->AddSplat(
                cameraSample.pFilm,
                L / learnedSampler.samplesPerPixel);  // normalize by the spp 
						
            // Free _MemoryArena_ memory from computing image sample
            // value
            arena.Reset();
            randSampler->StartNextSample();
        } while (learnedSampler.StartNextSample());
        reporter.Update();
    }
    reporter.Done();
    // Save final image after rendering
    camera->film->WriteImage();

    // -------------------------------------------

    //
    //    ParallelFor2D(
    //        [&](Point2i tile) {
    //            // Render section of image corresponding to _tile_

    //            // Allocate _MemoryArena_ for tile
    //            MemoryArena arena;

    //            // Get sampler instance for tile
    //            int seed = tile.y * nTiles.x + tile.x;
    //            std::unique_ptr<Sampler> tileSampler = sampler->Clone(seed);

    //            // Compute sample bounds for tile
    //            int x0 = sampleBounds.pMin.x + tile.x * tileSize;
    //            int x1 = std::min(x0 + tileSize, sampleBounds.pMax.x);
    //            int y0 = sampleBounds.pMin.y + tile.y * tileSize;
    //            int y1 = std::min(y0 + tileSize, sampleBounds.pMax.y);
    //            Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));
    //            LOG(INFO) << "Starting image tile " << tileBounds;

    //            // Get _FilmTile_ for tile
    //            std::unique_ptr<FilmTile> filmTile =
    //                camera->film->GetFilmTile(tileBounds);

    //            // Loop over pixels in tile to render them
    //            for (Point2i pixel : tileBounds) {
    //                {
    //                    ProfilePhase pp(Prof::StartPixel);
    //                    tileSampler->StartPixel(pixel);
    //                }

    //                // Do this check after the StartPixel() call; this keeps
    //                // the usage of RNG values from (most) Samplers that use
    //                // RNGs consistent, which improves reproducability /
    //                // debugging.
    //                if (!InsideExclusive(pixel, pixelBounds)) continue;

    //                do {
    //                    // Initialize _CameraSample_ for current sample
    //                    CameraSample cameraSample =
    //                        tileSampler->GetCameraSample(pixel);

    //                    // Generate camera ray for current sample
    //                    RayDifferential ray;
    //                    Float rayWeight =
    //                        camera->GenerateRayDifferential(cameraSample,
    //                        &ray);
    //                    ray.ScaleDifferentials(
    //                        1 /
    //                        std::sqrt((Float)tileSampler->samplesPerPixel));
    //                    ++nCameraRays;

    //                    // Evaluate radiance along camera ray
    //                    Spectrum L(0.f);
    //                    if (rayWeight > 0)
    //                        L = Li(ray, scene, *tileSampler, arena, maxDepth);

    //                    // Issue warning if unexpected radiance value returned
    //                    if (L.HasNaNs()) {
    //                        LOG(ERROR) << StringPrintf(
    //                            "Not-a-number radiance value returned "
    //                            "for pixel (%d, %d), sample %d. Setting to "
    //                            "black.",
    //                            pixel.x, pixel.y,
    //                            (int)tileSampler->CurrentSampleNumber());
    //                        L = Spectrum(0.f);
    //                    } else if (L.y() < -1e-5) {
    //                        LOG(ERROR) << StringPrintf(
    //                            "Negative luminance value, %f, returned "
    //                            "for pixel (%d, %d), sample %d. Setting to "
    //                            "black.",
    //                            L.y(), pixel.x, pixel.y,
    //                            (int)tileSampler->CurrentSampleNumber());
    //                        L = Spectrum(0.f);
    //                    } else if (std::isinf(L.y())) {
    //                        LOG(ERROR) << StringPrintf(
    //                            "Infinite luminance value returned "
    //                            "for pixel (%d, %d), sample %d. Setting to "
    //                            "black.",
    //                            pixel.x, pixel.y,
    //                            (int)tileSampler->CurrentSampleNumber());
    //                        L = Spectrum(0.f);
    //                    }
    //                    VLOG(1) << "Camera sample: " << cameraSample
    //                            << " -> ray: " << ray << " -> L = " << L;

    //                    // Add camera ray's contribution to image
    //                    filmTile->AddSample(cameraSample.pFilm, L, rayWeight);

    //                    // Free _MemoryArena_ memory from computing image
    //                    sample
    //                    // value
    //                    arena.Reset();
    //                } while (tileSampler->StartNextSample());
    //            }
    //            LOG(INFO) << "Finished image tile " << tileBounds;

    //            // Merge image tile into _Film_
    //            camera->film->MergeFilmTile(std::move(filmTile));
    //            reporter.Update();
    //        },
    //        nTiles);
    //    reporter.Done();
    //}
    // LOG(INFO) << "Rendering finished";

    //// Save final image after rendering
    // camera->film->WriteImage();
    // --------------------------------------------------------------
}

PSSIntegrator *CreatePSSIntegrator(const ParamSet &params,
                                   std::shared_ptr<const Camera> camera) {
    int maxDepth = params.FindOneInt("maxdepth", 1);  // maxDepth
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

    std::string pathSampleStrategy =
        params.FindOneString("pathsamplestrategy", "bsdf");
    bool nee = params.FindOneBool("usenee", true);

	  int ns = params.FindOneInt("pixelsamples", 4);

    // set sampler, maybe modify?
    std::shared_ptr<Sampler> sampler =
        std::shared_ptr<Sampler>(CreateRandomSampler(params));  

    LearnedSampler *learnedSampler = new LearnedSampler(ns, maxDepth);

    return new PSSIntegrator(maxDepth, camera, sampler, *learnedSampler,
                             pixBounds, rrThreshold, lightStrategy,
                             pathSampleStrategy, nee);
}


Spectrum PSSIntegrator::Li_standardPath(const RayDifferential &r, const Scene &scene,
                            Sampler &sampler, MemoryArena &arena,
                            int depth) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L(0.f), beta(1.f);
    RayDifferential ray(r);
    bool specularBounce = false;
    int bounces;
    // Added after book publication: etaScale tracks the accumulated effect
    // of radiance scaling due to rays passing through refractive
    // boundaries (see the derivation on p. 527 of the third edition). We
    // track this value in order to remove it from beta when we apply
    // Russian roulette; this is worthwhile, since it lets us sometimes
    // avoid terminating refracted rays that are about to be refracted back
    // out of a medium and thus have their beta value increased.
    Float etaScale = 1;

    for (bounces = 0;; ++bounces) {
        // Find next path vertex and accumulate contribution
        VLOG(2) << "Path tracer bounce " << bounces << ", current L = " << L
                << ", beta = " << beta;

        // Intersect _ray_ with scene and store intersection in _isect_
        SurfaceInteraction isect;
        bool foundIntersection = scene.Intersect(ray, &isect);

        // Possibly add emitted light at intersection
        if (bounces == 0 || specularBounce) {
            // Add emitted light at path vertex or from the environment
            if (foundIntersection) {
                L += beta * isect.Le(-ray.d);
                VLOG(2) << "Added Le -> L = " << L;
            } else {
                for (const auto &light : scene.infiniteLights)
                    L += beta * light->Le(ray);
                VLOG(2) << "Added infinite area lights -> L = " << L;
            }
        }

        // Terminate path if ray escaped or _maxDepth_ was reached
        if (!foundIntersection || bounces >= maxDepth) break;

        // Compute scattering functions and skip over medium boundaries
        isect.ComputeScatteringFunctions(ray, arena, true);
        if (!isect.bsdf) {
            VLOG(2) << "Skipping intersection due to null bsdf";
            ray = isect.SpawnRay(ray.d);
            bounces--;
            continue;
        }

        const Distribution1D *distrib = lightDistribution->Lookup(isect.p);

        // Sample illumination from lights to find path contribution.
        // (But skip this for perfectly specular BSDFs.)
        if (isect.bsdf->NumComponents(BxDFType(BSDF_ALL & ~BSDF_SPECULAR)) >
            0) {
            ++totalPaths;
            Spectrum Ld = beta * UniformSampleOneLight(isect, scene, arena,
                                                       sampler, false, distrib);
            VLOG(2) << "Sampled direct lighting Ld = " << Ld;
            if (Ld.IsBlack()) ++zeroRadiancePaths;
            CHECK_GE(Ld.y(), 0.f);
            L += Ld;
        }

        // Sample BSDF to get new path direction
        Vector3f wo = -ray.d, wi;
        Float pdf;
        BxDFType flags;
        Spectrum f = isect.bsdf->Sample_f(wo, &wi, sampler.Get2D(), &pdf,
                                          BSDF_ALL, &flags);
        VLOG(2) << "Sampled BSDF, f = " << f << ", pdf = " << pdf;
        if (f.IsBlack() || pdf == 0.f) break;
        beta *= f * AbsDot(wi, isect.shading.n) / pdf;
        VLOG(2) << "Updated beta = " << beta;
        CHECK_GE(beta.y(), 0.f);
        DCHECK(!std::isinf(beta.y()));
        specularBounce = (flags & BSDF_SPECULAR) != 0;
        if ((flags & BSDF_SPECULAR) && (flags & BSDF_TRANSMISSION)) {
            Float eta = isect.bsdf->eta;
            // Update the term that tracks radiance scaling for refraction
            // depending on whether the ray is entering or leaving the
            // medium.
            etaScale *= (Dot(wo, isect.n) > 0) ? (eta * eta) : 1 / (eta * eta);
        }
        ray = isect.SpawnRay(wi);

        // Account for subsurface scattering, if applicable
        if (isect.bssrdf && (flags & BSDF_TRANSMISSION)) {
            // Importance sample the BSSRDF
            SurfaceInteraction pi;
            Spectrum S = isect.bssrdf->Sample_S(
                scene, sampler.Get1D(), sampler.Get2D(), arena, &pi, &pdf);
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

        // Possibly terminate the path with Russian roulette.
        // Factor out radiance scaling due to refraction in rrBeta.
        Spectrum rrBeta = beta * etaScale;
        if (rrBeta.MaxComponentValue() < rrThreshold && bounces > 3) {
            Float q = std::max((Float).05, 1 - rrBeta.MaxComponentValue());
            if (sampler.Get1D() < q) break;
            beta /= 1 - q;
            DCHECK(!std::isinf(beta.y()));
        }
    }
    ReportValue(pathLength, bounces);
    return L;
}


}  // namespace pbrt
