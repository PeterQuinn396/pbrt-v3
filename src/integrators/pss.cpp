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

#include <fstream>
#include <iostream>

namespace pbrt {
STAT_PERCENT("Integrator/Zero-radiance paths", zeroRadiancePaths, totalPaths);
STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);
STAT_COUNTER("Integrator/Camera rays traced", nCameraRays);

// PSSIntegrator Method Definitions

PSSIntegrator::PSSIntegrator(int maxDepth, std::shared_ptr<const Camera> camera,
                             std::shared_ptr<Sampler> randSampler,
                             std::shared_ptr<LearnedSampler> learnedSampler,
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
                           MemoryArena &arena, int depth, bool train) const {
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
        if (usenee && bounces == 0 || specularBounce) {
            // Add emitted light at path vertex or from environment
            if (foundIntersection) {
               // ++totalPaths;
                L += beta * i.Le(-ray.d);
                VLOG(2) << "Added Le -> L = " << L;
               // if (L.IsBlack()) ++zeroRadiancePaths;
                //return L; // emitters do not scatter, return now
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
        if (i.bsdf->NumComponents(BxDFType(BSDF_ALL & ~BSDF_SPECULAR)) >
            0) {  // <--- changed
                  // for generating the training data we only do the explicit
                  // connection on the last bounce
            if (train && bounces == maxDepth - 1) {
                Spectrum Ld =
                    beta * UniformSampleOneLight(i, scene, arena,
                                                 learnedSampler,  
                                                 false, distrib);

                VLOG(2) << "Sampled direct lighting Ld = " << Ld;
                if (Ld.IsBlack()) ++zeroRadiancePaths;
                CHECK_GE(Ld.y(), 0.f);
                L += Ld;
            } else if (!train) {
                // not training, so check if we are at the
                // last bounce or using nee
                if (bounces == maxDepth - 1) {
                    ++totalPaths;
                    Spectrum Ld = beta * UniformSampleOneLight(
                                             i, scene, arena,
                                             learnedSampler,  
                                             false, distrib);

                    VLOG(2) << "Sampled direct lighting Ld = " << Ld;
                    if (Ld.IsBlack()) ++zeroRadiancePaths;
                    CHECK_GE(Ld.y(), 0.f);
                    L += Ld;

                } else if (usenee) {
                    ++totalPaths;
                    Spectrum Ld =
                        beta * UniformSampleOneLight(i, scene, arena,
                                                     randSampler,  // to change
                                                     false, distrib);

                    VLOG(2) << "Sampled direct lighting Ld = " << Ld;
                    if (Ld.IsBlack()) ++zeroRadiancePaths;
                    CHECK_GE(Ld.y(), 0.f);
                    L += Ld;
                }
            }
        }

        if (bounces == maxDepth - 1) break;  // break when we hit the maxDepth

        // Sample BSDF to get new path direction
        // Change to pull from the generated vector sample

        Vector3f wo = -ray.d, wi;  // these should be in world
        // wi = UniformSampleHemisphere(getSample(bounces));
        // sample a uniform hemisphere instead?
        // warp to uniform hemisphere
        Float pdf;
        BxDFType flags =
            BxDFType::BSDF_DIFFUSE;  // should set this to a clear initial value

        Spectrum f;
        Point2f rand2D = learnedSampler.Get2D();

        // --------------------------------------
        // Can use to compare different
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
            f = i.bsdf->Sample_f(wo, &wi, rand2D, &pdf, BSDF_ALL, &flags);
        } else {
            Error("Invalid path construction parameter specified: \"%s\"",
                  pathSampleStrategy);
            exit(2);
        }

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
}  // namespace pbrt

void PSSIntegrator::Render(const Scene &scene) {  // generate samples here

    Preprocess(scene, *randSampler);

    // Compute number of tiles, _nTiles_, to use for parallel rendering
    Bounds2i sampleBounds = camera->film->GetSampleBounds();
    // Vector2i sampleExtent = sampleBounds.Diagonal();
    int x_res = camera->film->fullResolution.x;
    int y_res = camera->film->fullResolution.y;

    // rewritten tracing loop
    // single thread
    bool train = false;
    if (train) {  // do training phase if needed
                  // generate training data
                  // save data to a csv file to be processed by the python code
        int sampleCount = 0;
        ProgressReporter reporter(x_res * y_res, "Generating Training Data");
        std::ofstream csv_file_training;
        csv_file_training.open("training_data.csv");

        for (Point2i pixel : sampleBounds) {  // for each pixel

            MemoryArena arena;

            // initialize/ reset sample counting paramters
            learnedSampler->StartPixel(pixel);
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

                // generate the set of samples that will be used to construct
                // the path save the pdf/jacobian of the warping process
                float warp_pdf = 1.f;

                learnedSampler->GenerateSample(&warp_pdf);

                CameraSample cameraSample;
                // adjust the point where the ray goes through
                // cameraSample.pFilm =  // this can be tweaked so the
                //    (Point2f)pixel +
                //    learnedSampler->Get2D();  // choose the coords in the
                //    pixel
                //                              // to start from
                // cameraSample.pLens = learnedSampler->Get2D();

                // the way its done in Zwicker paper (I think)
                Point2f rand_point = learnedSampler->Get2D();
                cameraSample.pFilm =
                    Point2f(x_res * rand_point[0],
                            y_res * rand_point[1]);  // pick point on film
                cameraSample.pLens = rand_point;     // direction is also
                // determined from point on film

                // I don't think these are used for anything
                cameraSample.time =
                    randSampler->Get1D();  // replace w/ 0 or something

                // Generate camera ray for current sample
                RayDifferential ray;
                Float rayWeight =
                    camera->GenerateRayDifferential(cameraSample, &ray);

                // not exactly sure what this scale factor is doing
                // makes some difference in the noise pattern
                ray.ScaleDifferentials(
                    1 / std::sqrt((Float)learnedSampler->samplesPerPixel));
                ++nCameraRays;

                // Evaluate radiance along camera ray
                Spectrum L(0.f);
                if (rayWeight > 0)
                    L = Li(ray, scene, *randSampler, *learnedSampler, arena,
                           maxDepth, train);

                // Issue warning if unexpected radiance value returned
                if (L.HasNaNs()) {
                    LOG(ERROR) << StringPrintf(
                        "Not-a-number radiance value returned "
                        "for pixel (%d, %d), sample %d. Setting to "
                        "black.",
                        pixel.x, pixel.y,
                        (int)learnedSampler->CurrentSampleNumber());
                    L = Spectrum(0.f);
                } else if (L.y() < -1e-5) {
                    LOG(ERROR) << StringPrintf(
                        "Negative luminance value, %f, returned "
                        "for pixel (%d, %d), sample %d. Setting to "
                        "black.",
                        L.y(), pixel.x, pixel.y,
                        (int)learnedSampler->CurrentSampleNumber());
                    L = Spectrum(0.f);
                } else if (std::isinf(L.y())) {
                    LOG(ERROR) << StringPrintf(
                        "Infinite luminance value returned "
                        "for pixel (%d, %d), sample %d. Setting to "
                        "black.",
                        pixel.x, pixel.y,
                        (int)learnedSampler->CurrentSampleNumber());
                    L = Spectrum(0.f);
                }
                VLOG(1) << "Camera sample: " << cameraSample
                        << " -> ray: " << ray << " -> L = " << L;

                // save camera ray contribution if more than 0 to training data
                if (!L.IsBlack()) {
                    float Li_max_component_val =
                        L.MaxComponentValue();  // would be nice to replace this
                                                // by a norm

                    std::vector<float> data = learnedSampler->getSampleValues();
                    data.emplace_back(Li_max_component_val);

                    std::string delim = "";
                    for (auto val : data) {
                        csv_file_training << delim << val;
                        delim = ",";
                    }
                    csv_file_training << "\n";
                    sampleCount++;
                }

                // Free _MemoryArena_ memory from computing image sample
                // value
                arena.Reset();
                randSampler->StartNextSample();
            } while (learnedSampler->StartNextSample());
            reporter.Update();
        }
        reporter.Done();
        // train network
        printf("\nNumber of samples generated: %i\n", sampleCount);

        csv_file_training.close();

        // call python script here
        printf("Made it to the python training part\n Exiting\n");
        exit(0);
      
       
    }  // end train
	
	 // learnedSampler->setEval();  // initialize the python code (or csv lookup) to generate new samples	
    // eval mode

    train = false;  // make this more elegant later
    ProgressReporter render_reporter(x_res * y_res, "Rendering");
    for (Point2i pixel : sampleBounds) {  // for each pixel

        MemoryArena arena;

        // initialize/ reset sample counting paramters
        learnedSampler->StartPixel(pixel);
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

            // generate the set of samples that will be used to construct
            // the path save the pdf/jacobian of the warping process
            float warp_pdf = 1.f;

            learnedSampler->GenerateSample(&warp_pdf);

            CameraSample cameraSample;
            // adjust the point where the ray goes through
            // cameraSample.pFilm =  // this can be tweaked so the
            //   (Point2f)pixel +
            //   randSampler->Get2D();  // choose the coords in the pixel
            //                             // to start from
            // cameraSample.pLens = randSampler->Get2D();

            // the way its done in Zwicker paper (I think)
            Point2f rand_point = learnedSampler->Get2D();
            cameraSample.pFilm =
                Point2f(x_res * rand_point[0],
                        y_res * rand_point[1]);  // pick point on film
            cameraSample.pLens = rand_point;     // direction is also
                                              // determined from point on film

            // I don't think these are used for anything
            cameraSample.time = randSampler->Get1D();

            // Generate camera ray for current sample
            RayDifferential ray;
            Float rayWeight =
                camera->GenerateRayDifferential(cameraSample, &ray);

            // not exactly sure what this scale factor is doing
            // makes some difference in the noise pattern
            ray.ScaleDifferentials(
                1 / std::sqrt((Float)learnedSampler->samplesPerPixel));
            ++nCameraRays;

            // Evaluate radiance along camera ray
            Spectrum L(0.f);
            if (rayWeight > 0)
                L = Li(ray, scene, *randSampler, *learnedSampler, arena,
                       maxDepth, train);

            // Issue warning if unexpected radiance value returned
            if (L.HasNaNs()) {
                LOG(ERROR) << StringPrintf(
                    "Not-a-number radiance value returned "
                    "for pixel (%d, %d), sample %d. Setting to "
                    "black.",
                    pixel.x, pixel.y,
                    (int)learnedSampler->CurrentSampleNumber());
                L = Spectrum(0.f);
            } else if (L.y() < -1e-5) {
                LOG(ERROR) << StringPrintf(
                    "Negative luminance value, %f, returned "
                    "for pixel (%d, %d), sample %d. Setting to "
                    "black.",
                    L.y(), pixel.x, pixel.y,
                    (int)learnedSampler->CurrentSampleNumber());
                L = Spectrum(0.f);
            } else if (std::isinf(L.y())) {
                LOG(ERROR) << StringPrintf(
                    "Infinite luminance value returned "
                    "for pixel (%d, %d), sample %d. Setting to "
                    "black.",
                    pixel.x, pixel.y,
                    (int)learnedSampler->CurrentSampleNumber());
                L = Spectrum(0.f);
            }
            VLOG(1) << "Camera sample: " << cameraSample << " -> ray: " << ray
                    << " -> L = " << L;

            // Add camera ray's contribution to image
            camera->film->AddSplat(
                cameraSample.pFilm,
                L / learnedSampler->samplesPerPixel/warp_pdf);  // normalize by the spp

            // Free _MemoryArena_ memory from computing image sample
            // value
            arena.Reset();
            randSampler->StartNextSample();
        } while (learnedSampler->StartNextSample());
        render_reporter.Update();
    }

    // Save final image after rendering
    camera->film->WriteImage();
    render_reporter.Done();
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

    std::shared_ptr<Sampler> sampler =
        std::shared_ptr<Sampler>(CreateRandomSampler(params));

    std::shared_ptr<LearnedSampler> learnedSampler =
        std::shared_ptr<LearnedSampler>(new LearnedSampler(ns, maxDepth));

    return new PSSIntegrator(maxDepth, camera, sampler, learnedSampler,
                             pixBounds, rrThreshold, lightStrategy,
                             pathSampleStrategy, nee);
}

}  // namespace pbrt
