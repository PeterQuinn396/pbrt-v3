
#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_SAMPLERS_LEARNED_H
#define PBRT_SAMPLERS_LEARNED_H

// samplers/random.h*
#include "sampler.h"
#include "rng.h"

namespace pbrt {

	class LearnedSampler : public Sampler {
	public:
		LearnedSampler(int ns, int seed = 0);
		void StartPixel(const Point2i &);
		Float Get1D();
		Point2f Get2D();
		std::unique_ptr<Sampler> Clone(int seed);

	private:
		RNG rng;
	};

	Sampler *CreateLearnedSampler(const ParamSet &params);

}  // namespace pbrt

#endif  // PBRT_SAMPLERS_RANDOM_H



