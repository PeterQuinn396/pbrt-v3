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

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

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
    } else if (usecsv) {

		if (current_line == lines_in_csv) { // load next file if we need to
            printf("\nloading file %i", current_file);
            loaded_data.clear();
            std::string name =
                "bsdf\\new_samples_sanmiguel_bsdf_" + std::to_string(current_file) + ".csv";
            std::ifstream in(name);
            std::string line;
            while (std::getline(in, line)) {
                std::stringstream ss(line);
                std::vector<float> row;
                std::string data;
                while (std::getline(ss, data, ',')) {
                    row.push_back(std::stof(data));
				}
                if (row.size() > 0) {
					loaded_data.push_back(row);
				}
			}	
			current_line = 0;
            current_file++;
            if (current_file > 100)
                Error("tried to load too many sample files\n");
		}
		// fill next data point
        int i = 0;
        for (; i < maxDepth; i++) {
            Point2f _sample = {loaded_data[current_line][2 * i],
                               loaded_data[current_line][2 * i + 1]};
            samples2D.emplace_back(_sample);
        }
        sample1D = loaded_data[current_line][2 * i];
        *pdf = loaded_data[current_line][2 * i+1];
        current_line++;
	} 
	else
	{         // network is trained and ready for use

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
        *pdf = net_samples[2 * i + 1];
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
    if (usecsv) {
		printf("Loading in values from csv files\n"); 
	
	} else {
		printf("Initialzing Python Sampler Object...");
		int err = PyImport_AppendInittab("sampler", PyInit_TrainGenSamples);
		if (PyErr_Occurred()) PyErr_Print();
		Py_Initialize();
		// PyRun_SimpleString("import sys"); // for debugging
		// PyRun_SimpleString("print(sys.path)");
		// PyRun_SimpleString("print(sys.exec_prefix)");
		// PyRun_SimpleString("print(sys.version)");	
		PyObject *mod = PyImport_ImportModule("sampler");
		if (PyErr_Occurred()) PyErr_Print();
    
		net = createSampleGenerator();
		if (PyErr_Occurred()) PyErr_Print();
		printf("Python Code loaded succesfully\n");
	}
}

void LearnedSampler::useRandValues() { eval = false;}

void LearnedSampler::useTrainedVals() { eval = true; }

bool LearnedSampler::isUsingRandVals() { return !eval; }

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