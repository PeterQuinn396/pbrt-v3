// samplers/random.cpp*
#include "samplers/learned.h"
#include "RealNVP.h"
#include "paramset.h"
#include "sampling.h"
#include "stats.h"
#include "torch/torch.h"

using namespace torch::nn;

namespace pbrt {

LearnedSampler::LearnedSampler(int ns, int maxDepth, int seed)
    : Sampler(ns),
      maxDepth(maxDepth),
      rng(seed),
      num_features(2 * (maxDepth + 2) + 1),
      net(RealNVP(2 * (maxDepth + 2) + 1)){};

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

    if (!eval) {
        for (int i = 0; i < maxDepth + 2; ++i) {
            Point2f _sample = {
                rng.UniformFloat(),
                rng.UniformFloat()};  // generate a random point for now
            samples2D.emplace_back(_sample);
        }

        // set the sample that will choose which light source to use for
        // direct illumination
        sample1D = rng.UniformFloat();

        *pdf = 1.f;  // to be changed later to the pdf corresponding to the pss
                     // warp
    } else {         // network is trained and ready for use
        torch::Tensor x = net.sample(1);
        float *ptr = x.data<float>();
        for (int i = 0; i < 2; i++) {
            float _x = *ptr;
            ptr++;
            float _y = *ptr;
            ptr++;
            Point2f _sample = {_x, _y};
            samples2D.emplace_back(_sample);
        }
        sample1D = *ptr;
        *pdf = *net.logProb(x).data<float>();
    }
}

void LearnedSampler::saveSample(
    float Li) {  // use the Li.length to get float
                 // save a data sample in a vector b/c we want things to be
                 // dynamic (possibly foolishly)
                 // create an array
    std::vector<float> vec;
    int i = 0;
    for (Point2f sample : samples2D) {
        vec.emplace_back(sample.x);
        vec.emplace_back(sample.y);
    }
    vec.emplace_back(sample1D);
    vec.emplace_back(Li);
    savedData_vec.emplace_back(vec);
    return;
}

void LearnedSampler::setEval() { eval = true; }

void LearnedSampler::train() {  // process the saved data

    // convert saved data to a tensor
    int rows = savedData_vec.size();
    savedData_tensor =
        torch::empty(rows * num_features + 1);  // double check this
    float *data = savedData_tensor.data<float>();

    // iterate over all vals in the vec
    // and store properly in the tensor
    for (const auto &i : savedData_vec) {
        for (const auto &j : i) {
            *data++ = j;
        }
    }

    savedData_tensor.resize_({rows, num_features});

    // get and set device
    c10::DeviceType device;
    if (torch::cuda::is_available) {
        device = c10::DeviceType::CUDA;
    } else {
        device = c10::DeviceType::CPU;
    }
    printf("Running on: %s", device);

    // do the resampling procedure
    torch::Tensor cleaned_data = savedData_tensor;
    cleaned_data.to(device);

    // compute weights for the resampling based on the illumination seen
    int num_data_points = cleaned_data.size(0);
    torch::Tensor resample_rands = torch::randn(num_data_points);
    int data_point_dim = cleaned_data.size(1);
    torch::Tensor weights =
        cleaned_data.narrow(2, data_point_dim - 1, data_point_dim - 1) /
        resample_rands;
    // do weighted sampling
    weights = weights / weights.sum();  // normalize
    int num_of_resamples = 10000;       // to optimize
    torch::Tensor samples = torch::empty(num_of_resamples * num_features);
    float rand;
    int j = 0;
    float *sample_ptr = samples.data<float>();
    float *weight_ptr = weights.data<float>();
    float accumulated_sum = 0;
    for (int i = 0; i < num_of_resamples; i++) {
        rand = rng.UniformFloat();  // get random num
        j = 0;
        accumulated_sum = *weight_ptr;  // first weight
        while (accumulated_sum < rand) {
            weight_ptr++;
            accumulated_sum += *weight_ptr;  // add next weight
            j++;                             // current index
        }
        // add sample at index j to sample tensor
        *sample_ptr = *cleaned_data[j].data<float>();
        sample_ptr++;
    }
    samples.resize_({num_of_resamples, num_features});

    // prepare data loader
    torch::data::DataLoaderOptions options;
    int batchsize = 100;
    options.batch_size(batchsize);
    options.workers(4);
    // options.enforce_ordering = false; // possible optimization in the future
    auto dataset = torch::data::datasets::TensorDataset(samples);
    auto dataloader = torch::data::make_data_loader(dataset, options);  
    
    // create and train the network
    net.to(device);
    net.train();
    torch::optim::Adam optimizer(net.parameters(), /*lr = */ .001);

    for (size_t epoch = 1; epoch <= 5; ++epoch) {
        size_t batch_ind = 0;

        // iterate the dataloader to get the batches
        for (auto& batch : *dataloader) {
            // reset grads
            optimizer.zero_grad();
            // Run model, compute loss
            torch::Tensor batchData = torch::zeros({batchsize, num_features});
            for (int i = 0; i < batchsize; i++) {
                batchData[i] = batch[i].data;
			}
           
            torch::Tensor loss = -net.logProb(batchData).mean();
            loss.backward();
            optimizer.step();

            if (++batch_ind % 100 == 0) {

				printf("Epoch: %i | Batch: %i | Loss: %f", epoch, batch_ind,
                       loss.item<float>());              
                // Serialize your model periodically as a checkpoint.
                //torch::save(net, "net.pt");
            }
        }
    }
    net.eval();
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

}  // namespace pbrt