// samplers/random.cpp*
#include "samplers/learned.h"
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

void LearnedSampler::train() {  // process the saved data

    // convert saved data to a tensor
    int rows = savedData_vec.size();
    savedData_tensor = torch::empty(rows * num_features+1); //double check this
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

    // do the resampling procedure <--------------
    torch::Tensor cleaned_data = savedData_tensor;
    cleaned_data.to(device);

    // compute weights for the resampling based on the illumination seen
    int num_data_points = cleaned_data.size[0];
    torch::Tensor resample_rands = torch::randn(num_data_points);
    int data_point_dim = cleaned_data.size[1];
    torch::Tensor weights = cleaned_data.narrow(2, data_point_dim - 1, data_point_dim - 1)/resample_rands;
	// do weighted sampling
    weights = weights / weights.sum(); //normalize

	int num_of_resamples = 10000; // to optimize
    torch::Tensor samples = torch::empty(num_of_resamples * num_features);
    float rand; 
	int j=0;
    float *sample_ptr = samples.data<float>();
    float *weight_ptr = weights.data<float>();
    float accumulated_sum = 0;
	for (int i = 0; i < num_of_resamples; i++) {
        rand = rng.UniformFloat(); // get random num
        j = 0;
        accumulated_sum = *weight_ptr; //first weight
        while (accumulated_sum < rand) {
            weight_ptr++;
            accumulated_sum += *weight_ptr; // add next weight
            j++; // current index
		}
		// add sample at index j to sample tensor
                *sample_ptr = cleaned_data[j];


	}
 

    samples.resize_({num_of_resamples, num_features});

    // prepare data loader
    torch::data::DataLoaderOptions options;
    options.batch_size = 100;
    options.workers = 4;
    // options.enforce_ordering = false; // possible optimization in the future
    auto dataloader = torch::data::make_data_loader(savedData_tensor, options);

    // create and train the network

    net.to(device);
    net.train();
    torch::optim::Adam optimizer(net.parameters(), /*lr = */ .001);

    for (size_t epoch = 1; epoch <= 5; ++epoch) {
        size_t batch_ind = 0;

        // iterate the dataloader to get the batches
        for (auto &batch : *dataloader) {
            // reset grads
            optimizer.zero_grad();
            // Run model, compute loss
            torch::Tensor loss = net.logProb(*batch);
            loss.backward();
            optimizer.step();

            if (++batch_ind % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_ind
                          << " | Loss: " << loss.item<float>() << std::endl;
                // Serialize your model periodically as a checkpoint.
                torch::save(net, "net.pt");
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

// implement the NN

RealNVP::RealNVP(int _num_dim) : torch::nn::Module() {
    int num_dim = 5;
    prior = new MultivariateNormal(num_dim);

    num_layers = 8;
    torch::Tensor mask_even = torch::tensor({1.f, 0.f, 1.f, 0.f, 1.f});
    torch::Tensor mask_odd = torch::tensor({0.f, 1.f, 0.f, 1.f, 0.f});

    // construct the different layers
    // each is a seperate NN
    for (int i = 0; i < num_layers; ++i) {
        // scale NN
        auto net_s = Sequential(
            BatchNorm(num_dim), Linear(num_dim, num_neurons),
            BatchNorm(num_neurons), Functional(torch::relu),  // input block
            Linear(num_neurons, num_neurons), BatchNorm(num_neurons),
            Functional(torch::relu),  // residual block 1
            Linear(num_neurons, num_neurons), BatchNorm(num_neurons),
            Functional(torch::relu),  // residual block 2
            BatchNorm(num_neurons), Functional(torch::relu),
            Linear(num_neurons, num_dim),
            Functional(torch::tanh));  // end block

        // translate NN
        auto net_t = Sequential(
            BatchNorm(num_dim), Linear(num_dim, num_neurons),
            BatchNorm(num_neurons), Functional(torch::relu),  // input block
            Linear(num_dim, num_neurons), BatchNorm(num_neurons),
            Functional(torch::relu),  // residual block 1
            Linear(num_dim, num_neurons), BatchNorm(num_neurons),
            Functional(torch::relu),  // residual block 2
            BatchNorm(num_neurons), Linear(num_dim, num_neurons),
            BatchNorm(num_neurons), Linear(num_neurons, num_dim));  // end block

        s.emplace_back(net_s);
        t.emplace_back(net_t);

        if (i % 2 == 0) {
            masks.emplace_back(mask_even);
        } else {
            masks.emplace_back(mask_odd);
        }
    }
}

torch::Tensor RealNVP::f(
    torch::Tensor x,
    torch::Tensor *jacobian) {  // map from primary to latent space
    auto log_det_J = torch::zeros(x.size[0]);
    auto z = x;
    for (int i = 0; i < num_layers; ++i) {
        auto z_ = masks[i] *
                  z;  // apply mask i to input vector, selecting parameters
                      // being sent to NN (part is unchanged after this layer)
        auto s_ = s[i]->forward(z_) *
                  (1 - masks[i]);  // compute the scale factor for this layer
                                   // and select only the active part
        auto t_ = t[i]->forward(z_) *
                  (1 - masks[i]);  // compute the translation factor
        z = z_ + (1 - masks[i]) * (z * torch::exp(s_) + t_);
        log_det_J += s_.sum({1});  // sum along dim 1
    }
    *jacobian = log_det_J;
    return z;
}

torch::Tensor RealNVP::g(torch::Tensor z) {  // map from latent to primary space
    auto x = z;
    // iterate over layers backwards
    for (int i = num_layers - 1; i >= 0; --i) {
        auto x_ = masks[i] * x;
        auto s_ = s[i]->forward(x_) * (1 - masks[i]);
        auto t_ = t[i]->forward(x_) * (1 - masks[i]);
        x = x_ + (1 - masks[i]) * ((x - t_) * torch::exp(-s_));
    }
    return x;
}

torch::Tensor RealNVP::logProb(torch::Tensor x) {
    // compute the log prob of a sample in primary space based on prior dist and
    // learned parameters
    torch::Tensor *log_det_jac;
    auto z = f(x, log_det_jac);
    torch::Tensor prior_log_prob = prior->logProb(x);
    return (*log_det_jac) + prior_log_prob;
}

}  // namespace pbrt