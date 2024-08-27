#include <NvInfer.h>
#include "cudaWrapper.h"
#include "ioHelper.h"
#include <NvOnnxParser.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <numeric>
#include <stdio.h>
#include <unistd.h>
#include <string.h>


using namespace nvinfer1;
using namespace cudawrapper;

static Logger gLogger;

// Number of times we run inference to calculate average time.
constexpr int ITERATIONS = 10;
// Allow TensorRT to use up to 1GB of GPU memory for tactic selection.
constexpr size_t MAX_WORKSPACE_SIZE = 1ULL << 30; // 1 GB

ICudaEngine* createCudaEngine(std::string const& onnxModelPath, int batchSize, bool float32)
{
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
    std::unique_ptr<nvinfer1::IBuilder, Destroy<nvinfer1::IBuilder>> builder{nvinfer1::createInferBuilder(gLogger)};
    std::unique_ptr<nvinfer1::INetworkDefinition, Destroy<nvinfer1::INetworkDefinition>> network{builder->createNetworkV2(explicitBatch)};
    std::unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser{nvonnxparser::createParser(*network, gLogger)};
    std::unique_ptr<nvinfer1::IBuilderConfig,Destroy<nvinfer1::IBuilderConfig>> config{builder->createBuilderConfig()};

    if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
    {
        std::cout << "ERROR: could not parse input engine." << std::endl;
        return nullptr;
    }

    config->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);
    // use FP16 mode if possible
	if (!float32){
		if (builder->platformHasFastFp16())
		{
			config->setFlag(nvinfer1::BuilderFlag::kFP16);
			std::cout << "Set fp16! " << std::endl;
		}
	}
    builder->setMaxBatchSize(batchSize);
    /*
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims4{1, 3, 256 , 256});
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims4{1, 3, 256 , 256});
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims4{32, 3, 256 , 256});    
    config->addOptimizationProfile(profile);*/

    return builder->buildEngineWithConfig(*network, *config);
}

ICudaEngine* getCudaEngine(std::string const& onnxModelPath, int batchSize, bool is_float32)
{
	std::string fp="_fp16";
	if (is_float32)
		fp="_fp32";
		
    std::string enginePath{getBasename(onnxModelPath) + fp + ".engine"};
    ICudaEngine* engine{nullptr};

    std::string buffer = readBuffer(enginePath);
    
    if (buffer.size())
    {
        // Try to deserialize engine.
        std::unique_ptr<IRuntime, Destroy<IRuntime>> runtime{createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);
		gLogger.log(ILogger::Severity::kINFO, "Loaded from engine");

    }

    if (!engine)
    {
        // Fallback to creating engine from scratch.
        engine = createCudaEngine(onnxModelPath, batchSize, is_float32);
		gLogger.log(ILogger::Severity::kINFO, "Loaded from onnx");

        if (engine)
        {
            std::unique_ptr<IHostMemory, Destroy<IHostMemory>> engine_plan{engine->serialize()};
            // Try to save engine for future uses.
            writeBuffer(engine_plan->data(), engine_plan->size(), enginePath);
        }
    }
    return engine;
}

static int getBindingInputIndex(IExecutionContext* context)
{
    return !context->getEngine().bindingIsInput(0); // 0 (false) if bindingIsInput(0), 1 (true) otherwise
}

void launchInference(IExecutionContext* context, cudaStream_t stream, std::vector<float> const& inputTensor, std::vector<float>& outputTensor, void** bindings, int batchSize)
{
    int inputId = getBindingInputIndex(context);

    cudaMemcpyAsync(bindings[inputId], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    context->enqueueV2(bindings, stream, nullptr);
    cudaMemcpyAsync(outputTensor.data(), bindings[1 - inputId], outputTensor.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
}

float doInference(IExecutionContext* context, cudaStream_t stream, std::vector<float> const& inputTensor, std::vector<float>& outputTensor, void** bindings, int batchSize)
{
    CudaEvent start;
    CudaEvent end;
    double totalTime = 0.0;
	
	//warming up
	launchInference(context, stream, inputTensor, outputTensor, bindings, batchSize);

    for (int i = 0; i < ITERATIONS; ++i)
    {		
        float elapsedTime;

        // Measure time it takes to copy input to GPU, run inference and move output back to CPU.
        cudaEventRecord(start, stream);
        launchInference(context, stream, inputTensor, outputTensor, bindings, batchSize);
        cudaEventRecord(end, stream);

        // Wait until the work is finished.
        cudaStreamSynchronize(stream);
        cudaEventElapsedTime(&elapsedTime, start, end);

        totalTime += elapsedTime;
    }

    std::cout << "Inference batch size " << batchSize << " average over " << ITERATIONS << " runs is " << totalTime / ITERATIONS << "ms" << std::endl;
	return totalTime / ITERATIONS;
}


void write_result(std::string const& onnxModelPath, float res, bool is_float32)
{
	std::string fp="fp16";
	if (is_float32)
		fp="fp32";
	
	// file pointer
	std::fstream fout;
	
	std::string csv_file{getBasename(onnxModelPath) + ".csv"};

	// opens an existing csv file or creates a new file.
	fout.open(csv_file.c_str(), std::ios::out | std::ios::app);

	// Insert the data to file
	fout << "TensorRT" << "; "
		<< getBasename(onnxModelPath) << "; "
		<< fp << "; "
		<< res
		<< "\n";
}

void usage(char *name)
{
    printf("usage: %s -m model_file -p prototxt_file -b mean.binaryproto \n"
                    "\t -d image-file-or-directory [-n iteration]\n"
                    "\t -c Calibrate-directory [-v (validation)] \n"
                    "\t [-e device] [-t FLOAT|HALF|INT8] [-h]\n\n", name);
}

int main(int argc, char* argv[])
{
    // Declaring cuda engine.
    std::unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine{nullptr};
    // Declaring execution context.
    std::unique_ptr<IExecutionContext, Destroy<IExecutionContext>> context{nullptr};
    std::vector<float> inputTensor;
    std::vector<float> outputTensor;
    void* bindings[2]{0};
    CudaStream stream;
	
	bool float32 = true; //default is fp32
	char model[256];
	
	int c;
	while ((c = getopt(argc, argv, "m:p:b:d:n:t:e:c:vh")) != -1) {
        switch (c) {
            case 'm':
                strcpy(model, optarg);
                break;
            case 't':
                if (strstr(optarg, "HALF") != nullptr)
                    float32=false;
                break;
            case 'h':
                usage(argv[0]);
                return 0;
        }
    }

    if (strlen(model) < 1) {
        std::cout << "model file not specified\n";
        return -1;
    }
	std::cout << float32 << std::endl;
	
/*
    if (argc < 2)
    {
        std::cout << "usage: " << argv[0] << " <path_to_model.onnx>" << std::endl;
        return 1;
    }*/

    std::string onnxModelPath(model);
	gLogger.log(ILogger::Severity::kINFO, model);

    const int batchSize = 1;

    // Create Cuda Engine.
    //engine.reset(createCudaEngine(onnxModelPath, batchSize));
    engine.reset(getCudaEngine(onnxModelPath, batchSize, float32));
	
	auto mem_size = engine->getDeviceMemorySize();
	std::cout << "Memory size " << mem_size << std::endl;

    if (!engine)
        return 1;

    // Assume networks takes exactly 1 input tensor and outputs 1 tensor.
	std::cout << "engine->getNbBindings() " << engine->getNbBindings() << std::endl;
    assert(engine->getNbBindings() == 2);
    assert(engine->bindingIsInput(0) ^ engine->bindingIsInput(1));

    for (int i = 0; i < engine->getNbBindings(); ++i)
    {
        Dims dims{engine->getBindingDimensions(i)};
		std::cout << "dims " << dims.d[0] << ";" << dims.d[1] << ";" << dims.d[2] << ";" << dims.d[3] << std::endl;
		std::cout << "nbDims " << dims.nbDims << std::endl;
        
        size_t size = std::accumulate(dims.d+1, dims.d + dims.nbDims, batchSize, std::multiplies<size_t>());
		std::cout << "size " << size << std::endl;
        // Create CUDA buffer for Tensor.
        cudaMalloc(&bindings[i], batchSize * size * sizeof(float));

        // Resize CPU buffers to fit Tensor.
        if (engine->bindingIsInput(i))
            inputTensor.resize(size);
        else
            outputTensor.resize(size);
    }

    // Create Execution Context.
    context.reset(engine->createExecutionContext());

    Dims dims_i{engine->getBindingDimensions(0)};
	dims_i.d[0]=batchSize;
    //Dims4 inputDims{batchSize, dims_i.d[1], dims_i.d[2], dims_i.d[3]};
	
	//std::cout << "inputDims nbDims " << inputDims.nbDims << std::endl;
	std::cout << "inputDims " << batchSize << ";" << dims_i.d[1] << ";" << dims_i.d[2] << ";" << dims_i.d[3] << std::endl;
	
    //context->setBindingDimensions(0, inputDims);
	context->setBindingDimensions(0, dims_i);
	
	std::cout << "start of inference " << std::endl;

    float res = doInference(context.get(), stream, inputTensor, outputTensor, bindings, batchSize);
	
	std::cout << "end of inference " << std::endl;
	
	write_result(onnxModelPath, res, float32);

    for (void* ptr : bindings)
        cudaFree(ptr);

    return 0;
}
