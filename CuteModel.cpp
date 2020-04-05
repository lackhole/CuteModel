//
// Created by YongGyu Lee on 2020-04-01.
//

//
// Created by YongGyu Lee on 2020-03-26.
//

#ifndef HELLO_LIBS_CUTEMODEL_H
#define HELLO_LIBS_CUTEMODEL_H

#endif //HELLO_LIBS_CUTEMODEL_H

#include <include/tensorflow/lite/c/c_api_internal.h>
#include "cutemodel/CuteModel.hpp"

using namespace ct;

CuteModel::CuteModel(void *buffer, size_t bufferSize)  :
        model(TfLiteModelCreate(buffer, bufferSize)),
        options(TfLiteInterpreterOptionsCreate())
{}

CuteModel::~CuteModel() {
    if(gpuDelegate != nullptr)     TfLiteGpuDelegateV2Delete(gpuDelegate);
    if(options != nullptr)      TfLiteInterpreterOptionsDelete(options);
    if(model != nullptr)        TfLiteModelDelete(model);
    if(interpreter != nullptr)  TfLiteInterpreterDelete(interpreter);
    delete nnApiDelegate;
}

void CuteModel::clear(){
    model = nullptr;
    options = nullptr;
    interpreter = nullptr;
    gpuDelegate = nullptr;
    nnApiDelegate = nullptr;
}

CuteModel::CuteModel(CuteModel &&other) :
        model(other.model),
        options(other.options),
        interpreter(other.interpreter),
        gpuDelegate(other.gpuDelegate),
        nnApiDelegate(other.nnApiDelegate)
{
    other.clear();
}

CuteModel& CuteModel::operator=(CuteModel &&other) {
    if((model == nullptr) & (options == nullptr) & (interpreter == nullptr) & (gpuDelegate == nullptr)){
        model = other.model;
        options = other.options;
        interpreter = other.interpreter;
        gpuDelegate = other.gpuDelegate;
        nnApiDelegate = other.nnApiDelegate;
        
        other.clear();
    }
    
    return *this;
}

void CuteModel::setCpuNumThreads(int numThread) {
    TfLiteInterpreterOptionsSetNumThreads(options, numThread);
}

void CuteModel::setGpuDelegate(const TfLiteGpuDelegateOptionsV2 &gpuDelegate_) {
    gpuDelegateOptionsV2 = gpuDelegate_;
    gpuDelegate = TfLiteGpuDelegateV2Create(&gpuDelegateOptionsV2);
    TfLiteInterpreterOptionsAddDelegate(options, gpuDelegate);
}

void CuteModel::setNnApiDelegate(const tflite::StatefulNnApiDelegate::Options &nnApiOptions) {
    nnApiDelegateOptions = nnApiOptions;
    nnApiDelegate = new tflite::StatefulNnApiDelegate(nnApiDelegateOptions);
    TfLiteInterpreterOptionsAddDelegate(options, nnApiDelegate);
}

TfLiteInterpreter* CuteModel::buildInterpreter() {
    interpreter = TfLiteInterpreterCreate(model, options);
    
    if (interpreter == NULL)
        return NULL;
    
    if(gpuDelegate == nullptr)
        TfLiteInterpreterAllocateTensors(interpreter);
    
    return interpreter;
}

std::string CuteModel::summary() const{
    if(interpreter == nullptr)
        return "";
    
    using std::string;
    using std::to_string;
    
    string log;
    
    log += " Input Tensor\n";
    log += " Number / Name / Byte / Type / Size\n";
    for(int i=0; i<inputTensorCount(); ++i){
        const TfLiteTensor* tensor = inputTensor(i);
        
        log +=  "  #" + std::to_string(i) + " " +
                TfLiteTensorName(tensor) + " " +
                to_string(TfLiteTensorByteSize(tensor)) + " " +
                TfLiteTypeGetName(TfLiteTensorType(tensor)) + " ";
        
        log += to_string(tensor->dims[0].data[0]);
        for(int s = 1; s < tensor->dims[0].size; ++s)
            log += "x" + to_string(tensor->dims[0].data[s]);
        
        log += '\n';
    }log += '\n';
    
    
    log += " Output Tensor\n";
    log += " Number / Name / Byte / Type / Size\n";
    for(int i=0; i<outputTensorCount(); ++i){
        const TfLiteTensor* tensor = outputTensor(i);
        
        log +=  "  #" + std::to_string(i) + " " +
                TfLiteTensorName(tensor) + " " +
                to_string(TfLiteTensorByteSize(tensor)) + " " +
                TfLiteTypeGetName(TfLiteTensorType(tensor)) + " ";
        
        log += to_string(tensor->dims[0].data[0]);
        for(int s = 1; s < tensor->dims[0].size; ++s)
            log += "x" + to_string(tensor->dims[0].data[s]);
        
        log += '\n';
    }log += '\n';
    
    
    return log;
}

std::string CuteModel::summarizeOptions() const {
    
    using std::string;
    using std::to_string;
    
    string log;
    
    log += "Num Threads: ";
    log += (options->num_threads == TfLiteInterpreterOptions::kDefaultNumThreads ? "default\n" : to_string(options->num_threads) + '\n');
    
    log += "Delegates: " + to_string(options->delegates.size()) + '\n';
    
    log += "Use NNAPI: " + (options->use_nnapi ? string("Yes") : string("No"));
    
    if(gpuDelegate != nullptr && options->delegates[0]->data_ != NULL) {
        auto getInferencePriority = [](int priority) -> std::string{
            switch(priority){
                case TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_AUTO: return "Auto";
                case TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION: return "MaxPrecision";
                case TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY: return "MinLatency";
                case TfLiteGpuInferencePriority::TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE: return "MinMemory";
                default: return "Auto";
            }
        };
        
        log += "\nGPU delegate Options:\n";
        log += "    Precision Loss Allowed: " + (gpuDelegateOptionsV2.is_precision_loss_allowed ? string("Yes\n") : string("No\n"));
        log += "    Inference Preference: " + (gpuDelegateOptionsV2.inference_preference ? string("Fast Single Answer\n") : string("Sustained Speed\n"));
        log += "    Inference Priority: " +
                        getInferencePriority(gpuDelegateOptionsV2.inference_priority1) + ' ' +
                        getInferencePriority(gpuDelegateOptionsV2.inference_priority2) + ' ' +
                        getInferencePriority(gpuDelegateOptionsV2.inference_priority3);
    }
    
    if(nnApiDelegate != nullptr && options->delegates[0]->data_ != NULL){
        using ExecutionPreference = tflite::StatefulNnApiDelegate::Options::ExecutionPreference;
        
        const auto getExecutionPreference = [](ExecutionPreference preference) -> std::string {
            switch(preference){
                case ExecutionPreference::kUndefined: return "Undefined";
                case ExecutionPreference::kLowPower: return "LowPower";
                case ExecutionPreference::kFastSingleAnswer: return "FastSingleAnswer";
                case ExecutionPreference::kSustainedSpeed: return "SustainedSpeed";
            }
        };
        
        const auto checkChar = [](const char* str) -> std::string{
            if (str == nullptr)
                return "-";
            return string(str);
        };
        
        log += "\nNNAPI delegate Options:\n";
        log += "    Execution Preference: " + getExecutionPreference(nnApiDelegateOptions.execution_preference) + '\n';
        log += "    Accelerator Name: " + checkChar(nnApiDelegateOptions.accelerator_name) + '\n';
        log += "    Cache Dir: " + checkChar(nnApiDelegateOptions.cache_dir) + '\n';
        log += "    Model Token: " + checkChar(nnApiDelegateOptions.model_token) + '\n';
        log += "    Disallow NNAPI CPU: " + (nnApiDelegateOptions.disallow_nnapi_cpu ? string("Yes\n") : string("No\n"));
        log += "    Max Numper Delegate Partition: " + to_string(nnApiDelegateOptions.max_number_delegated_partitions);
    }
    
    
    return std::move(log);
}


size_t CuteModel::elementByteSize(const TfLiteTensor* tensor) {
    switch(TfLiteTensorType(tensor)){
        case kTfLiteNoType:     return 0;
        case kTfLiteFloat32:    return sizeof(float);
        case kTfLiteInt32:      return sizeof(int32_t);
        case kTfLiteUInt8:      return sizeof(uint8_t);
        case kTfLiteInt64:      return sizeof(int64_t);
        case kTfLiteString:     return strlen(reinterpret_cast<const char*>(TfLiteTensorData(tensor)));
        case kTfLiteBool:       return sizeof(bool);
        case kTfLiteInt16:      return sizeof(int16_t);
        case kTfLiteComplex64:  return sizeof(TfLiteComplex64);
        case kTfLiteInt8:       return sizeof(int8_t);
        case kTfLiteFloat16:    return sizeof(TfLiteFloat16);
    }
}

size_t CuteModel::tensorLength(const TfLiteTensor *tensor) {
    return TfLiteTensorByteSize(tensor) / elementByteSize(tensor);
}