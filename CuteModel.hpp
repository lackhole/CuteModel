//
// Created by YongGyu Lee on 2020-03-26.
//

#ifndef HELLO_LIBS_CUTEMODEL_H
#define HELLO_LIBS_CUTEMODEL_H

#endif //HELLO_LIBS_CUTEMODEL_H

#include <vector>
#include <string>
#include <chrono>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api.h"

#include "tensorflow/lite/delegates/gpu/delegate.h"

#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/nnapi/nnapi_util.h"

namespace ct {
    
    class CuteModel {
    public:
        
        TfLiteModel *model = nullptr;
        TfLiteInterpreterOptions *options = nullptr;
        TfLiteInterpreter *interpreter = nullptr;
        
        TfLiteGpuDelegateOptionsV2 gpuDelegateOptionsV2{};
        TfLiteDelegate *gpuDelegate = nullptr;
        
        tflite::StatefulNnApiDelegate::Options nnApiDelegateOptions{};
        tflite::StatefulNnApiDelegate *nnApiDelegate = nullptr;
    
        /**
         * Constructor / Move Con/Op
         */

//    CuteModel() = delete;
        CuteModel() = default;
        
        ~CuteModel();
        
        CuteModel(void *buffer, size_t bufferSize);
        CuteModel(const char* model_path);
        
        CuteModel(CuteModel &&other);
        
        CuteModel &operator=(CuteModel &&other);
        
        /**
         *
         */
        
        
        void setCpuNumThreads(int numThread);
        
        void setGpuDelegate(
                const TfLiteGpuDelegateOptionsV2 &gpuOptions = TfLiteGpuDelegateOptionsV2Default());
        
        void setNnApiDelegate(
                const tflite::StatefulNnApiDelegate::Options &nnApiOptions = tflite::StatefulNnApiDelegate::Options());
        
        TfLiteInterpreter *buildInterpreter();
        
         bool isBuilt() const;
        
        template<typename Data, typename ...Rest>
        void setInput(const Data &data, const Rest &... data_rest);
        
        template<typename Data>
        void setInput(const Data &data);
        
         void invoke();
        
         long long int invokeGetTime();
        
        template<typename T>
        void getOutput(int index, std::vector<T> &output) const;
        
        template<typename T>
        void getOutput(std::vector<std::vector<T>> &output) const;
        
         int32_t inputTensorCount() const;
        
         int32_t outputTensorCount() const;
        
         size_t inputTensorLength(int index) const;
        
         size_t outputTensorLength(int index) const;
        
         TfLiteTensor *inputTensor(int index);
        
         const TfLiteTensor *inputTensor(int index) const;
        
         const TfLiteTensor *outputTensor(int index) const;
        
        std::string summary() const;
        std::string summarizeOptions() const;
        
        static size_t elementByteSize(const TfLiteTensor *tensor);
        
        static size_t tensorLength(const TfLiteTensor *tensor);
        
        void clear();
    
    private:
        int input_data_index = 0;
        
        CuteModel(const CuteModel &other) = delete;
        
        void operator=(const CuteModel &other) = delete;
    };
    
    bool CuteModel::isBuilt() const {
        return interpreter != NULL;
    }
    
     int32_t CuteModel::inputTensorCount() const {
        return TfLiteInterpreterGetInputTensorCount(interpreter);
    };
    
     int32_t CuteModel::outputTensorCount() const {
        return TfLiteInterpreterGetOutputTensorCount(interpreter);
    };
    
     size_t CuteModel::inputTensorLength(int index) const {
        return tensorLength(inputTensor(index));
    };
    
     size_t CuteModel::outputTensorLength(int index) const {
        return tensorLength(outputTensor(index));
    };
    
     TfLiteTensor *CuteModel::inputTensor(int index) {
        return TfLiteInterpreterGetInputTensor(interpreter, index);
    };
    
     const TfLiteTensor *CuteModel::inputTensor(int index) const {
        return TfLiteInterpreterGetInputTensor(interpreter, index);
    };
    
     const TfLiteTensor *CuteModel::outputTensor(int index) const {
        return TfLiteInterpreterGetOutputTensor(interpreter, index);
    };
    
    template<typename Data>
    void CuteModel::setInput(const Data &data) {
        TfLiteTensorCopyFromBuffer(
                inputTensor(input_data_index),
                data,
                TfLiteTensorByteSize(inputTensor(input_data_index))
        );
        ++input_data_index;
    }
    
    template<typename Data, typename ...Rest>
    void CuteModel::setInput(const Data &data, const Rest &... data_rest) {
        TfLiteTensorCopyFromBuffer(
                inputTensor(input_data_index),
                data,
                TfLiteTensorByteSize(inputTensor(input_data_index))
        );
        ++input_data_index;
        setInput(data_rest...);
    }
    
    void CuteModel::invoke() {
        TfLiteInterpreterInvoke(interpreter);
        input_data_index = 0;
    }
    
    long long int CuteModel::invokeGetTime() {
        auto t_beg = std::chrono::high_resolution_clock::now();
        TfLiteInterpreterInvoke(interpreter);
        input_data_index = 0;
        return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - t_beg).count();
    }
    
    template<typename T>
    void CuteModel::getOutput(std::vector<std::vector<T>> &output) const {
        if(!output.empty())
            output.clear();
        
        output.resize(outputTensorCount());
        for(int i = 0; i < output.size(); ++i) {
            output[i].resize((TfLiteTensorByteSize(outputTensor(i)) / sizeof(T)));
            TfLiteTensorCopyToBuffer(outputTensor(i), output[i].data(),
                                     TfLiteTensorByteSize(outputTensor(i)));
        }
    }
    
    template<typename T>
    void CuteModel::getOutput(int index, std::vector<T> &output) const {
        if(!output.empty())
            output.clear();
        
        output.resize((TfLiteTensorByteSize(outputTensor(index)) / sizeof(T)));
        TfLiteTensorCopyToBuffer(outputTensor(index), output.data(),
                                 TfLiteTensorByteSize(outputTensor(index)));
    }
    
}
