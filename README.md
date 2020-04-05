# CuteModel
TensorFlow Lite C API Wrapper Class (+ GPU &amp; NNAPI)


## Example

### Build
```
// build from file
ct::CuteModel cuteModel("converted_model.tflite");
```
```
// build from buffer
const char* buffer = ...;
int len = ...;

ct::CuteModel cuteModel(buffer, len);
```

### Set Input
```
using namespace ct;
float* data[] = { ... };

// set input with loop
for(int i=0; i<cuteModel.inputTensorCount(); ++i)
  cuteModel.setInput(data[i]);

// or set input manually
cuteModel.setInput(data[0], data[1], data[2]);
```

### Invoke & get Output
```
cuteModel.setCpuNumThreads(3);
cuteModel.buildInterpreter();
cuteModel.invoke();

// getting a whole output
std::vector<std::vector<int>> out;  // data type must match model's out tensor type. 
cuteModel.getOutput(out);

// or get a specific index of output
std::vector<int> out_first;
cuteModel.getOutput(0, out_first);
```

### Delegate
```
// GPU Delegate
TfLiteGpuDelegateOptionsV2 gpuOptions = TfLiteGpuDelegateOptionsV2Default();
cuteModel.setGpuDelegate(gpuOptions);

// NNAPI delegate
tflite::StatefulNnApiDelegate::Options nnApiOptions = tflite::StatefulNnApiDelegate::Options()
cuteModel.setNnApiDelegate(nnApiOptions);
```

### Etc
```
// Check if the interpreter is built
cuteMoel.isBuilt(); 

// invoke and returns elapsed time
cuteModel.invokeGetTime();

// get input/output Tensor count
cuteModel.inputTensorCount();
cuteModel.outputTensorCount();

// get input/output Tensor
cuteModel.inputTensor(index);
cuteModel.outputTensor(index);

// get input/output Tensor input/output data length(array *length* of input/output)
cuteModel.inputTensorLength(index);
cuteModel.outputTensorLength(index);

// Model summaries
cuteModel.summary();
cuteModel.summarizeOptions();

// Etc
size_t CuteModel::elementByteSize(const TfLiteTensor* tensor);
size_t CuteModel::tensorLength(const TfLiteTensor *tensor);
```
