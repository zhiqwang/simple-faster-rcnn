// One-stop header.
#include <torch/script.h>

// header for torchvision
#include <torchvision/nms.h>
#include <torchvision/ROIAlign.h>
// #include <torchvision/ROIPool.h>
// #include <torchvision/empty_tensor_op.h>
#include <torchvision/vision.h>

// headers for opencv
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/opencv.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define kIMAGE_SIZE 320
#define kCHANNELS 3

static auto registry =
  torch::RegisterOperators()
    .op("torchvision::nms", &nms)
    .op("torchvision::roi_align(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio, bool aligned) -> Tensor",
      &roi_align);
//     .op("torchvision::roi_pool", &roi_pool)
//     .op("torchvision::_new_empty_tensor_op", &new_empty_tensor);

int main() {
  torch::jit::getProfilingMode() = false;
  torch::jit::getExecutorMode() = false;
  torch::jit::setGraphExecutorOptimize(false);
  torch::jit::script::Module module = torch::jit::load("/mnt/simple-faster-rcnn/checkpoints/faster_fpn_pinochle/export/model.pt");

  torch::DeviceType device_type;
  device_type = torch::kCPU;
  torch::Device device(device_type, 0);

  module.to(device);

  torch::Tensor input_tensor = torch::ones({3, 224, 224});
  input_tensor = input_tensor.to(device);

  c10::List<torch::Tensor> images = c10::List<torch::Tensor>({input_tensor});

  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(images);

  torch::jit::IValue output = module.forward(inputs);

  std::cout << "output: " << output << std::endl;

  // auto out1 = output.toTuple();
  // auto dets = out1->elements().at(1).toGenericList();
  // auto det0 = dets.get(0).toGenericDict();
  // at::Tensor labels = det0.at("labels").toTensor();
  // at::Tensor boxes = det0.at("boxes").toTensor();
  // at::Tensor scores = det0.at("scores").toTensor();

}