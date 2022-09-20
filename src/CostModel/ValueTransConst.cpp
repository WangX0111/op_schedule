#include "mlir/IR/BuiltinTypes.h"
#include "src/CostModel/Device.hpp"
#include "src/CostModel/ValueTransCost.hpp"
#include <cstdint>

namespace onnx_mlir {

const int64_t kToMScale = (1024 * 1024);
const double kMinTransTime = 1.0;

double GetValueTransCost(mlir::Value value, const Device &source_device,
    const Device &target_device) {

  if (source_device.id() == target_device.id()) {
    return 0;
  }
  if (target_device.type() == DeviceType::CPU ||
      source_device.type() == DeviceType::CPU) {
    return 0;
  }

  // DTU -> DTU
  // 64 GB/s = 0.064 MB/ns PCIE
  if (target_device.type() == DeviceType::DTU &&
      source_device.type() == DeviceType::DTU) {
    // 0.064 MB/ns
    double bandwidth = 0.064;
    auto value_type = value.getType();
    if (value_type.isa<mlir::RankedTensorType>()) {
      auto ranked_value_type = value_type.cast<mlir::RankedTensorType>();
      auto bpe = ranked_value_type.getElementTypeBitWidth() / 8;
      auto load = ranked_value_type.getNumElements() * bpe / kToMScale;
      double time_cost = (double)load / bandwidth;
      return time_cost;
    }
  }

  return kMinTransTime;
}

} // namespace onnx_mlir