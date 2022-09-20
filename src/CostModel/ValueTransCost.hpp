#ifndef ONNX_MLIR_VALUETRANSCOST_HPP
#define ONNX_MLIR_VALUETRANSCOST_HPP

#include "mlir/IR/Value.h"
#include "src/CostModel/Device.hpp"

namespace onnx_mlir {

double GetValueTransCost(mlir::Value value, const Device &source_device,
    const Device &target_device);

} // namespace onnx_mlir

#endif /* ONNX_MLIR_VALUETRANSCOST_HPP */
