#ifndef ONNX_MLIR_OPLEVELCOST_H
#define ONNX_MLIR_OPLEVELCOST_H

#include "src/CostModel/Device.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Utils/Utils.hpp"

#include "mlir/IR/Operation.h"
#include "llvm/Support/Casting.h"

namespace onnx_mlir {


struct OpCost{

  double execution_time; // latency

  double compute_time;

  double memory_time;
};

OpCost GetOpCost(mlir::Operation *op);

} // namespace onnx_mlir

#endif /* ONNX_MLIR_OPLEVELCOST_H */
