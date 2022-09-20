#ifndef ONNX_MLIR_OPSCHEDULER_HPP
#define ONNX_MLIR_OPSCHEDULER_HPP

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "src/CostModel/Device.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <vector>

using namespace mlir;

namespace onnx_mlir {

constexpr const char *DEVICE_ATTR = "device";
constexpr const char *PRIORITY_ATTR = "priority";

void SetDefaultPlacementAndPriority(mlir::OpBuilder &builder,
    std::vector<Operation *> &ops, const DeviceSetT &device_set);

void HackSchedule(mlir::OpBuilder &builder,
    std::vector<Operation *> &ops, const DeviceSetT &device_set);

void RandomSchedule(mlir::OpBuilder &builder, std::vector<Operation *> &ops,
    const DeviceSetT &device_set);

void SingleDTUSchedule(mlir::OpBuilder &builder, std::vector<Operation *> &ops,
    const DeviceSetT &device_set);

void SingleCPUSchedule(mlir::OpBuilder &builder, std::vector<Operation *> &ops,
    const DeviceSetT &device_set);

void RandomSearchSchedule(mlir::OpBuilder &builder, std::vector<Operation *> &ops,
    const DeviceSetT &device_set, int32_t iter_num);

void GASchedule(mlir::OpBuilder &builder, std::vector<Operation *> &ops,
    const DeviceSetT &device_set, int32_t iter_num);


} // namespace onnx_mlir
#endif /* ONNX_MLIR_OPSCHEDULER_HPP */
