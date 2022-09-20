#include "src/CostModel/OpLevelCost.hpp"
#include "src/Log/Log.hpp"
#include "src/CostModel/Device.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Utils/Utils.hpp"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace onnx_mlir {

const double kToMScale = 1024 * 1024;
const double kMinComputeCost = 1;

OpCost PredictOperationCountBasedCost(double operations, double input_io_mb,
    double output_io_mb, double compute_power, double bandwidth) {
  double total_io_bytes = input_io_mb + output_io_mb;

  OpCost op_cost;
  // ns
  op_cost.compute_time = operations / compute_power;
  // ns
  op_cost.memory_time = total_io_bytes / bandwidth;
  // ns
  op_cost.execution_time = std::max(op_cost.compute_time, op_cost.memory_time);
  op_cost.execution_time = std::max(op_cost.execution_time, kMinComputeCost);
  return op_cost;
}

int64_t GetInputBytes(mlir::Operation *op) {
  int64_t input_bits = 0;
  for (const auto &operand : op->getOperands()) {
    auto operand_op = operand.getDefiningOp();
    if (!operand_op || isa<ONNXConstantOp>(op) || isa<ONNXNoneOp>(op)) {
      continue;
    }
    auto operand_type = operand.getType();
    if (!operand_type.isa<mlir::RankedTensorType>()) {
      continue;
    }
    auto ranked_opr_type = operand_type.cast<mlir::RankedTensorType>();
    input_bits += ranked_opr_type.getNumElements() *
                  ranked_opr_type.getElementTypeBitWidth();
  }
  return input_bits / 8;
}

int64_t GetResultBytes(mlir::Operation *op) {
  int64_t output_bits = 0;
  for (const auto &result : op->getResults()) {
    auto result_type = result.getType();
    if (!result_type.isa<mlir::RankedTensorType>()) {
      continue;
    }
    auto ranked_result_type = result_type.cast<mlir::RankedTensorType>();
    output_bits += ranked_result_type.getNumElements() *
                   ranked_result_type.getElementTypeBitWidth();
  }

  return output_bits / 8;
}

OpCost GetONNXDefaultOpCost(mlir::Operation *op, const Device &device) {

  NotSupportCostOp.insert(op->getName().getStringRef().str());

  int64_t input_bytes = GetInputBytes(op);
  int64_t output_bytes = GetResultBytes(op);

  int64_t bpe = op->getResult(0)
                    .getType()
                    .cast<mlir::RankedTensorType>()
                    .getElementTypeBitWidth() /
                8;

  // MFLOPS
  double operation_mflops = (double)output_bytes / bpe / kToMScale;
  // MB
  double input_io_mb = input_bytes / kToMScale;
  double output_io_mb = output_bytes / kToMScale;

  // device info
  auto device_info = GetDeviceInfo(device);

  auto op_cost = PredictOperationCountBasedCost(operation_mflops, input_io_mb,
      output_io_mb, device_info.compute_power_1d, device_info.bandwidth);
  return op_cost;
}

OpCost GetONNXConv2DOpCost(mlir::Operation *conv_opr, const Device &device) {

  auto conv_op = mlir::dyn_cast<ONNXConvOp>(conv_opr);

  // N * Hi * Wi * Hk * Wk * Ci * Co mac
  // 1 mac = 2 operarion
  // op count = N * Ho * Wo * Hk * Wk * Ci * Co * 2
  auto kernel = conv_op.W();
  auto output = conv_op.getResult();

  auto kernel_type = kernel.getType().cast<mlir::RankedTensorType>();
  auto output_type = output.getType().cast<mlir::RankedTensorType>();

  // [Co, Ci, Hk, Wk]
  auto kernel_shape = kernel_type.getShape();

  int64_t op_count = output_type.getNumElements() * kernel_shape[0] *
                     kernel_shape[2] * kernel_shape[3] * 2;

  // MFLOPS
  double operation_mflops = (double)op_count / kToMScale;
  // MB
  double input_io_mb = GetInputBytes(conv_opr) / kToMScale;
  double output_io_mb = GetResultBytes(conv_opr) / kToMScale;

  auto device_info = GetDeviceInfo(device);

  auto op_cost = PredictOperationCountBasedCost(operation_mflops, input_io_mb,
      output_io_mb, device_info.compute_power_2d, device_info.bandwidth);
  return op_cost;
}

OpCost GetONNXElementBinaryOpCost(mlir::Operation *op, const Device &device) {

  auto result_type =
      op->getOpResult(0).getType().cast<mlir::RankedTensorType>();

  int64_t op_count = result_type.getNumElements();

  // MFLOPS
  double operation_mflops = (double)op_count / kToMScale;
  // MB
  double input_io_mb = GetInputBytes(op) / kToMScale;
  double output_io_mb = GetResultBytes(op) / kToMScale;

  auto device_info = GetDeviceInfo(device);

  auto op_cost = PredictOperationCountBasedCost(operation_mflops, input_io_mb,
      output_io_mb, device_info.compute_power_1d, device_info.bandwidth);

  return op_cost;
}

OpCost GetONNXGemmOpCost(mlir::Operation *op, const Device &device) {

  // operations = m * k * n * 2(dot) + m * n(add)
  auto gemm_op = mlir::dyn_cast<ONNXGemmOp>(op);

  // [m, k]
  auto A = gemm_op.A();
  // [m, n]
  auto result = gemm_op.getResult();

  auto A_type = A.getType().cast<mlir::RankedTensorType>();
  auto A_shape = A_type.getShape();

  auto result_type = result.getType().cast<mlir::RankedTensorType>();

  // m * n * k + m * n
  int64_t op_count =
      result_type.getNumElements() * A_shape[1] + result_type.getNumElements();

  // MFLOPS
  double operation_mflops = (double)op_count / kToMScale;
  // MB
  double input_io_mb = GetInputBytes(op) / kToMScale;
  double output_io_mb = GetResultBytes(op) / kToMScale;

  auto device_info = GetDeviceInfo(device);

  auto op_cost = PredictOperationCountBasedCost(operation_mflops, input_io_mb,
      output_io_mb, device_info.compute_power_2d, device_info.bandwidth);
  return op_cost;
}

OpCost GetONNXMatmulOpCost(mlir::Operation *op, const Device &device) {
  auto matmul_op = mlir::dyn_cast<ONNXMatMulOp>(op);

  // [m, k]
  auto A = matmul_op.A();
  // [m, n]
  auto result = matmul_op.getResult();

  auto A_type = A.getType().cast<mlir::RankedTensorType>();
  auto A_shape = A_type.getShape();

  auto result_type = result.getType().cast<mlir::RankedTensorType>();

  // m * n * k
  int64_t op_count = result_type.getNumElements() * A_shape[1];

  // MFLOPS
  double operation_mflops = (double)op_count / kToMScale;
  // MB
  double input_io_mb = GetInputBytes(op) / kToMScale;
  double output_io_mb = GetResultBytes(op) / kToMScale;

  auto device_info = GetDeviceInfo(device);

  auto op_cost = PredictOperationCountBasedCost(operation_mflops, input_io_mb,
      output_io_mb, device_info.compute_power_2d, device_info.bandwidth);
  return op_cost;
}

OpCost GetONNXReduceOpCost(mlir::Operation *op, const Device &device) {
  // op_count = input element numbers

  auto input = op->getOperand(0);

  auto input_type = input.getType().cast<mlir::RankedTensorType>();

  int64_t op_count = input_type.getNumElements();

  // MFLOPS
  double operation_mflops = (double)op_count / kToMScale;
  // MB
  double input_io_mb = GetInputBytes(op) / kToMScale;
  double output_io_mb = GetResultBytes(op) / kToMScale;

  auto device_info = GetDeviceInfo(device);

  auto op_cost = PredictOperationCountBasedCost(operation_mflops, input_io_mb,
      output_io_mb, device_info.compute_power_1d, device_info.bandwidth);
  return op_cost;
}

OpCost GetONNXIdentityOpCost(mlir::Operation *op, const Device &device) {

  OpCost op_cost;

  op_cost.compute_time = kMinComputeCost;
  op_cost.memory_time = 0;
  op_cost.execution_time = std::max(op_cost.compute_time, op_cost.memory_time);

  return op_cost;
}

OpCost GetONNXPoolOpCost(mlir::Operation *op, const Device &device) {
  // Pooling
  // op_count = N * Ho * Wo * Co * Hk * Wk

  // Hk * Wk
  auto kernel_shape_attr = op->getAttr("kernel_shape");
  assert(kernel_shape_attr);
  auto kernel_shape = kernel_shape_attr.cast<mlir::ArrayAttr>().getValue();
  int64_t Hk_Wk = 1;
  for (mlir::Attribute dim : kernel_shape) {
    Hk_Wk *= dim.cast<mlir::IntegerAttr>().getInt();
  }

  auto result = op->getResult(0);
  auto result_type = result.getType().cast<mlir::RankedTensorType>();

  // N * Ho * Wo * Co * Hk * Wk
  int64_t op_count = result_type.getNumElements() * Hk_Wk;

  // MFLOPS
  double operation_mflops = (double)op_count / kToMScale;
  // MB
  double input_io_mb = GetInputBytes(op) / kToMScale;
  double output_io_mb = GetResultBytes(op) / kToMScale;

  auto device_info = GetDeviceInfo(device);

  auto op_cost = PredictOperationCountBasedCost(operation_mflops, input_io_mb,
      output_io_mb, device_info.compute_power_1d, device_info.bandwidth);
  return op_cost;
}

OpCost GetONNXTranscendentalOpCost(mlir::Operation* op, const Device& device){
  double transcendental_cost = 0;
  if (device.type() == DeviceType::CPU){
    transcendental_cost = 1;
  }else if (device.type() == DeviceType::DTU){
    transcendental_cost = 7;
  }

  auto result_type = op->getResult(0).getType();
  int64_t op_count = 1;
  if (result_type.isa<mlir::RankedTensorType>()){
    auto ranked_result_type = result_type.cast<mlir::RankedTensorType>();
    op_count = ranked_result_type.getNumElements();
  }

  op_count *= transcendental_cost;

  // MFLOPS
  double operation_mflops = (double)op_count / kToMScale;
  // MB
  double input_io_mb = GetInputBytes(op) / kToMScale;
  double output_io_mb = GetResultBytes(op) / kToMScale;

  auto device_info = GetDeviceInfo(device);

  auto op_cost = PredictOperationCountBasedCost(operation_mflops, input_io_mb,
      output_io_mb, device_info.compute_power_1d, device_info.bandwidth);
  return op_cost;

}

OpCost GetOpCost(mlir::Operation *op) {

  const auto &device_table = GetDeviceTable();
  std::string device_id = op->getAttr("device").cast<StringAttr>().data();
  auto device = device_table.at(device_id);

  OpCost op_cost;

#define CASE_OP(cost_func, ...)                                                \
  .Case<__VA_ARGS__>(                                                          \
      [&](mlir::Operation *op) { op_cost = cost_func(op, device); })

  // clang-format off
  llvm::TypeSwitch<mlir::Operation *>(op) \
      CASE_OP(GetONNXConv2DOpCost, ONNXConvOp) \
      CASE_OP(GetONNXElementBinaryOpCost, \
      ONNXAddOp, ONNXSubOp, ONNXMulOp, ONNXMulOp, ONNXDivOp, ONNXAndOp, ONNXOrOp, ONNXNegOp, ONNXPowOp) \
      CASE_OP(GetONNXGemmOpCost, ONNXGemmOp) \
      CASE_OP(GetONNXMatmulOpCost, ONNXMatMulOp) \
      CASE_OP(GetONNXReduceOpCost, ONNXReduceSumOp, ONNXReduceMeanOp, ONNXReduceMaxOp, ONNXReduceMinOp, ONNXReduceSumV11Op) \
      CASE_OP(GetONNXIdentityOpCost, ONNXFlattenOp, ONNXReshapeOp) \
      CASE_OP(GetONNXPoolOpCost, ONNXMaxPoolSingleOutOp, ONNXMaxPoolOp, ONNXAveragePoolOp) \
      CASE_OP(GetONNXTranscendentalOpCost, ONNXReluOp)
      .Default([&](mlir::Operation* op){
        op_cost = GetONNXDefaultOpCost(op, device);
      });
  // clang-format on

#undef CASE_OP

  return op_cost;
}

} // namespace onnx_mlir