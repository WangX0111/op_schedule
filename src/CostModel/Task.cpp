#include "src/CostModel/Task.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"

namespace onnx_mlir {

ComputeTask::ComputeTask(mlir::Operation *op)
    : BaseTask(TASK_STATE::WAIT), op_(op) {
  assert(op != nullptr);
  
  remain_time_ = onnx_mlir::GetOpCost(op).execution_time;
}

bool ComputeTask::IsReady(
    const std::unordered_map<std::string, llvm::DenseSet<mlir::Value>>
        &device_memory) {

  auto device_id = op_->getAttr("device").cast<StringAttr>().data();
  auto memory_values = device_memory.at(device_id);

  if (state_ == TASK_STATE::DONE) {
    return false;
  } else if (state_ == TASK_STATE::READY) {
    return true;
  }

  for (const auto &operand : op_->getOperands()) {

    // Ignore ConstOp and NoneOp
    auto operand_op = operand.getDefiningOp();
    if (!operand_op || isa<ONNXConstantOp>(operand_op) ||
        isa<ONNXNoneOp>(operand_op)) {
      continue;
    }

    if (!memory_values.contains(operand)) {
      return false;
    }
  }

  state_ = TASK_STATE::READY;
  return true;
}

void ComputeTask::Finish(double &cur_time,
    std::unordered_map<std::string, llvm::DenseSet<mlir::Value>>
        &device_memory) {

  // update current time
  cur_time += remain_time_;

  // update memory
  auto device_id = op_->getAttr("device").cast<StringAttr>().data();
  auto &memory_values = device_memory[device_id];
  for (const auto &result : op_->getOpResults()) {
    memory_values.insert(result);
  }
}

TransferTask::TransferTask(mlir::Value value,
    const onnx_mlir::Device &target_device,
    const std::unordered_map<std::string, Device> &device_table)
    : BaseTask(TASK_STATE::WAIT), value_(value), target_device_(target_device) {

  auto produce_op = value.getDefiningOp();
  assert(produce_op);

  auto produce_device_id =
      produce_op->getAttr("device").cast<StringAttr>().data();

  Device produced_device = device_table.at(produce_device_id);
  remain_time_ = GetValueTransCost(value_, produced_device, target_device);
}

std::string TransferTask::GetProducedDeviceId() const {
  auto produce_op = value_.getDefiningOp();
  assert(produce_op);
  auto produce_device_id =
      produce_op->getAttr("device").cast<mlir::StringAttr>().data();
  return produce_device_id;
}

bool TransferTask::IsReady(
    const std::unordered_map<std::string, llvm::DenseSet<mlir::Value>>
        &device_memory) {
  if (state_ == TASK_STATE::DONE) {
    return false;
  } else if (state_ == TASK_STATE::READY) {
    return true;
  }

  auto produce_op = value_.getDefiningOp();
  assert(produce_op);
  auto produce_device_id =
      produce_op->getAttr("device").cast<StringAttr>().data();

  auto produce_device_memory = device_memory.at(produce_device_id);

  if (produce_device_memory.contains(value_)) {
    state_ = TASK_STATE::READY;
    return true;
  }

  return false;
}

void TransferTask::Finish(double &cur_time,
    std::unordered_map<std::string, llvm::DenseSet<mlir::Value>>
        &device_memory) {
  // update current time
  cur_time += remain_time_;

  // update target device memory
  auto target_device_id = target_device_.id();
  auto &target_device_memory = device_memory[target_device_id];
  target_device_memory.insert(value_);
}

} // namespace onnx_mlir
