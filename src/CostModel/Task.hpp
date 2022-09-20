#ifndef ONNX_MLIR_TASK_HPP
#define ONNX_MLIR_TASK_HPP

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "src/CostModel/Device.hpp"
#include "src/CostModel/OpLevelCost.hpp"
#include "src/CostModel/ValueTransCost.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Utils/Utils.hpp"
#include "llvm/ADT/DenseSet.h"
#include <cassert>
#include <ostream>
#include <string>
#include <unordered_map>

namespace onnx_mlir {
enum class TASK_STATE { WAIT = 0, READY, DONE };

class BaseTask {

public:
  BaseTask() : state_(TASK_STATE::WAIT) {}
  BaseTask(TASK_STATE state) : state_(state) {}

  virtual bool IsReady(
      const std::unordered_map<std::string, llvm::DenseSet<mlir::Value>>
          &device_memory) = 0;
  virtual void Finish(double &cur_time,
      std::unordered_map<std::string, llvm::DenseSet<mlir::Value>>
          &device_memory) = 0;

  double RemainTime() const { return remain_time_; }

  void SetState(TASK_STATE new_state) { state_ = new_state; }

  void Continue(double past_time) {
    assert(past_time <= remain_time_);
    remain_time_ -= past_time;
  }

protected:
  double remain_time_;
  TASK_STATE state_;

  friend bool operator<(const BaseTask &lhs, const BaseTask &rhs) {
    return lhs.remain_time_ < rhs.remain_time_;
  }
};

class ComputeTask : public BaseTask {

public:
  ComputeTask() : BaseTask() {}

  ComputeTask(mlir::Operation *op);
  mlir::Operation *GetComputeOp() const { return op_; }

  bool IsReady(
      const std::unordered_map<std::string, llvm::DenseSet<mlir::Value>>
          &device_memory) override;

  void Finish(double &cur_time,
      std::unordered_map<std::string, llvm::DenseSet<mlir::Value>>
          &device_memory) override;

private:
  mlir::Operation *op_;

  friend std::ostream &operator<<(
      std::ostream &os, const ComputeTask &compute_task) {
    std::string op_type = compute_task.op_->getName().getStringRef().str();
    auto node_name_attr = compute_task.op_->getAttr("onnx_node_name");
    std::string node_name;
    if (node_name_attr) {
      node_name = node_name_attr.cast<mlir::StringAttr>().str();
    }
    std::string device_id = compute_task.op_->getAttr("device").cast<mlir::StringAttr>().str();
    os << "ComputeTask(" << op_type << " " << node_name << ") "
       << compute_task.remain_time_ << " " << device_id;
    return os;
  }
};

class TransferTask : public BaseTask {

public:
  TransferTask() : BaseTask() {}

  TransferTask(mlir::Value value, const onnx_mlir::Device &target_device,
      const std::unordered_map<std::string, Device> &device_table);

  mlir::Value GetTransferValue() { return value_; }

  std::string GetTargetDeviceId() const { return target_device_.id(); }

  std::string GetProducedDeviceId() const;

  bool IsReady(
      const std::unordered_map<std::string, llvm::DenseSet<mlir::Value>>
          &device_memory) override;

  void Finish(double &cur_time,
      std::unordered_map<std::string, llvm::DenseSet<mlir::Value>>
          &device_memory) override;

private:
  mlir::Value value_;
  Device target_device_;

  friend bool operator==(const TransferTask &lhs, const TransferTask &rhs) {
    return lhs.value_ == rhs.value_ && lhs.target_device_ == rhs.target_device_;
  }

  friend bool operator!=(const TransferTask &lhs, const TransferTask &rhs) {
    return !(lhs == rhs);
  }

  friend std::ostream &operator<<(
      std::ostream &os, const TransferTask &transfer_task) {
    auto produce_op = transfer_task.value_.getDefiningOp();
    std::string op_type = produce_op->getName().getStringRef().str();
    auto node_name_attr = produce_op->getAttr("onnx_node_name");
    std::string node_name;
    if (node_name_attr) {
      node_name = node_name_attr.cast<mlir::StringAttr>().str();
    }
    os << "TransferTask(" << op_type << " " << node_name << ") "
       << transfer_task.remain_time_;
    return os;
  }
};

} // namespace onnx_mlir

#endif /* ONNX_MLIR_TASK_HPP */
