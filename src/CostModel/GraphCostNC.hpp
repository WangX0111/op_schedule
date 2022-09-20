#ifndef ONNX_MLIR_GRAPHCOSTNC_HPP
#define ONNX_MLIR_GRAPHCOSTNC_HPP

#include "src/CostModel/Device.hpp"
#include "src/CostModel/Task.hpp"

#include "mlir/IR/Operation.h"

#include <algorithm>
#include <deque>
#include <set>
#include <unordered_set>
#include <vector>

namespace onnx_mlir {

struct Cost {
  double time_cost;
  double memory_peak;
};

class GraphCostNC {

public:
  explicit GraphCostNC(
      const std::vector<mlir::Operation *> &ops, const DeviceSetT &device_set)
      : ops_(std::move(ops)), device_set_(device_set) {}

  void UpdateOps(const std::vector<mlir::Operation *> &ops) { ops_ = ops; }

  void UpdateDeviceSet(const DeviceSetT &device_set) {
    device_set_ = device_set;
  }

  Cost GetGraphCost();

private:
  void GenDeviceTaskList(
      std::unordered_map<std::string, std::deque<ComputeTask>> &compute_task,
      std::unordered_map<std::string, std::deque<TransferTask>> &transfer_task,
      llvm::DenseMap<mlir::Value, int> &transfer_cnt,
      const std::unordered_map<std::string, Device> &device_table);

  std::vector<mlir::Operation *> ops_;
  DeviceSetT device_set_;
};

} // namespace onnx_mlir
#endif /* ONNX_MLIR_GRAPHCOSTNC_HPP */
