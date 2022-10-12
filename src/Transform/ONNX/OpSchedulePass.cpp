

// #include "src/Pass/Passes.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"

#include "src/CostModel/Device.hpp"
#include "src/CostModel/GraphCostNC.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Transform/ONNX/OpScheduler.hpp"
#include "src/Utils/Utils.hpp"
#include "src/Log/Log.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <vector>

using namespace mlir;

namespace onnx_mlir {

bool IgnoreOp(mlir::Operation *op) {
  if (isa<ONNXReturnOp>(op) || isa<ONNXConstantOp>(op) ||
      isa<func::ReturnOp>(op) || isa<ONNXNoneOp>(op) || isa<ONNXReturnOp>(op) || isa<ONNXSliceOp>(op)) {
    return true;
  }
  return false;
}

class OpSchedulePass
    : public mlir::PassWrapper<OpSchedulePass, OperationPass<ModuleOp>> {
public:
  llvm::StringRef getArgument() const override { return "op-schedule"; }

  llvm::StringRef getDescription() const override {
    return "op schedule for onnx mlr";
  }

  void runOnOperation() override {
    llvm::outs() << "Enter OpSchedulePass\n";

    auto module_op = getOperation();
    auto main_func = module_op.lookupSymbol<mlir::func::FuncOp>("main_graph");

    auto &block = main_func.front();
    auto builder = mlir::OpBuilder(&block, block.end());

    std::vector<Operation *> ops;
    for (auto &op : llvm::make_early_inc_range(
             llvm::make_range(block.begin(), block.end()))) {
      if (IgnoreOp(&op)) {
        // std::cout << op.getName().getStringRef().str() << "\n";
        continue;
      }
      ops.push_back(&op);
    }

    auto device_set = GetDeviceSet();
    SetPlacementAndPriority(builder, ops, device_set);

    GraphCostNC graph_cost_analyzer(ops, device_set);

    auto cost = graph_cost_analyzer.GetGraphCost();

    std::cout << "time_cost:  " << cost.time_cost << "\n";
    std::cout << "memory_peak:  " << cost.memory_peak << "\n";

    std::cout << "Not Support Op Level CostModel: \n";
    for (const auto &str : NotSupportCostOp) {
      std::cout << str << " ";
    }
    std::cout << "\n";
  }

private:
  void SetPlacementAndPriority(mlir::OpBuilder &builder,
      std::vector<Operation *> &ops, const DeviceSetT &device_set) {
    SetDefaultPlacementAndPriority(builder, ops, device_set);
    // Schedule Strategy
    // RandomSchedule(builder, ops, device_set);
    // HackSchedule(builder, ops, device_set);
    // SingleDTUSchedule(builder, ops, device_set);
    // SingleCPUSchedule(builder, ops, device_set);
    // RandomSearchSchedule(builder, ops, device_set, 500);
    // GASchedule(builder, ops, device_set, 500);
    // PSOSchedule(builder, ops, device_set, 500);

    BRKGASchedule(builder, ops, device_set, 500);
  }
};

std::unique_ptr<mlir::Pass> createOpSchedulePass() {
  return std::make_unique<OpSchedulePass>();
}
} // namespace onnx_mlir