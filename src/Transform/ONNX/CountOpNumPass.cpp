

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/TypeID.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "src/CostModel/OpLevelCost.hpp"
#include "src/Utils/Utils.hpp"

#include <memory>
#include <unordered_set>
#include <vector>

using namespace mlir;

bool IgnoreOp(mlir::Operation *op) {
  if (isa<ONNXReturnOp>(op) || isa<ONNXConstantOp>(op) ||
      isa<func::ReturnOp>(op) || isa<ONNXNoneOp>(op)) {
    return true;
  }
  return false;
}

namespace onnx_mlir {
class CountOpNumPass
    : public PassWrapper<CountOpNumPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const override { return "count-op-num"; }

  StringRef getDescription() const override {
    return "Shape inference for frontend dialects.";
  }
  void runOnOperation() override {
    llvm::outs() << "run CountOpNumPass\n";

    auto module_op = getOperation();

    auto main_func = module_op.lookupSymbol<mlir::func::FuncOp>("main_graph");

    auto &block = main_func.front();

    std::vector<Operation *> ops;
    // std::unordered_set<mlir::Value> conv_res;
    llvm::DenseSet<mlir::Value> conv_res;
    for (auto &op : llvm::make_early_inc_range(
             llvm::make_range(block.begin(), block.end()))) {
      if (IgnoreOp(&op)) {
        continue;
      }
      ops.push_back(&op);

      if (isa<ONNXConvOp>(op)) {
        for (const auto res : op.getResults()) {
          conv_res.insert(res);
        }
      }
      if (isa<ONNXReluOp>(op)) {
        for (const auto operand : op.getOperands()) {
          if (conv_res.contains(operand)) {
            llvm::outs() << ArrayToStr<llvm::ArrayRef<int64_t>>(
                                operand.getType()
                                    .cast<mlir::RankedTensorType>()
                                    .getShape())
                         << "\n";
            llvm::outs() << "conv res contains relu operand\n";
          }
        }
      }
    }


    llvm::outs() << "ops number: " << ops.size() << "\n";
  }
};

std::unique_ptr<mlir::Pass> createCountOpNumPass() {
  return std::make_unique<CountOpNumPass>();
};

} // namespace onnx_mlir
