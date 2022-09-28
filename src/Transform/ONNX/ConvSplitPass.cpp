
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include "src//Utils/Utils.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace onnx_mlir {

const std::string kSplitFlag = "has_split";

class ConvSplitPattern : public mlir::OpRewritePattern<mlir::ONNXConvOp> {
public:
  using mlir::OpRewritePattern<mlir::ONNXConvOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::ONNXConvOp conv_op, mlir::PatternRewriter &rewriter) const final {

    if (conv_op->hasAttr("has_split")) {
      return failure();
    }

    auto input = conv_op.X();
    auto kernel = conv_op.W();
    auto bias = conv_op.B();

    auto input_type = input.getType().cast<mlir::RankedTensorType>();
    auto kernel_type = kernel.getType().cast<mlir::RankedTensorType>();
    auto kernel_shape = kernel_type.getShape();

    auto conv_result_type =
        conv_op->getResult(0).getType().cast<mlir::RankedTensorType>();
    auto conv_result_shape = conv_result_type.getShape();

    const int64_t kSplitNum = 4;
    auto split_dim = kernel_shape[0];
    auto split_len = split_dim / kSplitNum;
    std::vector<int64_t> split_lens(kSplitNum - 1, split_len);
    split_lens.push_back(split_dim - split_len * (kSplitNum - 1));

    std::vector<mlir::ONNXConvOp> sub_conv_ops;
    std::vector<mlir::Value> sub_conv_res;
    auto sub_bias =
        rewriter.create<mlir::ONNXNoneOp>(conv_op->getLoc()).getResult();
    auto auto_pad_attr = rewriter.getStringAttr(conv_op.auto_pad());
    auto dilations_attr = conv_op.dilations().value_or(nullptr);
    auto group_attr = conv_op.group();
    auto kernel_shape_attr = conv_op.kernel_shape().value_or(nullptr);
    auto pads_attr = conv_op.pads().value_or(nullptr);
    auto strides_attr = conv_op.strides().value_or(nullptr);

    for (int64_t i = 0; i < kSplitNum; ++i) {
      std::vector<int64_t> slice_res_shape{kernel_shape};
      slice_res_shape[0] = split_lens[i];
      auto slice_res_type = mlir::RankedTensorType::get(
          slice_res_shape, kernel_type.getElementType());

      auto starts_op = CreateConstantOp(&rewriter, conv_op->getLoc(),
          rewriter.getI64Type(), {1}, i * split_len);
      auto ends_op = CreateConstantOp(&rewriter, conv_op->getLoc(),
          rewriter.getI64Type(), {1}, i * split_len + split_lens[i]);
      auto axis = CreateConstantOp(
          &rewriter, conv_op->getLoc(), rewriter.getI64Type(), {1}, 0);
      auto steps = CreateConstantOp(&rewriter, conv_op->getLoc(),
          rewriter.getI64Type(), {1}, split_lens[i]);

      auto slice_op = rewriter.create<mlir::ONNXSliceOp>(conv_op->getLoc(),
          slice_res_type, kernel, starts_op->getResult(0),
          ends_op->getOpResult(0), axis->getResult(0), steps->getOpResult(0));

      std::vector<int64_t> sub_conv_res_shape{conv_result_shape};
      sub_conv_res_shape[1] = split_lens[i];
      auto sub_conv_res_type = mlir::RankedTensorType::get(
          sub_conv_res_shape, input_type.getElementType());

      auto sub_kernel = slice_op->getResult(0);
      auto sub_conv_op = rewriter.create<mlir::ONNXConvOp>(conv_op->getLoc(),
          sub_conv_res_type, input, sub_kernel, sub_bias, auto_pad_attr,
          dilations_attr, group_attr, kernel_shape_attr, pads_attr,
          strides_attr);
      sub_conv_op->setAttr(kSplitFlag, rewriter.getBoolAttr(true));
      sub_conv_res.push_back(sub_conv_op->getResult(0));
    }

    auto concat_input = mlir::ValueRange(sub_conv_res);
    auto concat_axis = 1;
    auto concat_op = rewriter.create<mlir::ONNXConcatOp>(
        conv_op->getLoc(), conv_result_type, concat_input, concat_axis);

    // concat
    rewriter.replaceOp(conv_op, concat_op->getResult(0));
    return mlir::success();
  }
};

class ConvSplitPass : public mlir::PassWrapper<ConvSplitPass,
                          mlir::OperationPass<func::FuncOp>> {
public:
  llvm::StringRef getArgument() const override { return "conv-split"; }

  llvm::StringRef getDescription() const override { return "conv split"; }

  void runOnOperation() override {
    std::cout << "enter ConvSplitPass\n";

    mlir::RewritePatternSet patterns(&getContext());
    patterns.insert<ConvSplitPattern>(&getContext());

    (void)mlir::applyPatternsAndFoldGreedily(
        getOperation(), std::move(patterns));
  }
};

std::unique_ptr<mlir::Pass> createConvSplitPass() {
  return std::make_unique<ConvSplitPass>();
}
} // namespace onnx_mlir