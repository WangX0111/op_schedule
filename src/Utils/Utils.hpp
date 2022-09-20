#ifndef ONNX_MLIR_UTILS_HPP
#define ONNX_MLIR_UTILS_HPP

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include <functional>
#include <sstream>

using namespace mlir;

namespace onnx_mlir {

#define SET_CONTAINS(set, key) (set.find(key) != set.cend())

template <typename SourceOp>
bool isa(mlir::Operation *op) {
  return op ? (llvm::dyn_cast<SourceOp>(op) != nullptr) : false;
}

template<typename ArrayT>
std::string ArrayToStr(ArrayT arr){
  std::stringstream ss;
  ss << '[';
  for(auto dim: arr){
    ss << dim << ",";
  }
  ss << ']';
  return ss.str();
}


} // namespace onnx_mlir

#endif /* ONNX_MLIR_UTILS_HPP */
