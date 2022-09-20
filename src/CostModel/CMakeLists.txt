add_onnx_mlir_library(OMCost
  Device.cpp
  OpLevelCost.cpp
  ValueTransConst.cpp
  GraphCostNC.cpp
  Task.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
)