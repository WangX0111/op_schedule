# MLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir

source .env

echo $MLIR_DIR
echo $pythonLocation

cd build
if [[ -z "$pythonLocation" ]]; then
  cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DMLIR_DIR=${MLIR_DIR} \
        ..
else
  cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DPython3_ROOT_DIR=$pythonLocation \
        -DMLIR_DIR=${MLIR_DIR} \
        ..
fi

cmake --build . -j 50

# Run lit tests:
# export LIT_OPTS=-v
# cmake --build . --target check-onnx-lit