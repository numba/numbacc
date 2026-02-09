#!/bin/bash

set -euxo pipefail

if [[ "${target_platform}" == "linux-ppc64le" ]]; then
  export CFLAGS="${CFLAGS//-fno-plt/}"
  export CXXFLAGS="${CXXFLAGS//-fno-plt/}"
fi

export BUILD_NUMPY_INCLUDE_DIRS=$( $PYTHON -c "import numpy; print (numpy.get_include())")
export TARGET_NUMPY_INCLUDE_DIRS=$SP_DIR/numpy/core/include

echo $BUILD_NUMPY_INCLUDE_DIRS
echo $TARGET_NUMPY_INCLUDE_DIRS

if [[ "${CONDA_BUILD_CROSS_COMPILATION:-0}" == "1" ]]; then
  echo "Copying files from $BUILD_NUMPY_INCLUDE_DIRS to $TARGET_NUMPY_INCLUDE_DIRS"
  mkdir -p $TARGET_NUMPY_INCLUDE_DIRS
  cp -r $BUILD_NUMPY_INCLUDE_DIRS/numpy $TARGET_NUMPY_INCLUDE_DIRS
  CMAKE_ARGS="${CMAKE_ARGS} -DPython3_NumPy_INCLUDE_DIR=${TARGET_NUMPY_INCLUDE_DIRS}"

  # NATIVE_LLVM_DIR is used by what we patch in for cross-compilation
  CMAKE_ARGS="${CMAKE_ARGS} -DMLIR_TABLEGEN_EXE=$BUILD_PREFIX/bin/mlir-tblgen -DNATIVE_LLVM_DIR=$BUILD_PREFIX/lib/cmake/llvm"
  NATIVE_FLAGS="-DCMAKE_C_COMPILER=$CC_FOR_BUILD;-DCMAKE_CXX_COMPILER=$CXX_FOR_BUILD"
  NATIVE_FLAGS="${NATIVE_FLAGS};-DCMAKE_C_FLAGS=-O2;-DCMAKE_CXX_FLAGS=-O2"
  NATIVE_FLAGS="${NATIVE_FLAGS};-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,${BUILD_PREFIX}/lib"
  NATIVE_FLAGS="${NATIVE_FLAGS};-DCMAKE_MODULE_LINKER_FLAGS=;-DCMAKE_SHARED_LINKER_FLAGS="
  NATIVE_FLAGS="${NATIVE_FLAGS};-DCMAKE_STATIC_LINKER_FLAGS=;-DCMAKE_PREFIX_PATH=${BUILD_PREFIX}"
  CMAKE_ARGS="${CMAKE_ARGS} -DCROSS_TOOLCHAIN_FLAGS_NATIVE=${NATIVE_FLAGS}"
else
  rm -rf $BUILD_PREFIX/bin/mlir-tblgen
fi

mkdir -p build
cd build

# Build cmake arguments with conditional Intel JIT events support
CMAKE_INTEL_JITEVENTS_ARG=""
if [[ "${target_platform}" == "linux-64" ]]; then
  CMAKE_INTEL_JITEVENTS_ARG="-DLLVM_USE_INTEL_JITEVENTS=ON"
fi

cmake ${CMAKE_ARGS} \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLVM_BUILD_LLVM_DYLIB=OFF \
  -DLLVM_LINK_LLVM_DYLIB=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DLLVM_BUILD_TOOLS=ON \
  ${CMAKE_INTEL_JITEVENTS_ARG} \
  -DPython_EXECUTABLE="$PYTHON" \
  -DPython_INCLUDE_DIR="$PREFIX/include/python$PY_VER" \
  -DPython3_EXECUTABLE="$PYTHON" \
  -GNinja \
  ../mlir


echo "Building TableGen tools..."
ninja mlir-tblgen

# FIX: Generate all required headers before main build
echo "Generating MLIR headers..."
ninja mlir-headers

# Now do the main build
echo "Building MLIR..."
ninja 
ninja install 

cd $PREFIX

mkdir -p $SP_DIR
mv $PREFIX/python_packages/mlir_core/mlir $SP_DIR/

rm -rf $PREFIX/src $PREFIX/python_packages

# more clean up
rm $PREFIX/lib/libMLIR*.a
rm -rf $PREFIX/lib/objects-Release
rm -rf $PREFIX/lib/cmake
rm -rf $PREFIX/include
