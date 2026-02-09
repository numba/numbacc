@echo on
mkdir build
cd build

cmake -GNinja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DPython_FIND_REGISTRY=NEVER ^
  -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%" ^
  -DCMAKE_PREFIX_PATH="%LIBRARY_PREFIX%" ^
  -DLLVM_USE_INTEL_JITEVENTS=1 ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DLLVM_BUILD_LLVM_DYLIB=OFF ^
  -DLLVM_LINK_LLVM_DYLIB=OFF ^
  -DLLVM_BUILD_TOOLS=OFF ^
  -DMLIR_INCLUDE_TOOLS=ON ^
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON ^
  -DPython3_EXECUTABLE="%PYTHON%" ^
  ..\mlir

if %ERRORLEVEL% neq 0 exit 1

ninja -j%CPU_COUNT%
if %ERRORLEVEL% neq 0 exit 1

ninja install
if not exist "%SP_DIR%" mkdir "%SP_DIR%"
move "%PREFIX%"\Library\python_packages\mlir_core\mlir "%SP_DIR%"\

if exist "%PREFIX%"\Library\python_packages rmdir /s /q "%PREFIX%"\Library\python_packages
if exist "%PREFIX%"\Library\src rmdir /s /q "%PREFIX%"\Library\src
