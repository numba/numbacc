set -x
cmake -G Ninja -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCUDA_TILE_ENABLE_BINDINGS_PYTHON=ON \
    -DCMAKE_INSTALL_PREFIX=$PREFIX

# Build
cmake --build build --target install

set -e
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
cat record.txt
