import ctypes
from mlir import ir, execution_engine as ee
from mlir.passmanager import PassManager

print("MLIR python bindings self test")

mlir_asm = """
module {
  func.func @main() -> i32 attributes {llvm.emit_c_interface} {
    %0 = arith.constant 42 : i32
    return %0 : i32
  }
}
"""

with ir.Context():
    m = ir.Module.parse(mlir_asm)

    # Lower to LLVM dialect
    pm = PassManager.parse("builtin.module(convert-func-to-llvm,convert-arith-to-llvm,reconcile-unrealized-casts)")
    pm.run(m.operation)

    e = ee.ExecutionEngine(m)

    # Create output buffer for the i32 result
    result = ctypes.c_int32()
    result_ptr = ctypes.pointer(result)

    e.invoke("main", result_ptr)
    print(f"Result: {result.value}")  # Should print 42
    assert result.value == 42
