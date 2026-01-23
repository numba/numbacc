import os.path
from dataclasses import dataclass
from subprocess import check_call

import cuda.tile as ct
import torch
from cuda.tile._cext import TileDispatcher
from cuda.tile._compile import compile_cubin, get_sm_arch
from cuda.tile._compiler_options import CompilerOptions

mlir_kernel_path = "./out.mlir"
assert os.path.exists(mlir_kernel_path)

tilebc_file = "example.tilebc"
kernel_name = "spy_tile_example$exported$export_foo"
# Compile everything
# cubin_path = "./example.cubin"
CUDA_TILE_TRANSLATE_PATH = "../third-party/cuda-tile/build/bin/cuda-tile-translate"
sm_arch = get_sm_arch()
check_call([CUDA_TILE_TRANSLATE_PATH, mlir_kernel_path, "--bytecode-version=13.1", "--mlir-to-cudatilebc", "--no-implicit-module", "-o", tilebc_file])
# check_call(["/usr/local/cuda/bin/tileiras", "--gpu-name", "sm_120", tilebc_file, "-o", "example.cubin"])
cubin_path = compile_cubin(tilebc_file, CompilerOptions(), sm_arch, None)


@dataclass
class HackCompileCallback:
    # From: https://github.com/NVIDIA/cutile-python/blob/361048e152636435ec2a660e650481acbd001745/test/test_bytecode.py#L103-L113
    cubin_path: str
    func_name: str

    def __call__(self, args, ctx):
        return self.cubin_path, self.func_name

# Direct instantiation bypassing @kernel decorator
compile_callback = HackCompileCallback(str(cubin_path), kernel_name)
kernel = TileDispatcher((False,), compile_callback)
#                        ^ arg_constant_flags

x_tensor = torch.arange(128, dtype=torch.float64, device="cuda")
print(x_tensor)
# Launch directly
ct.launch(torch.cuda.current_stream(), (1,), kernel, (x_tensor,))
