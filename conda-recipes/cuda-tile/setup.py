from setuptools import setup

setup(
    name='cuda_tile_mlir',
    version='0.0.1',
    packages=[
        'cuda_tile',
        'cuda_tile._mlir',
        'cuda_tile._mlir.dialects',
        'cuda_tile._mlir.extras',
        'cuda_tile._mlir._mlir_libs',
    ],
    package_dir={'': 'build/python_packages'},
    package_data={
        'cuda_tile._mlir._mlir_libs': ['*.so', '*.so.*', 'include/**/*'],
    },
    include_package_data=True,
)