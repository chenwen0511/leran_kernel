from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_ops',
    ext_modules=[
        CUDAExtension('custom_ops', [
            '02_pytorch_extension_vector_add/vector_add_pt.cu',
        ]),
        CUDAExtension('custom_matmul', [
            '05_matmul_wmma/matmul_wmma_pt.cu',
        ]),
        CUDAExtension('toy_flash_attn', [
            '08_toy_flash_attention/toy_flash_attn.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
