from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_ops', # 这是你未来在 Python 里 import 的包名
    ext_modules=[
        CUDAExtension('custom_ops', [
            'vector_add_pt.cu', # 你的源文件
        ]),
        # CUDAExtension('custom_matmul', [
        #     'matmul_pt.cu', # 指向新写的文件
        # ]),
        CUDAExtension('custom_matmul', [
            'matmul_wmma_pt.cu', # 指向新写的文件
        ]),
        CUDAExtension('toy_flash_attn', [
            'toy_flash_attn.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
