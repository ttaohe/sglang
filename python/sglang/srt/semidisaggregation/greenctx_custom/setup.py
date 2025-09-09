from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

# 获取 PyTorch 的库路径和包含路径
torch_include = os.path.dirname(torch.__file__)  # PyTorch 安装路径
torch_lib = os.path.join(torch_include, "lib")  # PyTorch 库路径（包含 libc10.so 等）

setup(
    name='greenctx',
    ext_modules=[
        CUDAExtension(
            name='greenctx', 
            sources=['greenctx_stream.cu'],  # CUDA 源文件
            include_dirs=[
                os.path.abspath('.'),  # 当前目录的头文件
                os.path.join(torch_include, "include")  # PyTorch 头文件路径
            ],
            library_dirs=[torch_lib],  # 指定 PyTorch 库路径
            libraries=[
                'cuda', 'cudart',  # CUDA 库
                'c10', 'torch', 'torch_cpu', 'torch_python', 'c10_cuda', 'torch_cuda'  # PyTorch 相关库
            ],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2', '--ptxas-options=-v', '--resource-usage']
            },
            extra_link_args=[
                f'-Wl,-rpath,{torch_lib}'  # 强制运行时库路径（避免 LD_LIBRARY_PATH 未设置）
            ]
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
