import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(module.split('.')[-1], src) for src in sources]
    )
    print([os.path.join(*module.split('.'), src) for src in sources])
    return cuda_ext

if __name__ == '__main__':
    setup(
        name='ballquery',
        packages=find_packages(),
        ext_modules=[
            CUDAExtension('ball_query_cuda',[
                'ball_query_src/api.cpp',
                'ball_query_src/ball_query.cpp',
                'ball_query_src/ball_query_cuda.cu',  
            ])
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )