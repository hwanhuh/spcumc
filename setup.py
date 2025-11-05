from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='spcumc',
    version='0.1.0',
    author='Hwan Heo',
    author_email='gjghks950@naver.com',
    description='Sparse Marching Cubes with CUDA and PyTorch bindings',
    ext_modules=[
        CUDAExtension(
            name='spcumc._backend',
            sources=[
                'src/api.cpp',
                'src/spcumc.cu',
                'src/spdmc.cu',
                'src/decimation.cu',
            ],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2', '--extended-lambda']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=['spcumc'],
    py_modules=[],
)
