from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension

#from torch.utils import cpp_extension

#setup(name='pydijkstra_cpp',
#      ext_modules=[cpp_extension.CppExtension('pydijkstra_cpp', ['pydijkstra.cpp'])],
#      cmdclass={'build_ext': cpp_extension.BuildExtension})

ext_modules = [
    Pybind11Extension(
        "pydijkstra",
        ["pydijkstra.cpp"],
        extra_compile_args = ['-std=c++11', '-O3', '-fopenmp'],
        extra_link_args = ['-lgomp'])]

setup(
    name = 'pydijkstra',
    ext_modules = ext_modules)
