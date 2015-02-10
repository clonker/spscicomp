from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import os
from os.path import join as pjoin
import numpy


# adapted from https://github.com/rmcgibbo/npcuda-example
def find_in_path(name, path):
    "Find a file in a search path"
    # adapted from http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                                   'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig


def compiler_for_nvcc(self):
    """
    Inherited from the default compile, adapted for nvcc.
    """

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        # use the cuda for .cu files
        self.set_executable('compiler_so', CUDA['nvcc'])
        # use only a subset of the extra_postargs, which are 1-1 translated
        # from the extra_compile_args in the Extension class
        postargs = extra_postargs['nvcc']
        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customized_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


try:
    CUDA = locate_cuda()
    cuda_ext_modules = [
        Extension('kmeans_c_extension_cuda',
                  sources=['kmeans_c_extension.cpp', 'kmeans_chunk_center_cuda.cu'],
                  library_dirs=[CUDA['lib64']],
                  libraries=['cudart'],
                  runtime_library_dirs=[CUDA['lib64']],
                  extra_compile_args={
                      'nvcc': ['-arch=sm_20', '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'"]},
                  include_dirs=[numpy.get_include(), CUDA['include'], 'src'])
    ]
    cuda_cmdclass = {'build_ext': custom_build_ext}
except:
    cuda_ext_modules = []
    cuda_cmdclass = {}

setup(
    name='spscicomp',
    version='1.0',
    packages=['spscicomp',
              'spscicomp.common',
              'spscicomp.hmm', 'spscicomp.hmm.kernel', 'spscicomp.hmm.lib',
              'spscicomp.kmeans', 'spscicomp.kmeans.extension', 'spscicomp.kmeans.opencl', 'spscicomp.kmeans.cuda',
              'spscicomp.tica', 'spscicomp.tica.extension', 'spscicomp.tica.extension.ticaC'],
    package_dir={'spscicomp': './'},
    package_data={'spscicomp.kmeans.opencl': ['opencl_kmeans.cl']},
    url="https://github.com/florianlitzinger/spscicomp",
    include_dirs=[numpy.get_include()],
    requires=['numpy'],
    ext_modules=[
                    Extension('spscicomp.hmm.lib.c',
                              sources=['hmm/lib/c/extension.c', 'hmm/lib/c/hmm.c', 'hmm/lib/c/hmm32.c']),
                    Extension('spscicomp.kmeans.extension.kmeans_c_extension',
                              sources=['kmeans/extension/kmeans_c_extension.cpp',
                                       'kmeans/extension/kmeans_chunk_center.cpp']),
                    Extension('spscicomp.tica.extension.ticaC.ticaC',
                              sources=['tica/extension/ticaC/Tica_CExtension.cpp']),
                ] + cuda_ext_modules,
    extras_require={
        'OpenCL': ['pyopencl'],
        'Plots': ['matplotlib', 'pylab']
    },
    cmdclass=cuda_cmdclass
)
