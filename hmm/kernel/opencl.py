import pyopencl
import numpy
import string




class PyOpenCL_Instance:
    context = None
    queue   = None
    kernel  = None
    num_groups = 8
    num_units  = 256
    N = 0
    M = 0

cl = PyOpenCL_Instance()
mf = pyopencl.mem_flags

_FORWARD_SOURCE = 'lib/opencl/forward_blelloch.cl'

def initialize(N, M, num_groups=0, num_units=0, precision="float"):
    platform = pyopencl.get_platforms()[0]
    device   = platform.get_devices(device_type=pyopencl.device_type.GPU)[0]
    cl.context = pyopencl.Context([device])
    file     = open(_FORWARD_SOURCE, 'r')
    source   = string.Template(
        "".join(file.readlines())).substitute(N=N, M=M, precision=precision)
    cl.kernel  = pyopencl.Program(cl.context, source).build()
    cl.queue   = pyopencl.CommandQueue(cl.context)
    cl.N = N
    cl.M = M 

    return cl.context, cl.queue


def forward(
        A, B, pi, ob, T,
        alpha      = None,
        matrices   = None,
        scratch    = None,
        num_groups = 0,
        num_units  = 0 ):
    if cl.context == None:
        raise ValueError

    # prepare variables 
    N = cl.N
    if alpha == None:
        alpha = pyopencl.Buffer(cl.context, mf.READ_WRITE, T*N*N*4)
    if num_groups == 0:
        num_groups = cl.num_groups
    if num_units == 0:
        num_units = cl.num_units
    if scratch == None:
        scratch  = pyopencl.LocalMemory(4*N*N*2*num_units)
    
    def padding_time(time, num):
        return ((time-1)/(2*num)+1)*2*num

    padded = padding_time(T, num_units)
    if matrices == None:
        matrices = pyopencl.Buffer(cl.context, mf.READ_WRITE, padded*N*N*4)

    cl.kernel.forward_build_matrices(cl.queue, (num_groups*num_units,), (num_units,),
        matrices, alpha, A, B, pi, ob, numpy.uint64(T))
    if padded - T > 0:
        cl.kernel.append_identity(cl.queue, (padded-T,), None,
            matrices, numpy.uint64(T), numpy.uint64(padded - T))

    event = pyopencl.enqueue_barrier(cl.queue)
    event.wait()

    tmp_num_units = num_units

    stack = [ ]
    current = matrices      
    S = padded
    while S > 1:
        new_S = S / (2*num_units)
        new_num_units = num_units
        while new_S < new_num_units:
            new_num_units = new_num_units / 2
        new_padded = padding_time(new_S, new_num_units)
        if new_S == 1:
            new_padded = 1        

        reduced  = pyopencl.Buffer(cl.context, mf.READ_WRITE, new_padded*4*N*N)
        cl.kernel.forward_reduce(cl.queue, (num_units*num_groups,), (num_units,), 
            reduced, current, scratch, numpy.uint64(S))
        if new_padded - new_S > 0:
            cl.kernel.append_identity(cl.queue, (num_units*num_groups,), (num_units,),
                reduced, numpy.uint64(new_S), numpy.uint64(new_padded-new_S))
        stack.append( (current, S) )
        current = reduced
        S = new_padded
        num_units = new_num_units
    
    event = pyopencl.enqueue_barrier(cl.queue)
    event.wait()

    num_units = tmp_num_units

    reduced, rS = stack.pop()
    while stack:
        extended, S = stack.pop()
        cl.kernel.forward_collect(cl.queue, (num_units*num_groups,), (num_units,),
            extended, reduced, numpy.uint64(S))
        reduced = extended
        rS = S 

    event = pyopencl.enqueue_barrier(cl.queue)
    event.wait()

    cl.kernel.forward_build_alpha(cl.queue, (num_groups*num_units,), (num_units,),
        alpha, reduced, numpy.uint64(T))
    
    event = pyopencl.enqueue_barrier(cl.queue)
    event.wait()

    return alpha, matrices, scratch


def forward_no_scaling_naive(ctx, A, B, pi, ob):
    """Compute P(ob|A,B,pi) and all forward coefficients. No scaling done.

    This function is for developing the algorithm in general. Do not use this
    in real applications since there is no scaling done. For observation
    sequences larger than 1500 you will get underflow issues.

    Parameters
    ----------
    ctx : hmm.kernel.opencl.Context
          context object created by this module, needed for caching buffers
          handling device information and so on ...
    A : numpy.array of np.float32 and shape (N,N)
        transition matrix of the hidden states
    B : numpy.array of np.float32 and shape (N,M)
        symbol probability matrix for each hidden state
    pi : numpy.array of np.float32 and shape (N)
         initial distribution
    ob : numpy.array of np.int16 and shape (T)
         observation sequence of integer between 0 and M, used as indices in B

    Returns
    -------
    alpha : pyopencl.Buffer
            alpha[t,i] is the ith forward coefficient of time t. These can be
            used in many different algorithms related to HMMs.

    Notes
    -----
    The idea of this algorithm is to write alpha in terms of matrix
    multiplications as shown in paper [1]. Since matrix multiplication is an
    associative binary operator, one can easily apply well known parallelism
    algorithms [2] for cumulative sums. That way one gets alpha_t for each
    t logarithmic in time. See [3] for CUDA implementation details.

    See Also
    --------
    hmm.kernel.python.forward_no_scaling : simple implementation

    .. [1] "Algorithms for a parallel implementation of Hidden Markov Models 
       with a small state space", Jesper Nielsen, Andreas Sand, 2011.
       Bioinformatics Research Centre Aarhus University Aarhus, Denmark. 
       http://www.hicomb.org/papers/HICOMB2011-06.pdf
    .. [2] "Prefix Sums and Their Applications". Guy E. Blelloch. In John H. 
       Reif (Ed.), Synthesis of Parallel Algorithms, Morgan Kaufmann, 1990.
    .. [3] "Parallel Prefix Sum (Scan) with CUDA",
       http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf
    """
    T,N = len(ob), len(A)
    mf = pyopencl.mem_flags

    if (ctx.WORK_GROUP_SIZE == 1):
        raise ValueError

    A_, B_, pi_, ob_ = ctx.create_buffer(A, B, pi, ob)

    ctx.kernel.forward_build_matrices (
            ctx.queue,
            (ctx.NUM_UNITS,), (ctx.WORK_GROUP_SIZE,),
            ctx.matrices, ctx.alpha, A_, B_, pi_, ob_, numpy.uint64(T))

    T = T - 1 # ctx.matrices does not contain alpha_0!

    # forward upwind method
    last_results = ctx.matrices
    stack = [ (last_results, T) ]
    while T > 1:
        grouped_results = pyopencl.Buffer(ctx.context, 
            mf.READ_WRITE, (T/ctx.WORK_GROUP_SIZE+1)*N*N*4)

        ctx.kernel.forward_reduce (
                ctx.queue,
                (ctx.NUM_UNITS,), (ctx.WORK_GROUP_SIZE,),
                grouped_results, last_results, ctx.scratch, numpy.uint64(T))

        T = T / ctx.WORK_GROUP_SIZE
        stack.append( (grouped_results, T) )
        last_results = grouped_results

    # forward rewind method
    grouped_results = last_results
    while stack:
        last_results, T = stack.pop()
        ctx.kernel.forward_rewind(
                ctx.queue,
                (ctx.NUM_UNITS,), (ctx.WORK_GROUP_SIZE,),
                last_results, grouped_results, numpy.uint64(T))
        grouped_results = last_results

    T = T + 1
    ctx.kernel.forward_multiply_with_alpha_0(
                ctx.queue,
                (ctx.NUM_UNITS,), (ctx.WORK_GROUP_SIZE,),
                ctx.alpha, last_results, numpy.uint64(T))

    event = pyopencl.enqueue_barrier(ctx.queue)
    event.wait()

    return ctx.alpha



class Context:
    NUM_GROUPS      = 52
    WORK_GROUP_SIZE = 256
    NUM_UNITS = 52*256
    _FORWARD_SOURCE  = 'lib/opencl/forward_naive.cl'

    COMPILED_N = 0
    COMPILED_M = 0
    CACHED_T   = 0

    # forward buffer
    last_T = 0
    A_buf = None
    B_buf = None
    pi_buf = None
    ob_buf = None
    scaling = None

    def __init__(self, N, M, T=0):
        platform = pyopencl.get_platforms()[0]
        device = platform.get_devices(device_type=pyopencl.device_type.GPU)[0]
        self.context = pyopencl.Context([device])
        self.kernel = self._compile_kernel(N, M)
        self.queue = pyopencl.CommandQueue(self.context)
        self.CACHED_T = T
        if T > 0:
            self.create_buffer_forward()

    def _compile_kernel(self, N, M, DEBUG=1):
        f = open(self._FORWARD_SOURCE, 'r')
        source = string.Template("".join(f.readlines())).substitute(
            N=N, M=M, precision="float")
        kernel = pyopencl.Program(self.context, source).build()
        self.COMPILED_N = N
        self.COMPILED_M = M
        return kernel

    #@profile
    def create_buffer(self, A, B, pi, ob):
        mf = pyopencl.mem_flags
        N, M, T = len(pi), len(B[0]), len(ob)

        if N != self.COMPILED_N or M != self.COMPILED_M:
            self._compile_kernel(N, M)

        if T != self.CACHED_T:
            self.CACHED_T = T
            self.create_buffer_forward()

            A = numpy.array(A, numpy.float32)
            B = numpy.array(B, numpy.float32)
            pi = numpy.array(pi, numpy.float32)
            ob = numpy.array(ob, numpy.int16)

            self.A_buf = pyopencl.Buffer(self.context, 
                mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
            self.B_buf = pyopencl.Buffer(self.context, 
                mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
            self.pi_buf = pyopencl.Buffer(self.context, 
                mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pi)
            self.ob_buf = pyopencl.Buffer(self.context, 
                mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ob)

        
        return self.A_buf, self.B_buf, self.pi_buf, self.ob_buf

    def create_buffer_forward(self):
        mf = pyopencl.mem_flags
        N, M, T = self.COMPILED_N, self.COMPILED_M, self.CACHED_T
        self.alpha         = pyopencl.Buffer(self.context, mf.WRITE_ONLY, T*N*4)
        self.matrices      = pyopencl.Buffer(self.context, mf.READ_WRITE, T*N*N*4)
        self.group_results = pyopencl.Buffer(self.context, mf.READ_WRITE, T*N*N*4)
        self.scratch       = pyopencl.LocalMemory(self.WORK_GROUP_SIZE*N*N * 4)
        self.scaling       = pyopencl.Buffer(self.context, mf.WRITE_ONLY, T*4)
        if N != self.COMPILED_N or M != self.COMPILED_M:
            self._compile_kernel(N, M)

    def set_work_group_sizes(self, NUM_GROUPS, WORK_GROUP_SIZE):
        if self.NUM_GROUPS != NUM_GROUPS or \
                self.WORK_GROUP_SIZE != WORK_GROUP_SIZE:
            self.NUM_GROUPS = NUM_GROUPS
            self.WORK_GROUP_SIZE = WORK_GROUP_SIZE
            self.NUM_UNITS = NUM_GROUPS*WORK_GROUP_SIZE
            if self.CACHED_T > 0:
                self.create_buffer_forward()