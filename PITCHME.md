# GPU Programming using Numba

---

### How does GPU manage to accelerate compute intensive tasks?

---

@ul
- Uses Parallel programming strategy
- Breaks the tasks into several smaller sub tasks
- Many versions of the tasks operating different data
- Works on the tasks simultaneously (parallely) across different *threads*.
@ulend

---

### How does CPU approach this same task?

@ul
- Uses single thread. Single stream of instructions.
- Tries to accelerate that single stream of instruction.
- Sequential Processing!
@ulend

---

## What's are tasks?

@ul
- Computational tasks
    - Matrix Multiplications
    - Vector Addition
    - Fast Fourier Transforms
    - Signal Processing techniques
    - Deep Learning Workloads
@ulend

---

## Breaking a task into sub-tasks

@ul
- Crutial to attain maximum performance
- Depends from task to task
- Some tasks are easier and straight-forward than others
- Lets see an example
@ulend

---

## Vector Addition


- Vectors are columns of numbers

`$$\vec{a} = \begin{bmatrix}1 \\2\\3\\4\\\vdots\\n\end{bmatrix}$$`

@ul
- This is a $n\times1$ vectors
- Its a vector in $\mathbb{R}^n$ space

@ulend

+++

Lets take 2 vectors $\vec{a}, \vec{b}$ both in $\mathbb{R}^n$ (n-dimensional space).

$
\vec{a} = \begin{bmatrix}a_1 \\a_2\\\vdots\\a_n\end{bmatrix}
$

$
\vec{b} = \begin{bmatrix}b_1 \\b_2\\\vdots\\b_n\end{bmatrix}
$

Whats $\vec{a} + \vec{b}$ ? (Vector addition)

+++

$
\vec{a} + \vec{b} = \begin{bmatrix}a_1 + b_1\\a_2+b_2\\a_3+b_3\\\vdots\\a_n+b_n\end{bmatrix}
$

- Element wise addition

+++ 

# How to split it into sub tasks?
> In other words parallelize it?

---

Here are the steps (simple algorithms):

@ul
- Identify independent instructions (operations)
- Identify their input of these indepedent operations
- Finalze your fundamental unit that will have several versions running parallely.
@ulend

---

## Revisiting Vector Addition

$
\vec{a} = \begin{bmatrix}a_1 \\a_2\\\vdots\\a_n\end{bmatrix}
$

$
\vec{b} = \begin{bmatrix}b_1 \\b_2\\\vdots\\b_n\end{bmatrix}
$


$
\vec{a} + \vec{b} = \begin{bmatrix}a_1 + b_1\\a_2+b_2\\\vdots\\a_n+b_n\end{bmatrix}
$

@ul
- Whats the fundamental operation performed?
- Whats the input for this operation to be performed?
@ulend

---

Fundamental Operation: Addition
Input: One Elements from 2 vectors `a[i], b[i]`

---

Same operation is performed on different data items. (here a[i], b[i])

> **SIMD** Processing - **S**ingle **I**nstruction **M**ultiple **D**ata Processing

Our Approach to program this addition in GPU:

@ul
- Get the vectors from CPU to GPU
- Replicate `addition` on different compute units in GPUs
- Give appropriate inputs to these units so that they perform useful operation.
- Each unit individually performs the operation on their inputs.
- Aggregate the each units output and send it to CPU for further processing.
@ulend

---
## Terminologies

@ul
- **Device**: GPU (Device memory: GPU Memory)
- **Host**: CPU (Host Memory: CPU Memory)
- **Kernel**: The function that runs in GPU
    - Whats our kernel in vector addition?
- **Threads**: The computational units in GPUs. Runs a version of the kernel.
- **Blocks**: Collections of a set of threads
- **Grid**: Collection of set of blocks
@ulend

---

## Lets Code Vector Scaling in GPU using Numba!

---

```python
import numba
from numba import cuda
```

---

```python
import numpy as np

data = np.ones(256*4096) #1041644

threadsperblock = 256

# Calculating number of blocks per grid
blockspergrid = (data.size+(threadsperblock-1)) //threadsperblock
```

---

```python
from numba import cuda

@cuda.jit
def my_kernel(io_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < io_array.size:  # Check array boundaries
        io_array[pos] *= 2 # do the computation  
```

---
# Kernel Calling

```python
my_kernel[blockspergrid, threadsperblock](data)
```

This line will execute the scaling of the vector in GPU and store the result of the scaled  version in `data` array.

---

Average time taken in GPU: 15.3ms $\pm$ 826 $\mu s$

Average time taken in CPU: 2.37ms $\pm$ 171 $\mu s$
---

# Whats this `threadsperblock`, `blockspergrid` business?

---
@ul
- For effective parallelization of higher dimensional data structures, loopy data structures:  
    - Nvidia follows an hierarchy
    - threads, blocks, grids we saw remember?
@ulend
---

## Hierarchy

- **Threads**: The computational units in GPUs. Runs a version of the kernel.
- **Blocks**: Collections of a set of threads
- **Grid**: Collection of set of blocks

---

![1D_blocks](https://devblogs.nvidia.com/wp-content/uploads/2017/01/cuda_indexing.png)

---
```python
from numba import cuda
import numpy as np

data = np.ones(256*4096) #1041644

threadsperblock = 256

# Calculating number of blocks per grid
blockspergrid = (data.size+(threadsperblock-1)) //threadsperblock

@cuda.jit
def my_kernel(io_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < io_array.size:  # Check array boundaries
        io_array[pos] *= 2 # do the computation  
```
---

### Finding the global index of the thread seems difficult?

```python
# Thread id in a 1D block
tx = cuda.threadIdx.x
# Block id in a 1D grid
ty = cuda.blockIdx.x
# Block width, i.e. number of threads per block
bw = cuda.blockDim.x
# Compute flattened index inside the array
pos = tx + ty * bw

pos = cuda.grid(1)    
```
---
**numba.cuda.grid(ndim)** - Return the absolute position of the current thread in the entire grid of blocks.
---
# Lets do Matrix Multiplication in GPU!

---

![matrix_mul](https://www.mathsisfun.com/algebra/images/matrix-multiply-a.svg)

---

## Remember the guidelines:

- Identify independent instructions (operations)
- Identify their input of these indepedent operations
- Finalize your fundamental unit that will have several versions running parallely.

---
![matrix_mult](https://nyu-cds.github.io/python-numba/fig/05-matmul.png)
---
## What's the dimension of the block here?

> Is it 1D as we saw in scalar multiplication?

---

![image_matmul](https://s3.amazonaws.com/i.seelio.com/6f/fd/6ffd44cf043d8c0e80e4652da28bffb6ae1e.png)

---
# Coding in Numba

---
### Host Code

```python
# Host code

# Initialize the data arrays
m = 2**11 # 2048
n = 2**11
p = 2**11

A = numpy.full((m, n), 1, numpy.float) # matrix containing all 1's
B = numpy.full((n, p), 1, numpy.float) # matrix containing all 1's
```
---
### Host to device data transfer + Memory allocation in GPU

```python
# Copy the arrays to the device
A_global_mem = cuda.to_device(A)
B_global_mem = cuda.to_device(B)

# Allocate memory on the device for the result
C_global_mem = cuda.device_array((m, p))
```

---

## Kernel

```python
@cuda.jit
def matmul(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp        
```
---

## Defining `threadsperblock`, `blockspergrid`

```python
threadsperblock = (32, 32)
# Dimension of the matrix we defined is 2048x2048
```

- How many blocks per grid should I have?
    - Matrix is 2D

+++

```python
blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)
```

---

## Kernel Call

```python
# Start the kernel 
matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)

# Copy the result back to the host
C = C_global_mem.copy_to_host()

print(C)
```

+++

```
[[2048. 2048. 2048. ... 2048. 2048. 2048.]
 [2048. 2048. 2048. ... 2048. 2048. 2048.]
 [2048. 2048. 2048. ... 2048. 2048. 2048.]
 ...
 [2048. 2048. 2048. ... 2048. 2048. 2048.]
 [2048. 2048. 2048. ... 2048. 2048. 2048.]
 [2048. 2048. 2048. ... 2048. 2048. 2048.]]
```
---
# Lets time it!
---

@ul
- Average CPU Time: 592 ms $\pm$ 58.6 ms
- Average GPU Time: 381 $\mu s$ $\pm$ 76.4$\mu s$
- **Which is $1500$ times faster**
@ulend

---

Why did it the fail during the vector scaling operation?

---

# New Moore's Law

* Computers no longer get faster, just Wider

* Rethink your algorithms to be parallel

* Data-Parallel Computing is the most scalable solution

---

![gpu-applications](https://slideplayer.com/slide/3419442/12/images/2/GPU+Accelerated+Applications.jpg)

---

## Thanks for your patience

This presentation and extensive resources can be found in my github - [tlokeshkumar](github.com/tlokeshkumar). A notebook explaining problems in CPU world, how to effectively parallelize CPU code, vectorize them (using Numba) and other optimizations can also be found there.

Even generic code pipelines (API) related to Deep learning like Image classification, Data loading scripts in TensorFlow is also present.... Do check them out if interested!!!

Feel free to contact me at lokesh.karpagam@gmail.com
