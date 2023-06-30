# CUDAprocedure

### GPU Based Vector  Summation
#### AIM
```
(i) To modify or set the execution configuration of block.x as 1023 & 1024 and compare the elapsed time obtained on Host and GPU.
(ii) To set the number of threads as 256 and obtain the elapsed time on Host and GPU.
```
#### ALGORITHM
```
1. Initialize the device and set the device properties.
2. Allocate memory on the host for input and output arrays.
3. Initialize input arrays with random values on the host.
4. Allocate memory on the device for input and output arrays, and copy input data from host to device.
5. Launch a CUDA kernel to perform vector addition on the device.
6. Copy output data from the device to the host and verify the results against the host's sequential vector addition. Free memory on the host and the device.
```
#### RESULT 
```
The block size 1023 performs better in the GPU with an elapsed time of 0.0197 seconds, and the block size 1024 shows better results in the host with an elapsed time of 0.022 seconds. Using a block size of 256 and two threads simultaneously has provided the best results in the GPU with an elapsed time of 0.021 seconds. Thus, the differences between the execution configurations of GPU based vector summation has been explored successfully.
```
### Matrix Summation using 2D grids and 2D blocks
#### AIM
```
To perform  matrix summation with a 2D grid and 2D blocks and adapting it to integer matrix addition.
```
#### ALGORITHM
```
1. Include the required files and library.
2. Declare a function sumMatrixOnHost , to perform matrix summation on the host side . Declare three matrix A , B , C . Store the resultant matrix in C.
3. Declare a function with __ global __ , which is a CUDA C keyword , to execute the function to perform matrix summation on GPU .
4. Declare Main method/function .
5. In the Main function Set up device and data size of matrix ,Allocate Host Memory and device global memory, Initialize data at host side and then add matrix at host side ,transfer data from host to device.
6. Invoke kernel at host side , check for kernel error and copy kernel result back to host side.
7. Finally Free device global memory, host memory and reset device.
8. Save and Run the Program.
```
#### RESULT 
```
The host took 0.884061 seconds to complete it’s computation, while the GPU outperforms the host and completes the computation in 0.012146 seconds. Therefore, float variables in the GPU will result in the best possible result. Thus, matrix summation using 2D grids and 2D blocks has been performed successfully.
```
### Simple Warp Divergence: Sum reduction
### 8
#### AIM
```
To implement the kernel reduceUnrolling16 and comapare the performance of kernal reduceUnrolling16 with kernal reduceUnrolling8 using proper metrics and events with nvprof.
```
#### ALGORITHM
```
1. Initialize the input size n and allocate host memory (h_idata and h_odata) for input and 
    output data.

2. Initialize the input data on the host by assigning a value of 1 to each element in h_idata.

3. Allocate device memory (d_idata and d_odata) for input and output data on the GPU.

4. Copy the input data from the host to the device using cudaMemcpy.

5. Define the grid and block dimensions for the kernel launch. Each block will contain 256                        threads, and the grid size will be calculated based on the input size n and block size.

6. Start the CPU timer to measure the CPU execution time.

7. Compute the sum of input data on the CPU using a for loop and store the result in         -----sum_cpu.

8. Stop the CPU timer and calculate the elapsed CPU time.

9. Start the GPU timer to measure the GPU execution time.

10. Launch the reduceUnrolling8 kernel on the GPU with the specified grid and block dimensions.

11. Copy the result data from the device to the host using cudaMemcpy.

12. Compute the final sum on the GPU by adding up the elements in h_odata and store the result in sum_gpu.

13. Stop the GPU timer and calculate the elapsed GPU time.

14. Print the results: CPU sum, GPU sum, CPU elapsed time, and GPU elapsed time.

15. Free the allocated host and device memory using free and cudaFree.

16. Return from the main function.
```
#### RESULT 
```
Thus the program has been executed by unrolling by 8 and unrolling by 16. It is observed that Unrolling by 8 has executed with less elapsed time than unrolling by 16 with blocks 16.
```
### 16
#### AIM
```
To implement the kernel reduceUnrolling16 and comapare the performance of kernal reduceUnrolling16 with kernal reduceUnrolling8 using proper metrics and events with nvprof.
```
#### ALGORITHM
```
1. Initialize the input size n and allocate host memory (h_idata and h_odata) for input and 
    output data.

2. Initialize the input data on the host by assigning a value of 1 to each element in h_idata.

3. Allocate device memory (d_idata and d_odata) for input and output data on the GPU.

4. Copy the input data from the host to the device using cudaMemcpy.

5. Define the grid and block dimensions for the kernel launch. Each block will contain 256                        threads, and the grid size will be calculated based on the input size n and block size.

6. Start the CPU timer to measure the CPU execution time.

7. Compute the sum of input data on the CPU using a for loop and store the result in         -----sum_cpu.

8. Stop the CPU timer and calculate the elapsed CPU time.

9. Start the GPU timer to measure the GPU execution time.

10. Launch the reduceUnrolling16 kernel on the GPU with the specified grid and block dimensions.

11. Copy the result data from the device to the host using cudaMemcpy.

12. Compute the final sum on the GPU by adding up the elements in h_odata and store the result in sum_gpu.

13. Stop the GPU timer and calculate the elapsed GPU time.

14. Print the results: CPU sum, GPU sum, CPU elapsed time, and GPU elapsed time.

15. Free the allocated host and device memory using free and cudaFree.

16. Return from the main function..
```
#### RESULT 
```
Thus the program has been executed by unrolling by 8 and unrolling by 16. It is observed that Unrolling by 8 has executed with less elapsed time than unrolling by 16 with blocks 16.
```
### Matrix Addition with Unified Memory
#### AIM
```
To perform Matrix addition with unified memory and check its performance with nvprof.
```
#### ALGORITHM
```
1. Include the required files and library.
2. Introduce a function named "initialData","sumMatrixOnHost","checkResult" to return the initialize the data , perform matrix summation on the host and then check the result.
3. Create a grid 2D block 2D global function to perform matrix on the gpu.
4. Declare the main function. In the main function set up the device & data size of matrix , perform memory allocation on host memory & initialize the data at host side then add matrix at host side for result checks followed by invoking kernel at host side. Then warm-up kernel,check the kernel error, and check device for results.Finally free the device global memory and reset device.
5. Execute the program and run the terminal . Check the performance using nvprof.
```
#### RESULT 
```
The initialization process was completed in 0.418289seconds, and the matrix addition took 0.065890 seconds in the host, and 0.042262 seconds in the GPU and provides better performance among the host and GPU. Thus, matrix addition using CUDA programming with unified memory has been performed successfully.
```
### Matrix Multiplication on Host and GPU 
#### AIM
```
To implement Matrix Multiplication using GPU.
```
#### ALGORITHM
```
1. Allocate memory for matrices h_a , h_b , and h_c on the host. 
2. Initialize matrices h_a and h_b with random values between 0 and 1. 
3. Allocate memory for matrices d_a , d_b , and d_c on the device. 
4. Copy matrices h_a and h_b from the host to the device. 
5. Launch the kernel matrixMulGPU with numBlocks blocks of threadsPerBlock threads. 
6. Measure the time taken by the CPU and GPU implementations using CUDA events. 
7. Print the elapsed time for each implementation. 
8. Free the memory allocated on both the host and the device. 
```
#### RESULT 
```
The implementation of Matrix Multiplication using GPU is done successfully.
```
### CUDA Matrix  Transposition
#### AIM
```
To demonstrate the Matrix transposition on shared memory with grid (1,1) block (16,16).
```
#### ALGORITHM
```
1. The code implements various matrix transposition techniques using shared memory in CUDA. 

2. The different implementations include:
 SetRowReadRow : Transpose matrix using row-major ordering for both read and write operations. 
SetColReadCol : Transpose matrix using column-major ordering for both read and write operations.
 SetColReadCol2 : Transpose matrix using column-major ordering for write operation and row-major ordering for read operation. 
SetRowReadCol : Transpose matrix using row-major ordering for write operation and column-major ordering for read operation. 
SetRowReadColDyn : Transpose matrix using dynamic shared memory and rowmajor ordering for write operation and column-major ordering for read operation. 
SetRowReadColPad : Transpose matrix using row-major ordering for write operation and column-major ordering for read operation, with padding. 
SetRowReadColDynPad : Transpose matrix using dynamic shared memory, rowmajor ordering for write operation, column-major ordering for read operation, with padding. 

3. The code measures the execution time of each implementation using CUDA events. 

4. The results of the matrix transposition are verified by comparing the output with the expected result. 

5. The performance of each implementation is compared based on their execution times.
```
#### RESULT 
```
The Matrix transposition on shared memory with grid (1,1) block (16,16) is demonstrated successfully.
```
