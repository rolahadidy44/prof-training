# Parallel Programming Project Report Guide (Sobel Edge Detection)

## 1) Problem Statement
We implement Sobel edge detection on a grayscale image. For each pixel (excluding borders), Sobel computes horizontal and vertical gradients using 3x3 convolution kernels (Gx, Gy) and then computes gradient magnitude:

- `sx = sum(image * Gx)`
- `sy = sum(image * Gy)`
- `output = sqrt(sx^2 + sy^2)`

## 2) Why Parallelize?
Sobel is *data-parallel*: each output pixel depends only on a small 3x3 neighborhood in the input, and each pixel can be computed independently of others (no dependencies between output pixels). That makes it a good target for multicore acceleration.

## 3) Parallel Approach (OpenMP)
We parallelize the outer loops over pixels `(i, j)` using:

- `#pragma omp parallel for collapse(2) schedule(static)`

Each thread computes different pixels. Variables `sx` and `sy` are local to each iteration, so there are no data races. The input image is read-only and shared.

## 4) Correctness Check
We compute:
- Sequential output
- OpenMP output

Then we compare the two matrices element-by-element. If any mismatch occurs, we print the first mismatch location and values.

## 5) Results
Include screenshots of:
- Running sequential vs parallel timings
- Correctness PASS
- (Optional) Speedup for multiple thread counts (1,2,4,8)

## 6) Conclusion
Discuss:
- Observed speedup and why
- Overheads (thread startup, memory bandwidth)
- Why larger images show clearer speedup
