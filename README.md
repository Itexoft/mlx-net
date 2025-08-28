
# **MLX .NET** [![CI](https://github.com/Itexoft/mlx-net/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Itexoft/mlx-net/actions/workflows/ci.yml) [![NuGet](https://img.shields.io/nuget/v/Itexoft.MLX.svg)](https://www.nuget.org/packages/Itexoft.MLX)

  

**MLX .NET**  is a .NET API for MLX  – an array framework for machine learning on Apple silicon, brought to you by Apple Machine Learning Research  . MLX .NET expands MLX to the .NET platform, allowing .NET developers to leverage MLX for research and experimentation on Apple silicon devices  . In practice, MLX .NET provides .NET (C# and other .NET language) bindings to the MLX library (via the MLX C API), enabling high-performance machine learning computations with a NumPy-like API on supported hardware.

  

MLX is designed as a flexible and efficient array computation framework with  **unified memory**  and  **multi-device**support (CPU and GPU)  . This means MLX .NET can seamlessly execute operations on different devices without explicit data transfers, taking full advantage of Apple’s unified memory architecture. MLX also supports advanced features like automatic differentiation, vectorization, and dynamic computation graphs – these capabilities are available through the MLX .NET bindings as well, closely mirroring the functionality of MLX’s Python, C++ and Swift APIs  .

  

## **Installation**

  

**NuGet Package (Recommended):**  _MLX .NET will be available as a NuGet package._  You can add it to your .NET project by running the following command in the Package Manager Console or your CLI:

```
dotnet add package Itexoft.MLX
```

  

This will download the MLX .NET library and the necessary native MLX components. The NuGet package includes the native MLX C library (for Apple silicon) built-in, so no separate installation of MLX is required on macOS (arm64). Ensure you are targeting a compatible platform (e.g.  **macOS 14+ on Apple M1/M2**  chips for GPU support). Linux support (CPU and CUDA backends) will be provided in line with the upstream MLX availability.

  

**Building from Source:**  Alternatively, you can build MLX .NET from source. Clone this repository and its submodules, then use the .NET SDK to compile:

```
git clone --recurse-submodules https://github.com/Itexoft/mlx-net.git
cd ./mlx-net
./build-mlx-c.sh
cd ./src/MlxNet
dotnet build -c Release
```
(You will need a C++ compiler and CMake environment if you target non-macOS platforms or if building the native components manually. Refer to the MLX C documentation for platform-specific build requirements  .)

  

## **Usage**

  

MLX .NET aims to  **closely follow the MLX Python and C++ API**, so developers familiar with NumPy/PyTorch or MLX’s other bindings will find the interface intuitive. The core abstraction is the  MlxArray  – a multi-dimensional array/tensor type provided by this library. You can create and manipulate  MlxArray  objects with a variety of operations defined in the API.

  

For example, the snippet below shows basic usage in C#:

```
using Mlx;  // Namespace for MLX .NET (contains MlxArray, MlxOps, etc.)

// Create a 1-D MLX array from .NET data (e.g., an array of floats)
float[] data = { 1.0f, 2.0f, 3.0f };
MlxArray x = MlxArray.FromArray(data);       // wrap .NET array into an MLX array

// Perform an element-wise operation (e.g., add a scalar):
MlxArray y = MlxOps.Add(x, 5.0f);            // y will be [6.0, 7.0, 8.0]

// Create another array (e.g., a 3x3 matrix of ones):
MlxArray ones = MlxArray.Ones(new long[] {3, 3}); 

// Generate a random 3x3 matrix from a normal distribution:
MlxArray rand = MlxRandom.Normal(mean: 0, stdDev: 1, shape: new long[] {3, 3});

// Perform matrix multiplication:
MlxArray product = MlxLinalg.MatMul(ones, rand);

// Apply a transformation (e.g., compute the element-wise sine of the result):
MlxArray sinProduct = MlxOps.Sin(product);
```

In the above example,  MlxArray.FromArray  creates an MLX tensor from a native C# array. Most  **basic arithmetic operations**  are available via the  MlxOps  class (such as  Add,  Multiply,  Sin, etc.), and many  **utility constructors**  are provided in  MlxArray  (such as  Ones,  Zeros, or random initializers in  MlxRandom). Linear algebra routines are under  MlxLinalg, and so on. These operations call into the MLX engine, executing on the optimal device (CPU or GPU) with lazy evaluation – computations are done only when needed, as per MLX’s lazy execution model.

  

**Automatic Differentiation:**  MLX .NET also supports gradients and differentiation. For example, you can use MLX’s transform for automatic differentiation (grad  function) through the .NET API to compute gradients of tensor computations, similar to how you would in PyTorch or JAX. (See MLX documentation on function transformations for more details  .)

  

**Device Management:**  By default, MLX will execute operations on the best available device. You can query and control devices via  MlxDevice. For instance,  MlxDevice.All  gives a list of available devices (e.g., CPU, GPU), and you can specify a target device for computations. MLX .NET inherits the  _unified memory_  design of MLX  , so arrays do not need explicit transfers between CPU/GPU – you can perform operations on any device transparently.

  

## **Examples and Documentation**

  

You can find numerous usage examples of the MLX framework in the official MLX examples repository  (written for Python/Swift). These include training a Transformer language model, text generation with LLaMA, Stable Diffusion image generation, and more. Because MLX .NET closely mirrors the MLX Python API, you can refer to those examples and implement similar solutions in C# or other .NET languages with minimal changes in API usage.

  

For comprehensive information on available operations and functions, refer to the  **MLX C**  documentation and  **MLX Python**  documentation on the MLX website  . MLX .NET exposes equivalent functionality: for instance, if an API is described in MLX C (or Python) docs, the same would typically be accessible in MLX .NET (with analogous naming and .NET naming conventions). This binding is built on the tested and documented MLX C layer, ensuring that behavior and performance align with the upstream MLX library.

  

> **Note:**  MLX is an evolving project by Apple; MLX .NET will strive to stay up-to-date with upstream MLX releases. The goal is to provide .NET developers first-class access to MLX’s capabilities without having to leave the .NET ecosystem. If you encounter any missing features or issues, please check the GitHub repository for updates or to open an issue.

  

## **License**

  

MLX .NET is open-source software, provided under the  **Mozilla Public License 2.0**  (with “Incompatible With Secondary Licenses” clause, as noted in the LICENSE file). This means you are free to use, modify, and distribute this library in source or binary form, provided you adhere to the MPL-2.0 terms.
