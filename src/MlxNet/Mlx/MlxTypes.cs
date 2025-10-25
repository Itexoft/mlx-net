// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

[StructLayout(LayoutKind.Sequential)]
public struct MlxArrayHandle
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxVectorArrayHandle
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxVectorVectorArrayHandle
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxVectorIntHandle
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxVectorStringHandle
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxMapStringToArrayHandle
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxMapStringToArrayIteratorHandle
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxMapStringToStringHandle
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxMapStringToStringIteratorHandle
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxStringHandle
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxDeviceHandle
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxStreamHandle
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxClosureHandle
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxClosureKwargsHandle
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxClosureValueAndGradHandle
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxClosureCustomHandle
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxClosureCustomJvp
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxClosureCustomVmap
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxFastCudaKernel
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxFastCudaKernelConfig
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxFastMetalKernel
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxFastMetalKernelConfig
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxDistributedGroupHandle
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxFunctionExporter
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxImportedFunction
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxIoReader
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxIoWriter
{
    public nint ctx;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxOptionalInt
{
    public int value;
    public byte has_value;
}

[StructLayout(LayoutKind.Sequential)]
public struct MlxOptionalFloat
{
    public float value;
    public byte has_value;
}

[StructLayout(LayoutKind.Sequential)]
public struct Complex64
{
    public float r;
    public float i;
}

[StructLayout(LayoutKind.Sequential)]
public struct Complex128
{
    public double r;
    public double i;
}

[StructLayout(LayoutKind.Sequential)]
public unsafe struct MlxMetalDeviceInfoT
{
    public fixed byte architecture[256];
    public nuint max_buffer_length;
    public nuint max_recommended_working_set_size;
    public nuint memory_size;
}
