// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;
using Itexoft.Mlx;
using NUnit.Framework;

public static unsafe class TestHelpers
{
    public static void RequireNativeOrIgnore()
    {
        try
        {
            if (!TryGetVersionString(out _))
                Assert.Ignore("MlxNativeAot not available");
        }
        catch (DllNotFoundException)
        {
            Assert.Ignore("MlxNativeAot not available");
        }
        catch (EntryPointNotFoundException)
        {
            Assert.Ignore("MlxNativeAot entrypoint missing");
        }
        catch (BadImageFormatException)
        {
            Assert.Ignore("MlxNativeAot bad image");
        }
    }

    public static bool TryGetVersionString(out string value)
    {
        value = string.Empty;
        try
        {
            var rc = MlxVersion.Version(out var h);

            if (rc != 0)
                return false;
            var ptr = MlxString.Data(h);
            var s = Marshal.PtrToStringUTF8(ptr);
            MlxString.Free(h);

            if (string.IsNullOrWhiteSpace(s))
                return false;
            value = s!;

            return true;
        }
        catch
        {
            return false;
        }
    }

    public static void Ok(int rc, string what)
    {
        if (rc != 0)
            Assert.Fail(what);
    }

    public static MlxStreamHandle CpuStream() => MlxStream.DefaultCpuStreamNew();

    public static int[] ShapeOf(MlxArrayHandle a)
    {
        var ndim = (int)MlxArray.Ndim(a);
        var p = (nint)MlxArray.Shape(a);
        var result = new int[ndim];
        for (var i = 0; i < ndim; i++)
            result[i] = Marshal.ReadInt32(p, sizeof(int) * i);

        return result;
    }

    public static float[] ToFloat32(MlxArrayHandle a)
    {
        var n = (int)MlxArray.Size(a);
        var p = MlxArray.DataFloat32(a);
        var arr = new float[n];
        for (var i = 0; i < n; i++)
            arr[i] = *(p + i);

        return arr;
    }

    public static int[] ToInt32(MlxArrayHandle a)
    {
        var n = (int)MlxArray.Size(a);
        var p = MlxArray.DataInt32(a);
        var arr = new int[n];
        for (var i = 0; i < n; i++)
            arr[i] = *(p + i);

        return arr;
    }

    public static double[] ToFloat64(MlxArrayHandle a)
    {
        var n = (int)MlxArray.Size(a);
        var p = MlxArray.DataFloat64(a);
        var arr = new double[n];
        for (var i = 0; i < n; i++)
            arr[i] = *(p + i);

        return arr;
    }
}