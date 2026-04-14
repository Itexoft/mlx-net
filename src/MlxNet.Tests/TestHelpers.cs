// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Itexoft.Mlx;
using NUnit.Framework;

public static unsafe class TestHelpers
{
    public delegate void WithShapeAction(int* shape, nuint length);

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static sbyte* MemLabel(void* desc)
    {
        var ms = (MemStream*)desc;

        if (ms->Label == 0)
            ms->Label = Marshal.StringToHGlobalAnsi("<mem>");

        return (sbyte*)ms->Label;
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static byte MemIsOpen(void* desc) => (byte)(desc != null ? 1 : 0);

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static byte MemGood(void* desc)
    {
        if (desc == null)
            return 0;

        var m = (MemStream*)desc;

        return (byte)(m->Err == 0 ? 1 : 0);
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static nuint MemTell(void* desc) => ((MemStream*)desc)->Pos;

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static void MemSeek(void* desc, long off, int whence)
    {
        var m = (MemStream*)desc;
        var size = (long)m->Size;
        var cur = (long)m->Pos;

        var np = whence switch
        {
            0 => off,
            1 => cur + off,
            2 => size + off,
            _ => cur,
        };

        if (np < 0 || np > size)
            m->Err = 1;
        else
            m->Pos = (nuint)np;
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static void MemRead(void* desc, sbyte* data, nuint n)
    {
        var m = (MemStream*)desc;

        if (n == 0)
            return;

        if (n > m->Size || m->Pos > m->Size - n)
        {
            m->Err = 1;

            return;
        }

        for (nuint i = 0; i < n; i++)
            data[i] = (sbyte)m->Data[m->Pos + i];

        m->Pos += n;
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static void MemReadAtOffset(void* desc, sbyte* data, nuint n, nuint off)
    {
        var m = (MemStream*)desc;

        if (n == 0)
            return;

        if (off > m->Size || n > m->Size - off)
        {
            m->Err = 1;

            return;
        }

        for (nuint i = 0; i < n; i++)
            data[i] = (sbyte)m->Data[off + i];
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static void MemWrite(void* desc, sbyte* data, nuint n)
    {
        var m = (MemStream*)desc;

        if (n == 0)
            return;

        if (n > m->Size || m->Pos > m->Size - n)
        {
            m->Err = 1;

            return;
        }

        for (nuint i = 0; i < n; i++)
            m->Data[m->Pos + i] = (byte)data[i];

        m->Pos += n;
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static void MemFree(void* desc)
    {
        var m = (MemStream*)desc;

        if (m->FreeData != 0 && m->Data != null)
        {
            NativeMemory.Free(m->Data);
            m->Data = null;
        }

        if (m->Label != 0)
        {
            Marshal.FreeHGlobal(m->Label);
            m->Label = 0;
        }
    }

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

            value = s;

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

    public static MlxStreamHandle NewCpuStream() => MlxStream.DefaultCpuStreamNew();

    public static void WithStream(Action<MlxStreamHandle> action)
    {
        var stream = NewCpuStream();

        try
        {
            action(stream);
        }
        finally
        {
            ReleaseStream(stream);
        }
    }

    public static T WithStream<T>(Func<MlxStreamHandle, T> action)
    {
        var stream = NewCpuStream();

        try
        {
            return action(stream);
        }
        finally
        {
            ReleaseStream(stream);
        }
    }

    public static void ReleaseStream(MlxStreamHandle stream)
    {
        if (stream.ctx != 0)
        {
            MlxStream.Synchronize(stream);
            MlxStream.Free(stream);
        }
    }

    public static int* AllocShape(ReadOnlySpan<int> dims)
    {
        if (dims.Length == 0)
            return null;

        var bytes = sizeof(int) * dims.Length;
        var ptr = (int*)Marshal.AllocHGlobal(bytes);

        for (var i = 0; i < dims.Length; i++)
            ptr[i] = dims[i];

        return ptr;
    }

    public static void FreeShape(int* shape)
    {
        if (shape != null)
            Marshal.FreeHGlobal((nint)shape);
    }

    public static void Copy(ReadOnlySpan<float> source, float* destination)
    {
        for (var i = 0; i < source.Length; i++)
            destination[i] = source[i];
    }

    public static void Copy(ReadOnlySpan<double> source, double* destination)
    {
        for (var i = 0; i < source.Length; i++)
            destination[i] = source[i];
    }

    public static void CopyDimensions(ReadOnlySpan<int> dims, int* destination)
    {
        for (var i = 0; i < dims.Length; i++)
            destination[i] = dims[i];
    }

    public static void WithShape(ReadOnlySpan<int> dims, WithShapeAction action)
    {
        var shape = AllocShape(dims);

        try
        {
            action(shape, (nuint)dims.Length);
        }
        finally
        {
            FreeShape(shape);
        }
    }

    public static int[] ShapeOf(MlxArrayHandle a)
    {
        var ndim = (int)MlxArray.Ndim(a);
        var p = (nint)MlxArray.Shape(a);
        var result = new int[ndim];

        for (var i = 0; i < ndim; i++)
            result[i] = Marshal.ReadInt32(p, sizeof(int) * i);

        return result;
    }

    public static float[] ToFloat32(MlxArrayHandle a) => ReadStrided(a, MlxArray.DataFloat32(a));

    public static int[] ToInt32(MlxArrayHandle a) => ReadStrided(a, MlxArray.DataInt32(a));

    public static double[] ToFloat64(MlxArrayHandle a) => ReadStrided(a, MlxArray.DataFloat64(a));

    private static T[] ReadStrided<T>(MlxArrayHandle handle, T* data) where T : unmanaged
    {
        var rank = (int)MlxArray.Ndim(handle);
        var count = (int)MlxArray.Size(handle);
        var shape = MlxArray.Shape(handle);
        var strides = MlxArray.Strides(handle);
        var values = new T[count];

        if (count == 0)
            return values;

        if (rank == 0)
        {
            values[0] = *data;

            return values;
        }

        var coordinates = stackalloc int[rank];

        for (var linearIndex = 0; linearIndex < count; linearIndex++)
        {
            var remainder = linearIndex;

            for (var axis = rank - 1; axis >= 0; axis--)
            {
                var axisLength = shape[axis];
                coordinates[axis] = remainder % axisLength;
                remainder /= axisLength;
            }

            nuint offset = 0;

            for (var axis = 0; axis < rank; axis++)
                offset += (nuint)coordinates[axis] * strides[axis];

            values[linearIndex] = data[offset];
        }

        return values;
    }

    public static void EvalArray(MlxArrayHandle handle, string label = "eval")
    {
        if (handle.ctx == 0)
            return;

        Ok(MlxArray.Eval(handle), label);
    }

    public static void EvalArrays(params MlxArrayHandle[] handles)
    {
        foreach (var handle in handles)
            EvalArray(handle);
    }

    private static MlxIoVTable MakeMemVTable()
    {
        MlxIoVTable vt;
        vt.is_open = &MemIsOpen;
        vt.good = &MemGood;
        vt.tell = &MemTell;
        vt.seek = &MemSeek;
        vt.read = &MemRead;
        vt.read_at_offset = &MemReadAtOffset;
        vt.write = &MemWrite;
        vt.label = &MemLabel;
        vt.free = &MemFree;

        return vt;
    }

    public static byte[] SaveToBuffer(MlxMapStringToArrayHandle data, MlxMapStringToStringHandle meta, nuint capacity)
    {
        var buf = NativeMemory.Alloc(capacity);
        MemStream mem;
        mem.Data = (byte*)buf;
        mem.Pos = 0;
        mem.Size = capacity;
        mem.Err = 0;
        mem.FreeData = 0;
        mem.Label = 0;

        var vt = MakeMemVTable();
        var writer = MlxIoTypes.IoWriterNew(&mem, vt);
        Ok(MlxIo.SaveSafetensorsWriter(writer, data, meta), "save_mem");

        var len = checked((int)mem.Pos);
        var result = new byte[len];
        Marshal.Copy((nint)mem.Data, result, 0, len);

        MlxIoTypes.IoWriterFree(writer);
        NativeMemory.Free(buf);

        return result;
    }

    public static Dictionary<string, float[]> SnapshotTensors(MlxMapStringToArrayHandle data)
    {
        var dict = new Dictionary<string, float[]>();
        var it = MlxMap.StringToArrayIteratorNew(data);

        while (MlxMap.StringToArrayIteratorNext(out var key, out var arr, it) == 0)
        {
            var name = Marshal.PtrToStringUTF8(key)!;
            Ok(MlxArray.Eval(arr), "eval");
            dict[name] = ToFloat32(arr);
        }

        MlxMap.StringToArrayIteratorFree(it);

        return dict;
    }

    public static void LoadAndCompare(byte[] buffer, MlxStreamHandle stream, Dictionary<string, float[]> originals)
    {
        var unmanaged = NativeMemory.Alloc((nuint)buffer.Length);
        Marshal.Copy(buffer, 0, (nint)unmanaged, buffer.Length);

        MemStream mem;
        mem.Data = (byte*)unmanaged;
        mem.Pos = 0;
        mem.Size = (nuint)buffer.Length;
        mem.Err = 0;
        mem.FreeData = 1;
        mem.Label = 0;

        var vt = MakeMemVTable();
        var reader = MlxIoTypes.IoReaderNew(&mem, vt);
        Ok(MlxIo.LoadSafetensorsReader(out var data, out var meta, reader, stream), "load_mem");
        MlxIoTypes.IoReaderFree(reader);

        var loaded = SnapshotTensors(data);

        foreach (var kv in originals)
        {
            var vals = loaded[kv.Key];
            var expv = kv.Value;
            Assert.That(vals.Length, Is.EqualTo(expv.Length));

            for (var i = 0; i < vals.Length; i++)
                Assert.That(vals[i], Is.EqualTo(expv[i]).Within(1e-6));
        }

        MlxMap.StringToArrayFree(data);
        MlxMap.StringToStringFree(meta);
    }

    internal struct MemStream
    {
        public byte* Data;
        public nuint Pos;
        public nuint Size;
        public byte Err;
        public byte FreeData;
        public nint Label;
    }
}
