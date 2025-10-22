// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Text;
using NUnit.Framework;
using Itexoft.Mlx;
using MemStream = TestHelpers.MemStream;

[TestFixture]
public unsafe class ExampleTests
{
    private struct BogusPayload
    {
        public MlxArrayHandle value;
        public fixed byte error[256];
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static sbyte* MemLabel(void* desc)
    {
        var ms = (MemStream*)desc;
        if (ms->label == 0)
            ms->label = Marshal.StringToHGlobalAnsi("<mem>");

        return (sbyte*)ms->label;
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static byte MemGood(void* desc)
    {
        if (desc == null)
            return 0;
        var m = (MemStream*)desc;

        return (byte)(m->err == 0 ? 1 : 0);
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static void MemSeek(void* desc, long off, int whence)
    {
        var m = (MemStream*)desc;
        var size = (long)m->size;
        var cur = (long)m->pos;
        var np = whence switch
        {
            0 => off,
            1 => cur + off,
            2 => size + off,
            _ => cur
        };
        if (np < 0 || np > size)
            m->err = 1;
        else
            m->pos = (nuint)np;
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static void MemRead(void* desc, sbyte* data, nuint n)
    {
        var m = (MemStream*)desc;

        if (n == 0)
            return;
        if (n > m->size || m->pos > m->size - n)
        {
            m->err = 1;

            return;
        }

        for (nuint i = 0; i < n; i++)
            data[i] = (sbyte)m->data[m->pos + i];
        m->pos += n;
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static void MemReadAtOffset(void* desc, sbyte* data, nuint n, nuint off)
    {
        var m = (MemStream*)desc;

        if (n == 0)
            return;
        if (off > m->size || n > m->size - off)
        {
            m->err = 1;

            return;
        }

        for (nuint i = 0; i < n; i++)
            data[i] = (sbyte)m->data[off + i];
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static void MemWrite(void* desc, sbyte* data, nuint n)
    {
        var m = (MemStream*)desc;

        if (n == 0)
            return;
        if (n > m->size || m->pos > m->size - n)
        {
            m->err = 1;

            return;
        }

        for (nuint i = 0; i < n; i++)
            m->data[m->pos + i] = (byte)data[i];
        m->pos += n;
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static void MemFree(void* desc)
    {
        var m = (MemStream*)desc;
        if (m->free_data != 0 && m->data != null)
            NativeMemory.Free(m->data);
    }


    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static byte MemIsOpen(void* desc) => (byte)(desc != null ? 1 : 0);

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static nuint MemTell(void* desc) => ((MemStream*)desc)->pos;

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static int IncFunPayload(MlxVectorArrayHandle* vres, MlxVectorArrayHandle input, void* payload)
    {
        var s = TestHelpers.CpuStream();
        MlxVector.ArrayGet(out var src, input, 0);
        var value = *(MlxArrayHandle*)payload;
        var rc0 = MlxOps.Add(out var res, src, value, s);

        if (rc0 != 0)
            return rc0;
        var rc1 = MlxVector.ArraySetValue(ref *vres, res);
        MlxArray.Free(src);
        MlxArray.Free(res);

        return rc1;
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static int IncFun(MlxArrayHandle* res, MlxArrayHandle input)
    {
        var s = TestHelpers.CpuStream();
        var one = MlxArray.NewFloat(1f);
        var rc = MlxOps.Add(out var tmp, input, one, s);
        MlxArray.Free(one);

        if (rc != 0)
            return rc;
        *res = tmp;

        return 0;
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static int IncFunValue(MlxVectorArrayHandle* vres, MlxVectorArrayHandle input, void* payload)
    {
        var s = TestHelpers.CpuStream();
        MlxVector.ArrayGet(out var src, input, 0);
        var val = *(MlxArrayHandle*)payload;
        var rc0 = MlxOps.Add(out var res, src, val, s);

        if (rc0 != 0)
            return rc0;
        var rc1 = MlxVector.ArraySetValue(ref *vres, res);
        MlxArray.Free(src);
        MlxArray.Free(res);

        return rc1;
    }

    private static byte HasNan(MlxArrayHandle value, MlxStreamHandle s)
    {
        var rc0 = MlxOps.Isnan(out var tmp, value, s);

        if (rc0 != 0)
            return 0;
        var rc1 = MlxOps.Any(out var any, tmp, false, s);
        MlxArray.Free(tmp);
        if (rc1 != 0)
        {
            MlxArray.Free(any);

            return 0;
        }

        MlxArray.ItemBool(out var flag, any);
        MlxArray.Free(any);

        return flag;
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static int IncFunBogus(MlxVectorArrayHandle* vres, MlxVectorArrayHandle input, void* payloadPtr)
    {
        var payload = (BogusPayload*)payloadPtr;
        var s = TestHelpers.CpuStream();
        if (HasNan(payload->value, s) != 0)
        {
            var msg = "nan detected";
            for (var i = 0; i < msg.Length; i++)
                payload->error[i] = (byte)msg[i];
            payload->error[msg.Length] = 0;

            return 1;
        }

        MlxVector.ArrayGet(out var src, input, 0);
        var rc0 = MlxOps.Add(out var res, src, payload->value, s);
        if (rc0 != 0)
        {
            MlxArray.Free(src);

            return rc0;
        }

        var rc1 = MlxVector.ArraySetValue(ref *vres, res);
        MlxArray.Free(src);
        MlxArray.Free(res);

        return rc1;
    }

    [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
    private static void ErrorHandler(sbyte* msg, void* data) { }

    [Test]
    public void Divide_And_Arrange_Float32()
    {
        TestHelpers.RequireNativeOrIgnore();
        var s = TestHelpers.CpuStream();
        var data = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var shape = new[] { 2, 3 };
        fixed (float* pd = data)
        fixed (int* ps = shape)
        {
            var arr = MlxArray.NewData(pd, ps, shape.Length, MlxDType.MLX_FLOAT32);
            var two = MlxArray.NewInt(2);
            TestHelpers.Ok(MlxOps.Divide(out var divided, arr, two, s), "divide");
            TestHelpers.Ok(MlxArray.Eval(divided), "eval");
            var dv = TestHelpers.ToFloat32(divided);
            Assert.That(dv[0], Is.EqualTo(0.5f).Within(1e-6));
            Assert.That(dv[5], Is.EqualTo(3f).Within(1e-6));
            TestHelpers.Ok(MlxOps.Arange(out var ranged, 0, 3, 0.5, MlxDType.MLX_FLOAT32, s), "arange");
            TestHelpers.Ok(MlxArray.Eval(ranged), "eval");
            var rv = TestHelpers.ToFloat32(ranged);
            Assert.That(rv[0], Is.EqualTo(0f).Within(1e-6));
            Assert.That(rv[5], Is.EqualTo(2.5f).Within(1e-6));
            MlxArray.Free(arr);
            MlxArray.Free(two);
            MlxArray.Free(divided);
            MlxArray.Free(ranged);
        }
    }

    [Test]
    public void Multiply_Divide_And_Arrange_Float64()
    {
        TestHelpers.RequireNativeOrIgnore();
        var s = TestHelpers.CpuStream();
        var data = new[] { 1d, 2d, 3d, 4d, 5d, 6d };
        var shape = new[] { 2, 3 };
        fixed (double* pd = data)
        fixed (int* ps = shape)
        {
            var arr = MlxArray.NewData(pd, ps, shape.Length, MlxDType.MLX_FLOAT64);
            var three = MlxArray.NewFloat64(3d);
            TestHelpers.Ok(MlxOps.Multiply(out var multiplied, arr, three, s), "multiply");
            var two = MlxArray.NewInt(2);
            TestHelpers.Ok(MlxOps.Divide(out var divided, multiplied, two, s), "divide");
            TestHelpers.Ok(MlxArray.Eval(divided), "eval");
            var dv = TestHelpers.ToFloat64(divided);
            Assert.That(dv[0], Is.EqualTo(1.5d).Within(1e-6));
            TestHelpers.Ok(MlxOps.Arange(out var ranged, 0, 3, 0.5, MlxDType.MLX_FLOAT64, s), "arange");
            TestHelpers.Ok(MlxArray.Eval(ranged), "eval");
            var rv = TestHelpers.ToFloat64(ranged);
            Assert.That(rv[0], Is.EqualTo(0d).Within(1e-6));
            Assert.That(rv[5], Is.EqualTo(2.5d).Within(1e-6));
            MlxArray.Free(arr);
            MlxArray.Free(three);
            MlxArray.Free(two);
            MlxArray.Free(multiplied);
            MlxArray.Free(divided);
            MlxArray.Free(ranged);
        }
    }

    [Test]
    public void Closure_Adds_Payload()
    {
        TestHelpers.RequireNativeOrIgnore();
        var x = MlxArray.NewFloat(1f);
        var val = MlxArray.NewFloat(2f);
        var cls = MlxClosure.NewFuncPayload(&IncFunPayload, &val, (delegate* unmanaged[Cdecl]<void*, void>)0);
        var vx = MlxVector.ArrayNewValue(x);
        TestHelpers.Ok(MlxClosure.Apply(out var vy, cls, vx), "closure_apply");
        MlxVector.ArrayGet(out var y, vy, 0);
        TestHelpers.Ok(MlxArray.Eval(y), "eval");
        var arr = TestHelpers.ToFloat32(y);
        Assert.That(arr[0], Is.EqualTo(3f).Within(1e-6));
        MlxArray.Free(x);
        MlxArray.Free(val);
        MlxArray.Free(y);
        MlxVector.ArrayFree(vx);
        MlxVector.ArrayFree(vy);
        MlxClosure.Free(cls);
    }

    [Test]
    public void Jvp_And_ValueAndGrad()
    {
        TestHelpers.RequireNativeOrIgnore();
        var cls = MlxClosure.NewUnary(&IncFun);
        var x = MlxArray.NewFloat(1f);
        var vx = MlxVector.ArrayNewValue(x);
        var one = MlxArray.NewFloat(1f);
        var tang = MlxVector.ArrayNewValue(one);
        TestHelpers.Ok(MlxTransforms.Jvp(out var vout, out var vdout, cls, vx, tang), "jvp");
        MlxVector.ArrayGet(out var out0, vout, 0);
        MlxVector.ArrayGet(out var dout0, vdout, 0);
        TestHelpers.Ok(MlxArray.Eval(out0), "eval");
        TestHelpers.Ok(MlxArray.Eval(dout0), "eval");
        var ov = TestHelpers.ToFloat32(out0);
        var dv = TestHelpers.ToFloat32(dout0);
        Assert.That(ov[0], Is.EqualTo(2f).Within(1e-6));
        Assert.That(dv[0], Is.EqualTo(1f).Within(1e-6));
        var arg = 0;
        TestHelpers.Ok(MlxTransforms.ValueAndGrad(out var vag, cls, &arg, 1), "value_and_grad");
        TestHelpers.Ok(MlxClosure.ValueAndGradApply(out var vout1, out var vdout1, vag, vx), "value_and_grad_apply");
        MlxVector.ArrayGet(out var out1, vout1, 0);
        MlxVector.ArrayGet(out var dout1, vdout1, 0);
        TestHelpers.Ok(MlxArray.Eval(out1), "eval");
        TestHelpers.Ok(MlxArray.Eval(dout1), "eval");
        var ov1 = TestHelpers.ToFloat32(out1);
        var dv1 = TestHelpers.ToFloat32(dout1);
        Assert.That(ov1[0], Is.EqualTo(2f).Within(1e-6));
        Assert.That(dv1[0], Is.EqualTo(1f).Within(1e-6));
        MlxArray.Free(x);
        MlxArray.Free(one);
        MlxArray.Free(out0);
        MlxArray.Free(dout0);
        MlxArray.Free(out1);
        MlxArray.Free(dout1);
        MlxVector.ArrayFree(vx);
        MlxVector.ArrayFree(tang);
        MlxVector.ArrayFree(vout);
        MlxVector.ArrayFree(vdout);
        MlxVector.ArrayFree(vout1);
        MlxVector.ArrayFree(vdout1);
        MlxClosure.ValueAndGradFree(vag);
        MlxClosure.Free(cls);
    }

    [Test]
    public void ValueAndGrad_With_Payload()
    {
        TestHelpers.RequireNativeOrIgnore();
        var x = MlxArray.NewFloat(1f);
        var y = MlxArray.NewFloat(1f);
        var cls = MlxClosure.NewFuncPayload(&IncFunValue, &y, (delegate* unmanaged[Cdecl]<void*, void>)0);
        var arg = 0;
        TestHelpers.Ok(MlxTransforms.ValueAndGrad(out var vag, cls, &arg, 1), "value_and_grad");
        var vx = MlxVector.ArrayNewValue(x);
        TestHelpers.Ok(MlxClosure.ValueAndGradApply(out var vout, out var vdout, vag, vx), "value_and_grad_apply");
        MlxVector.ArrayGet(out var out0, vout, 0);
        MlxVector.ArrayGet(out var dout0, vdout, 0);
        TestHelpers.Ok(MlxArray.Eval(out0), "eval");
        TestHelpers.Ok(MlxArray.Eval(dout0), "eval");
        var ov = TestHelpers.ToFloat32(out0);
        var dv = TestHelpers.ToFloat32(dout0);
        Assert.That(ov[0], Is.EqualTo(2f).Within(1e-6));
        Assert.That(dv[0], Is.EqualTo(1f).Within(1e-6));
        MlxArray.Free(x);
        MlxArray.Free(y);
        MlxArray.Free(out0);
        MlxArray.Free(dout0);
        MlxVector.ArrayFree(vx);
        MlxVector.ArrayFree(vout);
        MlxVector.ArrayFree(vdout);
        MlxClosure.ValueAndGradFree(vag);
        MlxClosure.Free(cls);
    }

    [Test]
    public void ClosurePayloadValidation()
    {
        TestHelpers.RequireNativeOrIgnore();
        var x = MlxArray.NewFloat(1f);
        var vx = MlxVector.ArrayNewValue(x);
        BogusPayload payload = default;
        payload.value = MlxArray.NewFloat(2f);
        var cls = MlxClosure.NewFuncPayload(&IncFunBogus, &payload, (delegate* unmanaged[Cdecl]<void*, void>)0);
        TestHelpers.Ok(MlxClosure.Apply(out var vout, cls, vx), "apply");
        MlxVector.ArrayGet(out var y, vout, 0);
        TestHelpers.Ok(MlxArray.Eval(y), "eval");
        var arr = TestHelpers.ToFloat32(y);
        Assert.That(arr[0], Is.EqualTo(3f).Within(1e-6));
        MlxArray.SetFloat(ref payload.value, float.NaN);
        MlxError.SetErrorHandler(&ErrorHandler, null, null);
        var rc1 = MlxClosure.Apply(out var vout2, cls, vx);
        Assert.That(rc1, Is.Not.EqualTo(0));
        if (rc1 == 0)
            MlxVector.ArrayFree(vout2);
        MlxError.SetErrorHandler(null, null, null);
        var bytes = new byte[256];
        var end = 0;
        for (; end < 256 && payload.error[end] != 0; end++)
            bytes[end] = payload.error[end];
        var msg = Encoding.UTF8.GetString(bytes, 0, end);
        Assert.That(msg, Is.EqualTo("nan detected"));
        MlxArray.Free(x);
        MlxArray.Free(y);
        MlxArray.Free(payload.value);
        MlxVector.ArrayFree(vx);
        MlxVector.ArrayFree(vout);
        MlxClosure.Free(cls);
    }

    [Test]
    public void FunctionExportImport()
    {
        TestHelpers.RequireNativeOrIgnore();
        var cls = MlxClosure.NewUnary(&IncFun);
        var x = MlxArray.NewFloat(1f);
        var args = MlxVector.ArrayNewValue(x);
        var path = Path.GetTempFileName();
        TestHelpers.Ok(MlxExport.Function(path, cls, args, false), "export");
        var xfunc = MlxExport.ImportedFunctionNew(path);
        TestHelpers.Ok(MlxExport.ImportedFunctionApply(out var vres, xfunc, args), "apply");
        MlxVector.ArrayGet(out var y, vres, 0);
        TestHelpers.Ok(MlxArray.Eval(y), "eval");
        var arr = TestHelpers.ToFloat32(y);
        Assert.That(arr[0], Is.EqualTo(2f).Within(1e-6));
        var emptyArgs = MlxVector.ArrayNew();
        var kwargs = MlxMap.StringToArrayNew();
        MlxMap.StringToArrayInsert(kwargs, "x", x);
        TestHelpers.Ok(MlxExport.ImportedFunctionApplyKwargs(out var vres2, xfunc, emptyArgs, kwargs), "apply_kwargs");
        MlxVector.ArrayGet(out var y2, vres2, 0);
        TestHelpers.Ok(MlxArray.Eval(y2), "eval");
        var arr2 = TestHelpers.ToFloat32(y2);
        Assert.That(arr2[0], Is.EqualTo(2f).Within(1e-6));
        MlxVector.ArrayFree(vres);
        MlxVector.ArrayFree(vres2);
        MlxMap.StringToArrayFree(kwargs);
        MlxVector.ArrayFree(emptyArgs);
        MlxArray.Free(x);
        MlxArray.Free(y);
        MlxArray.Free(y2);
        MlxVector.ArrayFree(args);
        MlxClosure.Free(cls);
        MlxExport.ImportedFunctionFree(xfunc);
        File.Delete(path);
    }

    [Test]
    public void SafetensorsMemoryIO()
    {
        TestHelpers.RequireNativeOrIgnore();
        var file = Path.GetTempFileName();
        try
        {
            TestHelpers.WithStream(stream =>
            {
                var seed = MlxMap.StringToArrayNew();
                var metaSeed = MlxMap.StringToStringNew();
                try
                {
                    var data = new[] { 1f, 2f, 3f, 4f };
                    TestHelpers.WithShape(
                        [2, 2],
                        (shapePtr, rank) =>
                        {
                            var scratch = stackalloc float[data.Length];
                            TestHelpers.Copy(data, scratch);
                            var tensor = MlxArray.NewData(scratch, shapePtr, (int)rank, MlxDType.MLX_FLOAT32);
                            MlxMap.StringToArrayInsert(seed, "a", tensor);
                            MlxArray.Free(tensor);
                        });

                    var bytesSeed = TestHelpers.SaveToBuffer(seed, metaSeed, 4096);
                    File.WriteAllBytes(file, bytesSeed);

                    TestHelpers.Ok(MlxIo.LoadSafetensors(out var diskData, out var diskMeta, file, stream), "load_disk");
                    try
                    {
                        var originals = TestHelpers.SnapshotTensors(diskData);

                        var memoryBytes = TestHelpers.SaveToBuffer(diskData, diskMeta, 2048);
                        TestHelpers.LoadAndCompare(memoryBytes, stream, originals);
                    }
                    finally
                    {
                        MlxMap.StringToArrayFree(diskData);
                        MlxMap.StringToStringFree(diskMeta);
                    }
                }
                finally
                {
                    MlxMap.StringToArrayFree(seed);
                    MlxMap.StringToStringFree(metaSeed);
                }
            });
        }
        finally
        {
            File.Delete(file);
        }
    }

    [Test]
    public void CustomMetalKernelExp()
    {
        TestHelpers.RequireNativeOrIgnore();
        MlxStreamHandle s;
        try
        {
            s = MlxStream.DefaultGpuStreamNew();
        }
        catch
        {
            Assert.Ignore("no gpu");

            return;
        }

        var data = new[] { 0f, 1f, 2f, 3f };
        var shape = new[] { 4 };
        fixed (float* pd = data)
        fixed (int* ps = shape)
        {
            var input = MlxArray.NewData(pd, ps, shape.Length, MlxDType.MLX_FLOAT32);
            TestHelpers.Ok(MlxOps.Exp(out var expected, input, s), "exp");
            TestHelpers.Ok(MlxArray.Eval(expected), "eval");
            var inNames = MlxVector.StringNewValue("inp");
            var outNames = MlxVector.StringNewValue("out");
            const string source = "uint elem = thread_position_in_grid.x; T tmp = inp[elem]; out[elem] = metal::exp(tmp);";
            var kernel = MlxFast.MetalKernelNew("myexp", inNames, outNames, source, "", true, false);
            var config = MlxFast.MetalKernelConfigNew();
            MlxFast.MetalKernelConfigAddTemplateArgDType(config, "T", MlxDType.MLX_FLOAT32);
            MlxFast.MetalKernelConfigSetGrid(config, data.Length, 1, 1);
            MlxFast.MetalKernelConfigSetThreadGroup(config, 256, 1, 1);
            MlxFast.MetalKernelConfigAddOutputArg(config, ps, (nuint)shape.Length, MlxDType.MLX_FLOAT32);
            var inputs = MlxVector.ArrayNewValue(input);
            TestHelpers.Ok(MlxFast.MetalKernelApply(out var outputs, kernel, inputs, config, s), "metal_apply");
            MlxVector.ArrayGet(out var actual, outputs, 0);
            TestHelpers.Ok(MlxArray.Eval(actual), "eval");
            var ev = TestHelpers.ToFloat32(expected);
            var av = TestHelpers.ToFloat32(actual);
            for (var i = 0; i < ev.Length; i++)
                Assert.That(av[i], Is.EqualTo(ev[i]).Within(1e-6));
            MlxArray.Free(input);
            MlxArray.Free(expected);
            MlxArray.Free(actual);
            MlxVector.ArrayFree(inputs);
            MlxVector.ArrayFree(outputs);
            MlxVector.StringFree(inNames);
            MlxVector.StringFree(outNames);
            MlxFast.MetalKernelFree(kernel);
            MlxFast.MetalKernelConfigFree(config);
        }

        MlxStream.Free(s);
    }
}