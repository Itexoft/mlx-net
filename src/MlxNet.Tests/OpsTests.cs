// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using NUnit.Framework;
using Itexoft.Mlx;

[TestFixture]
public unsafe class OpsTests
{
    [Test]
    public void Zeros_Ones_Full_Sum()
    {
        TestHelpers.RequireNativeOrIgnore();
        var s = TestHelpers.CpuStream();
        var dims = new[] { 2, 3 };
        fixed (int* shape = dims)
        {
            TestHelpers.Ok(MlxOps.Zeros(out var z, shape, (nuint)dims.Length, MlxDType.MLX_FLOAT32, s), "zeros");
            TestHelpers.Ok(MlxOps.Ones(out var o, shape, (nuint)dims.Length, MlxDType.MLX_FLOAT32, s), "ones");
            var two = MlxArray.NewFloat32(2f);
            TestHelpers.Ok(MlxOps.Full(out var f, shape, (nuint)dims.Length, two, MlxDType.MLX_FLOAT32, s), "full");
            TestHelpers.Ok(MlxOps.Sum(out var sz, z, false, s), "sum z");
            TestHelpers.Ok(MlxOps.Sum(out var so, o, false, s), "sum o");
            TestHelpers.Ok(MlxOps.Sum(out var sf, f, false, s), "sum f");
            TestHelpers.Ok(MlxArray.Eval(sz), "eval sz");
            var vz = TestHelpers.ToFloat32(sz);
            TestHelpers.Ok(MlxArray.Eval(so), "eval so");
            var vo = TestHelpers.ToFloat32(so);
            TestHelpers.Ok(MlxArray.Eval(sf), "eval sf");
            var vf = TestHelpers.ToFloat32(sf);
            Assert.That(vz[0], Is.EqualTo(0f).Within(1e-6));
            Assert.That(vo[0], Is.EqualTo(6f).Within(1e-6));
            Assert.That(vf[0], Is.EqualTo(12f).Within(1e-6));
            MlxArray.Free(z);
            MlxArray.Free(o);
            MlxArray.Free(two);
            MlxArray.Free(f);
            MlxArray.Free(sz);
            MlxArray.Free(so);
            MlxArray.Free(sf);
        }
    }

    [Test]
    public void Add_Two_Small_Arrays()
    {
        TestHelpers.RequireNativeOrIgnore();
        var s = TestHelpers.CpuStream();
        var dims = new[] { 2, 2 };
        fixed (int* shape = dims)
        {
            var one = MlxArray.NewFloat32(1f);
            TestHelpers.Ok(MlxOps.Full(out var a, shape, (nuint)dims.Length, one, MlxDType.MLX_FLOAT32, s), "full a");
            TestHelpers.Ok(MlxOps.Full(out var b, shape, (nuint)dims.Length, one, MlxDType.MLX_FLOAT32, s), "full b");
            TestHelpers.Ok(MlxOps.Add(out var c, a, b, s), "add");
            TestHelpers.Ok(MlxArray.Eval(c), "eval");
            var v = TestHelpers.ToFloat32(c);
            Assert.That(v[0], Is.EqualTo(2f).Within(1e-6));
            MlxArray.Free(one);
            MlxArray.Free(a);
            MlxArray.Free(b);
            MlxArray.Free(c);
        }
    }

    [Test]
    public void Matmul_2x3_3x2_Float32()
    {
        TestHelpers.RequireNativeOrIgnore();
        var s = TestHelpers.CpuStream();
        var aData = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var bData = new[] { 7f, 8f, 9f, 10f, 11f, 12f };
        var aShape = new[] { 2, 3 };
        var bShape = new[] { 3, 2 };
        fixed (float* pa = aData)
        fixed (float* pb = bData)
        fixed (int* sa = aShape)
        fixed (int* sb = bShape)
        {
            var a = MlxArray.NewData(pa, sa, aShape.Length, MlxDType.MLX_FLOAT32);
            var b = MlxArray.NewData(pb, sb, bShape.Length, MlxDType.MLX_FLOAT32);
            TestHelpers.Ok(MlxOps.Matmul(out var c, a, b, s), "matmul");
            TestHelpers.Ok(MlxArray.Eval(c), "eval");
            var v = TestHelpers.ToFloat32(c);
            Assert.That(v.Length, Is.EqualTo(4));
            Assert.That(v[0], Is.EqualTo(58f).Within(1e-4));
            Assert.That(v[1], Is.EqualTo(64f).Within(1e-4));
            Assert.That(v[2], Is.EqualTo(139f).Within(1e-4));
            Assert.That(v[3], Is.EqualTo(154f).Within(1e-4));
            MlxArray.Free(a);
            MlxArray.Free(b);
            MlxArray.Free(c);
        }
    }

    [Test]
    public void Softmax_Oneline_Float32()
    {
        TestHelpers.RequireNativeOrIgnore();
        var s = TestHelpers.CpuStream();
        var x = new[] { 1f, 2f, 3f };
        var shape = new[] { 1, 3 };
        fixed (float* px = x)
        fixed (int* ps = shape)
        {
            var a = MlxArray.NewData(px, ps, shape.Length, MlxDType.MLX_FLOAT32);
            TestHelpers.Ok(MlxOps.Softmax(out var y, a, true, s), "softmax");
            TestHelpers.Ok(MlxArray.Eval(y), "eval");
            var v = TestHelpers.ToFloat32(y);
            Assert.That(v[0], Is.EqualTo(0.09003057f).Within(1e-5));
            Assert.That(v[1], Is.EqualTo(0.24472847f).Within(1e-5));
            Assert.That(v[2], Is.EqualTo(0.66524096f).Within(1e-5));
            MlxArray.Free(a);
            MlxArray.Free(y);
        }
    }
}