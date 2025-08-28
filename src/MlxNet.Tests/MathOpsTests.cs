// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using NUnit.Framework;
using Itexoft.Mlx;

[TestFixture]
public unsafe class MathOpsTests
{
    [Test]
    public void Abs_On_Negative()
    {
        TestHelpers.RequireNativeOrIgnore();
        var s = TestHelpers.CpuStream();
        var data = new[] { -1f, -2f, 3f };
        var shape = new[] { 3 };
        fixed (float* pd = data)
        fixed (int* ps = shape)
        {
            var a = MlxArray.NewData(pd, ps, shape.Length, MlxDType.MLX_FLOAT32);
            TestHelpers.Ok(MlxOps.Abs(out var b, a, s), "abs");
            TestHelpers.Ok(MlxArray.Eval(b), "eval");
            var v = TestHelpers.ToFloat32(b);
            Assert.That(v[0], Is.EqualTo(1f).Within(1e-6));
            Assert.That(v[1], Is.EqualTo(2f).Within(1e-6));
            Assert.That(v[2], Is.EqualTo(3f).Within(1e-6));
            MlxArray.Free(a);
            MlxArray.Free(b);
        }
    }

    [Test]
    public void Negative_On_Positive()
    {
        TestHelpers.RequireNativeOrIgnore();
        var s = TestHelpers.CpuStream();
        var data = new[] { 1f, -2f, 3f };
        var shape = new[] { 3 };
        fixed (float* pd = data)
        fixed (int* ps = shape)
        {
            var a = MlxArray.NewData(pd, ps, shape.Length, MlxDType.MLX_FLOAT32);
            TestHelpers.Ok(MlxOps.Negative(out var n, a, s), "negative");
            TestHelpers.Ok(MlxArray.Eval(n), "eval");
            var v = TestHelpers.ToFloat32(n);
            Assert.That(v[0], Is.EqualTo(-1f).Within(1e-6));
            Assert.That(v[1], Is.EqualTo(2f).Within(1e-6));
            Assert.That(v[2], Is.EqualTo(-3f).Within(1e-6));
            MlxArray.Free(a);
            MlxArray.Free(n);
        }
    }

    [Test]
    public void Multiply_Two_Arrays()
    {
        TestHelpers.RequireNativeOrIgnore();
        var s = TestHelpers.CpuStream();
        var dims = new[] { 2, 2 };
        fixed (int* shape = dims)
        {
            var two = MlxArray.NewFloat32(2f);
            var three = MlxArray.NewFloat32(3f);
            TestHelpers.Ok(MlxOps.Full(out var a, shape, (nuint)dims.Length, two, MlxDType.MLX_FLOAT32, s), "full a");
            TestHelpers.Ok(MlxOps.Full(out var b, shape, (nuint)dims.Length, three, MlxDType.MLX_FLOAT32, s), "full b");
            TestHelpers.Ok(MlxOps.Multiply(out var c, a, b, s), "multiply");
            TestHelpers.Ok(MlxArray.Eval(c), "eval");
            var v = TestHelpers.ToFloat32(c);
            Assert.That(v[0], Is.EqualTo(6f).Within(1e-6));
            MlxArray.Free(two);
            MlxArray.Free(three);
            MlxArray.Free(a);
            MlxArray.Free(b);
            MlxArray.Free(c);
        }
    }
}