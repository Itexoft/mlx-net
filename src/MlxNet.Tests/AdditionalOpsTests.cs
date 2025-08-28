// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using NUnit.Framework;
using Itexoft.Mlx;

[TestFixture]
public unsafe class AdditionalOpsTests
{
    [Test]
    public void Arange_Produces_Sequence()
    {
        TestHelpers.RequireNativeOrIgnore();
        var s = TestHelpers.CpuStream();
        var rc = MlxOps.Arange(out var a, 0, 5, 1, MlxDType.MLX_FLOAT32, s);
        TestHelpers.Ok(rc, "arange");
        TestHelpers.Ok(MlxArray.Eval(a), "eval");
        var v = TestHelpers.ToFloat32(a);
        Assert.That(v.Length, Is.EqualTo(5));
        Assert.That(v[0], Is.EqualTo(0f).Within(1e-6));
        Assert.That(v[4], Is.EqualTo(4f).Within(1e-6));
        MlxArray.Free(a);
    }

    [Test]
    public void Reshape_Changes_Shape()
    {
        TestHelpers.RequireNativeOrIgnore();
        var s = TestHelpers.CpuStream();
        TestHelpers.Ok(MlxOps.Arange(out var a, 0, 6, 1, MlxDType.MLX_FLOAT32, s), "arange");
        var dims = new[] { 3, 2 };
        fixed (int* shape = dims)
        {
            TestHelpers.Ok(MlxOps.Reshape(out var r, a, shape, (nuint)dims.Length, s), "reshape");
            var sh = TestHelpers.ShapeOf(r);
            Assert.That(sh[0], Is.EqualTo(3));
            Assert.That(sh[1], Is.EqualTo(2));
            MlxArray.Free(a);
            MlxArray.Free(r);
        }
    }

    [Test]
    public void Exp_On_Zeros_Gives_Ones()
    {
        TestHelpers.RequireNativeOrIgnore();
        var s = TestHelpers.CpuStream();
        var zero = MlxArray.NewFloat32(0f);
        var dims = new[] { 2, 2 };
        fixed (int* shape = dims)
        {
            TestHelpers.Ok(MlxOps.Full(out var a, shape, (nuint)dims.Length, zero, MlxDType.MLX_FLOAT32, s), "full");
            TestHelpers.Ok(MlxOps.Exp(out var e, a, s), "exp");
            TestHelpers.Ok(MlxArray.Eval(e), "eval");
            var v = TestHelpers.ToFloat32(e);
            Assert.That(v[0], Is.EqualTo(1f).Within(1e-6));
            MlxArray.Free(zero);
            MlxArray.Free(a);
            MlxArray.Free(e);
        }
    }

    [Test]
    public void Argmax_Returns_Index()
    {
        TestHelpers.RequireNativeOrIgnore();
        var s = TestHelpers.CpuStream();
        var data = new[] { 1f, 3f, 2f };
        var shape = new[] { 3 };
        fixed (float* pd = data)
        fixed (int* ps = shape)
        {
            var a = MlxArray.NewData(pd, ps, shape.Length, MlxDType.MLX_FLOAT32);
            TestHelpers.Ok(MlxOps.Argmax(out var idx, a, false, s), "argmax");
            TestHelpers.Ok(MlxArray.Eval(idx), "eval");
            var v = TestHelpers.ToInt32(idx);
            Assert.That(v[0], Is.EqualTo(1));
            MlxArray.Free(a);
            MlxArray.Free(idx);
        }
    }
}