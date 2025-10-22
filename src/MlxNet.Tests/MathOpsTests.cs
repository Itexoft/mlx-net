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
        var data = new[] { -1f, -2f, 3f };
        TestHelpers.WithStream(stream =>
        {
            TestHelpers.WithShape(
                [3],
                (shape, rank) =>
                {
                    fixed (float* pd = data)
                    {
                        var a = MlxArray.NewData(pd, shape, (int)rank, MlxDType.MLX_FLOAT32);
                        TestHelpers.Ok(MlxOps.Abs(out var b, a, stream), "abs");
                        TestHelpers.EvalArray(b);
                        var v = TestHelpers.ToFloat32(b);
                        Assert.That(v[0], Is.EqualTo(1f).Within(1e-6));
                        Assert.That(v[1], Is.EqualTo(2f).Within(1e-6));
                        Assert.That(v[2], Is.EqualTo(3f).Within(1e-6));
                        MlxArray.Free(a);
                        MlxArray.Free(b);
                    }
                });
        });
    }

    [Test]
    public void Negative_On_Positive()
    {
        TestHelpers.RequireNativeOrIgnore();
        var data = new[] { 1f, -2f, 3f };
        TestHelpers.WithStream(stream =>
        {
            TestHelpers.WithShape(
                [3],
                (shape, rank) =>
                {
                    fixed (float* pd = data)
                    {
                        var a = MlxArray.NewData(pd, shape, (int)rank, MlxDType.MLX_FLOAT32);
                        TestHelpers.Ok(MlxOps.Negative(out var n, a, stream), "negative");
                        TestHelpers.EvalArray(n);
                        var v = TestHelpers.ToFloat32(n);
                        Assert.That(v[0], Is.EqualTo(-1f).Within(1e-6));
                        Assert.That(v[1], Is.EqualTo(2f).Within(1e-6));
                        Assert.That(v[2], Is.EqualTo(-3f).Within(1e-6));
                        MlxArray.Free(a);
                        MlxArray.Free(n);
                    }
                });
        });
    }

    [Test]
    public void Multiply_Two_Arrays()
    {
        TestHelpers.RequireNativeOrIgnore();
        TestHelpers.WithStream(stream =>
        {
            TestHelpers.WithShape(
                [2, 2],
                (shape, rank) =>
                {
                    var two = MlxArray.NewFloat32(2f);
                    var three = MlxArray.NewFloat32(3f);
                    TestHelpers.Ok(MlxOps.Full(out var a, shape, rank, two, MlxDType.MLX_FLOAT32, stream), "full a");
                    TestHelpers.Ok(MlxOps.Full(out var b, shape, rank, three, MlxDType.MLX_FLOAT32, stream), "full b");
                    TestHelpers.Ok(MlxOps.Multiply(out var c, a, b, stream), "multiply");
                    TestHelpers.EvalArray(c);
                    var v = TestHelpers.ToFloat32(c);
                    Assert.That(v[0], Is.EqualTo(6f).Within(1e-6));
                    MlxArray.Free(two);
                    MlxArray.Free(three);
                    MlxArray.Free(a);
                    MlxArray.Free(b);
                    MlxArray.Free(c);
                });
        });
    }
}