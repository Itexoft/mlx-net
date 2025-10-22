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
        TestHelpers.WithStream(stream =>
        {
            TestHelpers.WithShape(
                [2, 3],
                (shape, rank) =>
                {
                    TestHelpers.Ok(MlxOps.Zeros(out var z, shape, rank, MlxDType.MLX_FLOAT32, stream), "zeros");
                    TestHelpers.Ok(MlxOps.Ones(out var o, shape, rank, MlxDType.MLX_FLOAT32, stream), "ones");
                    var two = MlxArray.NewFloat32(2f);
                    TestHelpers.Ok(MlxOps.Full(out var f, shape, rank, two, MlxDType.MLX_FLOAT32, stream), "full");
                    TestHelpers.Ok(MlxOps.Sum(out var sz, z, false, stream), "sum z");
                    TestHelpers.Ok(MlxOps.Sum(out var so, o, false, stream), "sum o");
                    TestHelpers.Ok(MlxOps.Sum(out var sf, f, false, stream), "sum f");
                    TestHelpers.EvalArrays(sz, so, sf);
                    var vz = TestHelpers.ToFloat32(sz);
                    var vo = TestHelpers.ToFloat32(so);
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
                });
        });
    }

    [Test]
    public void Add_Two_Small_Arrays()
    {
        TestHelpers.RequireNativeOrIgnore();
        TestHelpers.WithStream(stream =>
        {
            TestHelpers.WithShape(
                [2, 2],
                (shape, rank) =>
                {
                    var one = MlxArray.NewFloat32(1f);
                    TestHelpers.Ok(MlxOps.Full(out var a, shape, rank, one, MlxDType.MLX_FLOAT32, stream), "full a");
                    TestHelpers.Ok(MlxOps.Full(out var b, shape, rank, one, MlxDType.MLX_FLOAT32, stream), "full b");
                    TestHelpers.Ok(MlxOps.Add(out var c, a, b, stream), "add");
                    TestHelpers.EvalArray(c);
                    var v = TestHelpers.ToFloat32(c);
                    Assert.That(v[0], Is.EqualTo(2f).Within(1e-6));
                    MlxArray.Free(one);
                    MlxArray.Free(a);
                    MlxArray.Free(b);
                    MlxArray.Free(c);
                });
        });
    }

    [Test]
    public void Matmul_2x3_3x2_Float32()
    {
        TestHelpers.RequireNativeOrIgnore();
        TestHelpers.WithStream(stream =>
        {
            var aData = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
            var bData = new[] { 7f, 8f, 9f, 10f, 11f, 12f };
            var aShape = new[] { 2, 3 };
            var bShape = new[] { 3, 2 };

            TestHelpers.WithShape(
                aShape,
                (sa, ra) =>
                {
                    TestHelpers.WithShape(
                        bShape,
                        (sb, rb) =>
                        {
                            fixed (float* pa = aData)
                            fixed (float* pb = bData)
                            {
                                var a = MlxArray.NewData(pa, sa, (int)ra, MlxDType.MLX_FLOAT32);
                                var b = MlxArray.NewData(pb, sb, (int)rb, MlxDType.MLX_FLOAT32);
                                TestHelpers.Ok(MlxOps.Matmul(out var c, a, b, stream), "matmul");
                                TestHelpers.EvalArray(c);
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
                        });
                });
        });
    }

    [Test]
    public void Softmax_Oneline_Float32()
    {
        TestHelpers.RequireNativeOrIgnore();
        var x = new[] { 1f, 2f, 3f };
        TestHelpers.WithStream(stream =>
        {
            TestHelpers.WithShape(
                [1, 3],
                (shape, rank) =>
                {
                    fixed (float* px = x)
                    {
                        var a = MlxArray.NewData(px, shape, (int)rank, MlxDType.MLX_FLOAT32);
                        TestHelpers.Ok(MlxOps.Softmax(out var y, a, true, stream), "softmax");
                        TestHelpers.EvalArray(y);
                        var v = TestHelpers.ToFloat32(y);
                        Assert.That(v[0], Is.EqualTo(0.09003057f).Within(1e-5));
                        Assert.That(v[1], Is.EqualTo(0.24472847f).Within(1e-5));
                        Assert.That(v[2], Is.EqualTo(0.66524096f).Within(1e-5));
                        MlxArray.Free(a);
                        MlxArray.Free(y);
                    }
                });
        });
    }
}