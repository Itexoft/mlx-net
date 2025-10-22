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
        var stream = TestHelpers.NewCpuStream();
        try
        {
            var rc = MlxOps.Arange(out var array, 0, 5, 1, MlxDType.MLX_FLOAT32, stream);
            TestHelpers.Ok(rc, "arange");
            TestHelpers.EvalArray(array, "eval arange");

            var values = TestHelpers.ToFloat32(array);
            Assert.That(values.Length, Is.EqualTo(5));
            Assert.That(values[0], Is.EqualTo(0f).Within(1e-6));
            Assert.That(values[4], Is.EqualTo(4f).Within(1e-6));

            MlxArray.Free(array);
        }
        finally
        {
            TestHelpers.ReleaseStream(stream);
        }
    }

    [Test]
    public void Reshape_Changes_Shape()
    {
        TestHelpers.RequireNativeOrIgnore();
        var stream = TestHelpers.NewCpuStream();
        try
        {
            TestHelpers.Ok(MlxOps.Arange(out var array, 0, 6, 1, MlxDType.MLX_FLOAT32, stream), "arange");
            var dims = new[] { 3, 2 };

            var shape = TestHelpers.AllocShape(dims);
            try
            {
                TestHelpers.Ok(MlxOps.Reshape(out var reshaped, array, shape, (nuint)dims.Length, stream), "reshape");
                TestHelpers.EvalArray(reshaped, "eval reshape");

                var detectedShape = TestHelpers.ShapeOf(reshaped);
                Assert.That(detectedShape[0], Is.EqualTo(3));
                Assert.That(detectedShape[1], Is.EqualTo(2));

                MlxArray.Free(array);
                MlxArray.Free(reshaped);
            }
            finally
            {
                TestHelpers.FreeShape(shape);
            }
        }
        finally
        {
            TestHelpers.ReleaseStream(stream);
        }
    }

    [Test]
    public void Exp_On_Zeros_Gives_Ones()
    {
        TestHelpers.RequireNativeOrIgnore();
        var stream = TestHelpers.NewCpuStream();
        try
        {
            var zero = MlxArray.NewFloat32(0f);
            var dims = new[] { 2, 2 };

            var shape = TestHelpers.AllocShape(dims);
            try
            {
                TestHelpers.Ok(MlxOps.Full(out var array, shape, (nuint)dims.Length, zero, MlxDType.MLX_FLOAT32, stream), "full");
                TestHelpers.Ok(MlxOps.Exp(out var exponentiated, array, stream), "exp");
                TestHelpers.EvalArray(exponentiated, "eval exp");

                var values = TestHelpers.ToFloat32(exponentiated);
                Assert.That(values[0], Is.EqualTo(1f).Within(1e-6));

                MlxArray.Free(zero);
                MlxArray.Free(array);
                MlxArray.Free(exponentiated);
            }
            finally
            {
                TestHelpers.FreeShape(shape);
            }
        }
        finally
        {
            TestHelpers.ReleaseStream(stream);
        }
    }

    [Test]
    public void Argmax_Returns_Index()
    {
        TestHelpers.RequireNativeOrIgnore();
        var stream = TestHelpers.NewCpuStream();
        try
        {
            var data = new[] { 1f, 3f, 2f };
            var dims = new[] { 3 };

            fixed (float* source = data)
            {
                var shape = TestHelpers.AllocShape(dims);
                try
                {
                    var input = MlxArray.NewData(source, shape, dims.Length, MlxDType.MLX_FLOAT32);
                    TestHelpers.Ok(MlxOps.Argmax(out var indices, input, false, stream), "argmax");
                    TestHelpers.EvalArray(indices, "eval argmax");

                    var values = TestHelpers.ToInt32(indices);
                    Assert.That(values[0], Is.EqualTo(1));

                    MlxArray.Free(input);
                    MlxArray.Free(indices);
                }
                finally
                {
                    TestHelpers.FreeShape(shape);
                }
            }
        }
        finally
        {
            TestHelpers.ReleaseStream(stream);
        }
    }
}