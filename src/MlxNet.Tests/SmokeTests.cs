// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using NUnit.Framework;
using Itexoft.Mlx;

[TestFixture]
public class SmokeTests
{
    [Test]
    public void Version_NotEmpty()
    {
        TestHelpers.RequireNativeOrIgnore();
        var rc = MlxVersion.Version(out var h);
        TestHelpers.Ok(rc, "version");
        var s = System.Runtime.InteropServices.Marshal.PtrToStringUTF8(MlxString.Data(h));
        MlxString.Free(h);
        Assert.That(string.IsNullOrWhiteSpace(s), Is.False);
    }

    [Test]
    public void CpuStream_New_Works()
    {
        TestHelpers.RequireNativeOrIgnore();
        var stream = TestHelpers.NewCpuStream();
        try
        {
            var rc = MlxStream.ToString(out var sh, stream);
            TestHelpers.Ok(rc, "stream tostring");
            var txt = System.Runtime.InteropServices.Marshal.PtrToStringUTF8(MlxString.Data(sh));
            MlxString.Free(sh);
            Assert.That(string.IsNullOrWhiteSpace(txt), Is.False);
        }
        finally
        {
            TestHelpers.ReleaseStream(stream);
        }
    }

    [Test]
    public void Array_NewFloat32_And_Free()
    {
        TestHelpers.RequireNativeOrIgnore();
        var a = MlxArray.NewFloat32(3.5f);
        var dtype = MlxArray.DType(a);
        var size = MlxArray.Size(a);
        Assert.That(dtype, Is.EqualTo(MlxDType.MLX_FLOAT32));
        Assert.That((int)size, Is.EqualTo(1));
        var rc = MlxArray.Free(a);
        TestHelpers.Ok(rc, "array free");
    }
}