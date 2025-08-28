// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static unsafe partial class MlxFft
{
    /// <summary>Computes the 1-dimensional discrete Fourier Transform (FFT) of the input array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fft_fft")]
    public static partial int Fft(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int n,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Computes the 2-dimensional Fourier Transform of the input (performing FFT on each dimension of a 2D array).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fft_fft2")]
    public static partial int Fft2(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* n,
        nuint n_num,
        int* axes,
        nuint axes_num,
        MlxStreamHandle s
    );

    /// <summary>Computes the n-dimensional discrete Fourier Transform of the input array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fft_fftn")]
    public static partial int Fftn(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* n,
        nuint n_num,
        int* axes,
        nuint axes_num,
        MlxStreamHandle s
    );

    /// <summary>Computes the inverse 1-D Fourier Transform of the input (inverse FFT).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fft_ifft")]
    public static partial int Ifft(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int n,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Computes the inverse 2-D Fourier Transform of the input array.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fft_ifft2")]
    public static partial int Ifft2(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* n,
        nuint n_num,
        int* axes,
        nuint axes_num,
        MlxStreamHandle s
    );

    /// <summary>Computes the inverse n-dimensional Fourier Transform of the input.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fft_ifftn")]
    public static partial int Ifftn(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* n,
        nuint n_num,
        int* axes,
        nuint axes_num,
        MlxStreamHandle s
    );

    /// <summary>Computes the inverse FFT for a real-input transform (the inverse of a real FFT, yielding a real output).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fft_irfft")]
    public static partial int Irfft(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int n,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Computes the 2-D inverse FFT for real-valued frequency domain data.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fft_irfft2")]
    public static partial int Irfft2(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* n,
        nuint n_num,
        int* axes,
        nuint axes_num,
        MlxStreamHandle s
    );

    /// <summary>Computes the n-dimensional inverse FFT for real-valued frequency domain data.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fft_irfftn")]
    public static partial int Irfftn(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* n,
        nuint n_num,
        int* axes,
        nuint axes_num,
        MlxStreamHandle s
    );

    /// <summary>Computes the FFT of a real-valued input, returning only the non-redundant half of the frequency spectrum.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fft_rfft")]
    public static partial int Rfft(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int n,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Computes the 2-D FFT of real-valued input, returning the half-spectrum.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fft_rfft2")]
    public static partial int Rfft2(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* n,
        nuint n_num,
        int* axes,
        nuint axes_num,
        MlxStreamHandle s
    );

    /// <summary>Computes the n-dimensional FFT of real-valued input, returning the half-spectrum.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_fft_rfftn")]
    public static partial int Rfftn(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* n,
        nuint n_num,
        int* axes,
        nuint axes_num,
        MlxStreamHandle s
    );
}