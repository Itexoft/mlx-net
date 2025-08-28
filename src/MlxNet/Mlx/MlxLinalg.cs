// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static unsafe partial class MlxLinalg
{
    /// <summary>Computes the Cholesky decomposition of a symmetric positive-definite matrix (result is lower-triangular L such that A = L*L^T).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linalg_cholesky")]
    public static partial int Cholesky(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        [MarshalAs(UnmanagedType.I1)] bool upper,
        MlxStreamHandle s
    );

    /// <summary>Computes the inverse of a matrix given its Cholesky factor (uses the Cholesky decomposition to find A-1 efficiently).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linalg_cholesky_inv")]
    public static partial int CholeskyInv(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        [MarshalAs(UnmanagedType.I1)] bool upper,
        MlxStreamHandle s
    );

    /// <summary>Computes the cross product of two 3-element vectors, or of corresponding 3-element vectors along an axis of higher-dimensional arrays.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linalg_cross")]
    public static partial int Cross(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        int axis,
        MlxStreamHandle s
    );

    /// <summary>Computes the eigenvalues and eigenvectors of a square matrix.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linalg_eig")]
    public static partial int Eig(
        out MlxArrayHandle res_0,
        out MlxArrayHandle res_1,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the eigenvalues and eigenvectors of a Hermitian (symmetric) matrix (more efficient for symmetric matrices).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linalg_eigh", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Eigh(
        out MlxArrayHandle res_0,
        out MlxArrayHandle res_1,
        MlxArrayHandle a,
        string UPLO,
        MlxStreamHandle s
    );

    /// <summary>Computes the eigenvalues of a square matrix.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linalg_eigvals")]
    public static partial int Eigvals(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the eigenvalues of a Hermitian (symmetric) matrix.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linalg_eigvalsh", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int Eigvalsh(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        string UPLO,
        MlxStreamHandle s
    );

    /// <summary>Computes the inverse of a square matrix.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linalg_inv")]
    public static partial int Inv(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the LU decomposition of a matrix (factors the matrix into L (lower triangular) and U (upper triangular) matrices).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linalg_lu")]
    public static partial int Lu(
        out MlxVectorArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the LU factorization of a matrix with partial pivoting (returns L, U, and pivot indices).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linalg_lu_factor")]
    public static partial int LuFactor(
        out MlxArrayHandle res_0,
        out MlxArrayHandle res_1,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the specified norm of the array (default might be L2 norm for vectors or Frobenius norm for matrices, depending on parameters).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linalg_norm")]
    public static partial int Norm(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        double ord,
        int* axis,
        nuint axis_num,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Computes the Frobenius norm of a matrix (sqrt of sum of squares of all entries) or other matrix norm as specified.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linalg_norm_matrix", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int NormMatrix(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        string ord,
        int* axis,
        nuint axis_num,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Computes the L2 norm (Euclidean norm) of the array (sqrt of sum of squares of elements).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linalg_norm_l2")]
    public static partial int NormL2(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        int* axis,
        nuint axis_num,
        [MarshalAs(UnmanagedType.I1)] bool keepdims,
        MlxStreamHandle s
    );

    /// <summary>Computes the Moore-Penrose pseudoinverse of a matrix (generalized inverse for possibly non-invertible or non-square matrices).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linalg_pinv")]
    public static partial int Pinv(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Computes the QR decomposition of a matrix (factors the matrix into an orthogonal matrix Q and an upper-triangular matrix R).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linalg_qr")]
    public static partial int Qr(
        out MlxArrayHandle res_0,
        out MlxArrayHandle res_1,
        MlxArrayHandle a,
        MlxStreamHandle s
    );

    /// <summary>Solves the linear system A * x = b for x, given matrix A and vector or matrix b.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linalg_solve")]
    public static partial int Solve(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        MlxStreamHandle s
    );

    /// <summary>Solves a triangular linear system (where A is triangular) for x.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linalg_solve_triangular")]
    public static partial int SolveTriangular(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        MlxArrayHandle b,
        [MarshalAs(UnmanagedType.I1)] bool upper,
        MlxStreamHandle s
    );

    /// <summary>Computes the singular value decomposition of a matrix (factors A = U * S * V^T, returning singular values and optionally U, V).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linalg_svd")]
    public static partial int Svd(
        out MlxVectorArrayHandle res,
        MlxArrayHandle a,
        [MarshalAs(UnmanagedType.I1)] bool compute_uv,
        MlxStreamHandle s
    );

    /// <summary>Computes the inverse of a triangular matrix (either upper or lower triangular invert).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_linalg_tri_inv")]
    public static partial int TriInv(
        out MlxArrayHandle res,
        MlxArrayHandle a,
        [MarshalAs(UnmanagedType.I1)] bool upper,
        MlxStreamHandle s
    );
}