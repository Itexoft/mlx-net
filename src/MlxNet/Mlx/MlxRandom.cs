// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Runtime.InteropServices;

namespace Itexoft.Mlx;

public static unsafe partial class MlxRandom
{
    /// <summary>Generates an array of random booleans (or 0/1 values) from a Bernoulli distribution with a given probability of True(1).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_random_bernoulli")]
    public static partial int Bernoulli(
        out MlxArrayHandle res,
        MlxArrayHandle p,
        int* shape,
        nuint shape_num,
        MlxArrayHandle key,
        MlxStreamHandle s
    );

    /// <summary>Generates an array of random bits (as integers of a specified bit-width).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_random_bits")]
    public static partial int Bits(
        out MlxArrayHandle res,
        int* shape,
        nuint shape_num,
        int width,
        MlxArrayHandle key,
        MlxStreamHandle s
    );

    /// <summary>Draws random samples (indices) from a categorical distribution defined by given probabilities, returning samples in a specified output shape.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_random_categorical_shape")]
    public static partial int CategoricalShape(
        out MlxArrayHandle res,
        MlxArrayHandle logits,
        int axis,
        int* shape,
        nuint shape_num,
        MlxArrayHandle key,
        MlxStreamHandle s
    );

    /// <summary>Draws random samples from a categorical distribution with a specified number of samples.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_random_categorical_num_samples")]
    public static partial int CategoricalNumSamples(
        out MlxArrayHandle res,
        MlxArrayHandle logits_,
        int axis,
        int num_samples,
        MlxArrayHandle key,
        MlxStreamHandle s
    );

    /// <summary>Draws random samples from a categorical distribution defined by logits.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_random_categorical")]
    public static partial int Categorical(
        out MlxArrayHandle res,
        MlxArrayHandle logits,
        int axis,
        MlxArrayHandle key,
        MlxStreamHandle s
    );

    /// <summary>Generates samples from a Gumbel distribution.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_random_gumbel")]
    public static partial int Gumbel(
        out MlxArrayHandle res,
        int* shape,
        nuint shape_num,
        MlxDType dtype,
        MlxArrayHandle key,
        MlxStreamHandle s
    );

    /// <summary>Returns a new random PRNG key.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_random_key")]
    public static partial int Key(
        out MlxArrayHandle res,
        ulong seed
    );

    /// <summary>Generates samples from a Laplace (double exponential) distribution.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_random_laplace")]
    public static partial int Laplace(
        out MlxArrayHandle res,
        int* shape,
        nuint shape_num,
        MlxDType dtype,
        float loc,
        float scale,
        MlxArrayHandle key,
        MlxStreamHandle s
    );

    /// <summary>Generates samples from a multivariate normal (Gaussian) distribution given a mean vector and covariance matrix.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_random_multivariate_normal")]
    public static partial int MultivariateNormal(
        out MlxArrayHandle res,
        MlxArrayHandle mean,
        MlxArrayHandle cov,
        int* shape,
        nuint shape_num,
        MlxDType dtype,
        MlxArrayHandle key,
        MlxStreamHandle s
    );

    /// <summary>Generates samples from a normal (Gaussian) distribution.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_random_normal")]
    public static partial int Normal(
        out MlxArrayHandle res,
        int* shape,
        nuint shape_num,
        MlxDType dtype,
        float loc,
        float scale,
        MlxArrayHandle key,
        MlxStreamHandle s
    );

    /// <summary>Generates samples from a normal distribution, broadcasting parameters across a larger shape.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_random_normal_broadcast")]
    public static partial int NormalBroadcast(
        out MlxArrayHandle res,
        int* shape,
        nuint shape_num,
        MlxDType dtype,
        MlxArrayHandle loc,
        MlxArrayHandle scale,
        MlxArrayHandle key,
        MlxStreamHandle s
    );

    /// <summary>Returns a random permutation of integers or permutes a given array along its first axis.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_random_permutation")]
    public static partial int Permutation(
        out MlxArrayHandle res,
        MlxArrayHandle x,
        int axis,
        MlxArrayHandle key,
        MlxStreamHandle s
    );

    /// <summary>Generates a random permutation of the range [0, n).</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_random_permutation_arange")]
    public static partial int PermutationArange(
        out MlxArrayHandle res,
        int x,
        MlxArrayHandle key,
        MlxStreamHandle s
    );

    /// <summary>Generates random integers uniformly between a low (inclusive) and high (exclusive) bound.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_random_randint")]
    public static partial int Randint(
        out MlxArrayHandle res,
        MlxArrayHandle low,
        MlxArrayHandle high,
        int* shape,
        nuint shape_num,
        MlxDType dtype,
        MlxArrayHandle key,
        MlxStreamHandle s
    );

    /// <summary>Sets the global seed for MLX’s random number generator.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_random_seed")]
    public static partial int Seed(
        ulong seed
    );

    /// <summary>Splits a random key into the specified number of new keys.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_random_split_num")]
    public static partial int SplitNum(
        out MlxArrayHandle res,
        MlxArrayHandle key,
        int num,
        MlxStreamHandle s
    );

    /// <summary>Splits a random key into two new random keys.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_random_split")]
    public static partial int Split(
        out MlxArrayHandle res_0,
        out MlxArrayHandle res_1,
        MlxArrayHandle key,
        MlxStreamHandle s
    );

    /// <summary>Generates samples from a truncated normal distribution.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_random_truncated_normal")]
    public static partial int TruncatedNormal(
        out MlxArrayHandle res,
        MlxArrayHandle lower,
        MlxArrayHandle upper,
        int* shape,
        nuint shape_num,
        MlxDType dtype,
        MlxArrayHandle key,
        MlxStreamHandle s
    );

    /// <summary>Generates samples from a uniform distribution over a specified range.</summary>
    [LibraryImport(Common.Lib, EntryPoint = "mlx_random_uniform")]
    public static partial int Uniform(
        out MlxArrayHandle res,
        MlxArrayHandle low,
        MlxArrayHandle high,
        int* shape,
        nuint shape_num,
        MlxDType dtype,
        MlxArrayHandle key,
        MlxStreamHandle s
    );
}