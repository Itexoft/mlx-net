// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;

namespace Itexoft.Mlx.Nn;

/// <summary>
/// Base class for convolutional layers, handling common parameter initialization.
/// </summary>
public abstract class ConvolutionBase : Module, IUnaryLayer
{
    protected ConvolutionBase(MlxArrayHandle weight, MlxArrayHandle? bias)
    {
        this.Weight = this.RegisterParameter("weight", weight);

        if (!TensorUtilities.IsNull(bias ?? default))
            this.Bias = this.RegisterParameter("bias", bias!.Value);
    }

    protected ConvolutionBase(MlxArrayHandle weight, bool bias)
    {
        this.Weight = this.RegisterParameter("weight", weight);

        if (bias)
            this.Bias = this.RegisterParameter("bias", TensorFactory.Zeros([weight.Dim(0)]));
    }

    protected ModuleParameter Weight { get; }

    protected ModuleParameter? Bias { get; }

    public ModuleParameter KernelParameter => this.Weight;

    public ModuleParameter? BiasParameter => this.Bias;

    public abstract MlxArrayHandle Forward(MlxArrayHandle input);

    protected static float ComputeScale(int denominator) => MathF.Sqrt(1f / denominator);
}

/// <summary>
/// 1D convolution over NLC-formatted tensors.
/// </summary>
public sealed class Conv1D : ConvolutionBase
{
    private readonly int dilation;
    private readonly int groups;
    private readonly int padding;
    private readonly int stride;

    public Conv1D(
        int inputChannels,
        int outputChannels,
        int kernelSize,
        int stride = 1,
        int padding = 0,
        int dilation = 1,
        int groups = 1,
        bool bias = true) : base(
        TensorFactory.Uniform(
            -ComputeScale(inputChannels * kernelSize),
            ComputeScale(inputChannels * kernelSize),
            [outputChannels, kernelSize, inputChannels / groups]),
        bias)
    {
        if (inputChannels % groups != 0)
            throw new ArgumentException("Input channels must be divisible by number of groups.", nameof(groups));

        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        this.groups = groups;
    }

    public override MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var status = MlxOps.Conv1D(
            out var result,
            input,
            this.Weight.Value,
            this.stride,
            this.padding,
            this.dilation,
            this.groups,
            TensorUtilities.DefaultStream());

        TensorUtilities.CheckStatus(status, "conv1d");

        if (this.Bias is { } bias)
        {
            var output = result.Add(bias.Value);
            MlxArray.Free(result);

            return output;
        }

        return result;
    }
}

/// <summary>
/// 2D convolution over NHWC-formatted tensors.
/// </summary>
public sealed class Conv2D : ConvolutionBase
{
    private readonly IntPair dilation;
    private readonly int groups;
    private readonly IntPair padding;
    private readonly IntPair stride;

    public Conv2D(
        int inputChannels,
        int outputChannels,
        IntPair kernelSize,
        IntPair? stride = null,
        IntPair? padding = null,
        IntPair? dilation = null,
        int groups = 1,
        bool bias = true) : base(
        TensorFactory.Uniform(
            -ComputeScale(inputChannels * kernelSize.First * kernelSize.Second),
            ComputeScale(inputChannels * kernelSize.First * kernelSize.Second),
            [outputChannels, kernelSize.First, kernelSize.Second, inputChannels / groups]),
        bias)
    {
        if (inputChannels % groups != 0)
            throw new ArgumentException("Input channels must be divisible by number of groups.", nameof(groups));

        this.stride = stride ?? new IntPair(1, 1);
        this.padding = padding ?? new IntPair(0, 0);
        this.dilation = dilation ?? new IntPair(1, 1);
        this.groups = groups;
    }

    public override MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var status = MlxOps.Conv2D(
            out var result,
            input,
            this.Weight.Value,
            this.stride.First,
            this.stride.Second,
            this.padding.First,
            this.padding.Second,
            this.dilation.First,
            this.dilation.Second,
            this.groups,
            TensorUtilities.DefaultStream());

        TensorUtilities.CheckStatus(status, "conv2d");

        if (this.Bias is { } bias)
        {
            var output = result.Add(bias.Value);
            MlxArray.Free(result);

            return output;
        }

        return result;
    }
}

/// <summary>
/// 3D convolution over NDHWC-formatted tensors.
/// </summary>
public sealed class Conv3D(
    int inputChannels,
    int outputChannels,
    IntTriple kernelSize,
    IntTriple? stride = null,
    IntTriple? padding = null,
    IntTriple? dilation = null,
    int groups = 1,
    bool bias = true) : ConvolutionBase(CreateWeight(inputChannels, outputChannels, kernelSize, groups), bias)
{
    private readonly IntTriple dilation = dilation ?? new IntTriple(1, 1, 1);
    private readonly int groups = groups;
    private readonly IntTriple padding = padding ?? new IntTriple(0, 0, 0);
    private readonly IntTriple stride = stride ?? new IntTriple(1, 1, 1);

    public override MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var status = MlxOps.Conv3D(
            out var result,
            input,
            this.Weight.Value,
            this.stride.First,
            this.stride.Second,
            this.stride.Third,
            this.padding.First,
            this.padding.Second,
            this.padding.Third,
            this.dilation.First,
            this.dilation.Second,
            this.dilation.Third,
            this.groups,
            TensorUtilities.DefaultStream());

        TensorUtilities.CheckStatus(status, "conv3d");

        if (this.Bias is { } bias)
        {
            var output = result.Add(bias.Value);
            MlxArray.Free(result);

            return output;
        }

        return result;
    }

    private static MlxArrayHandle CreateWeight(int inputChannels, int outputChannels, IntTriple kernelSize, int groups)
    {
        if (groups <= 0)
            throw new ArgumentOutOfRangeException(nameof(groups));

        if (inputChannels % groups != 0)
            throw new ArgumentException("Input channels must be divisible by number of groups.", nameof(groups));

        var kernelVolume = kernelSize.First * kernelSize.Second * kernelSize.Third;
        var limit = ComputeScale(inputChannels * kernelVolume);

        return TensorFactory.Uniform(-limit, limit, [outputChannels, kernelSize.First, kernelSize.Second, kernelSize.Third, inputChannels / groups]);
    }
}

/// <summary>
/// 1D transposed convolution over NLC-formatted tensors.
/// </summary>
public sealed class ConvTranspose1D : ConvolutionBase
{
    private readonly int dilation;
    private readonly int groups;
    private readonly int outputPadding;
    private readonly int padding;
    private readonly int stride;

    public ConvTranspose1D(
        int inputChannels,
        int outputChannels,
        int kernelSize,
        int stride = 1,
        int padding = 0,
        int dilation = 1,
        int groups = 1,
        int outputPadding = 0,
        bool bias = true) : base(
        TensorFactory.Uniform(
            -ComputeScale(inputChannels * kernelSize),
            ComputeScale(inputChannels * kernelSize),
            [outputChannels, kernelSize, inputChannels]),
        bias)
    {
        if (groups <= 0)
            throw new ArgumentOutOfRangeException(nameof(groups));

        if (outputPadding < 0)
            throw new ArgumentOutOfRangeException(nameof(outputPadding));

        this.stride = stride;
        this.padding = padding;
        this.dilation = dilation;
        this.outputPadding = outputPadding;
        this.groups = groups;
    }

    public override MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var status = MlxOps.ConvTranspose1D(
            out var result,
            input,
            this.Weight.Value,
            this.stride,
            this.padding,
            this.dilation,
            this.outputPadding,
            this.groups,
            TensorUtilities.DefaultStream());

        TensorUtilities.CheckStatus(status, "conv_transpose1d");

        if (this.Bias is { } bias)
        {
            var output = result.Add(bias.Value);
            MlxArray.Free(result);

            return output;
        }

        return result;
    }
}

/// <summary>
/// 2D transposed convolution over NHWC-formatted tensors.
/// </summary>
public sealed class ConvTranspose2D : ConvolutionBase
{
    private readonly IntPair dilation;
    private readonly int groups;
    private readonly IntPair outputPadding;
    private readonly IntPair padding;
    private readonly IntPair stride;

    public ConvTranspose2D(
        int inputChannels,
        int outputChannels,
        IntPair kernelSize,
        IntPair? stride = null,
        IntPair? padding = null,
        IntPair? dilation = null,
        int groups = 1,
        IntPair? outputPadding = null,
        bool bias = true) : base(
        TensorFactory.Uniform(
            -ComputeScale(inputChannels * kernelSize.First * kernelSize.Second),
            ComputeScale(inputChannels * kernelSize.First * kernelSize.Second),
            [outputChannels, kernelSize.First, kernelSize.Second, inputChannels]),
        bias)
    {
        if (groups <= 0)
            throw new ArgumentOutOfRangeException(nameof(groups));

        this.stride = stride ?? new IntPair(1, 1);
        this.padding = padding ?? new IntPair(0, 0);
        this.dilation = dilation ?? new IntPair(1, 1);
        this.outputPadding = outputPadding ?? new IntPair(0, 0);
        this.groups = groups;
    }

    public override MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var status = MlxOps.ConvTranspose2D(
            out var result,
            input,
            this.Weight.Value,
            this.stride.First,
            this.stride.Second,
            this.padding.First,
            this.padding.Second,
            this.dilation.First,
            this.dilation.Second,
            this.outputPadding.First,
            this.outputPadding.Second,
            this.groups,
            TensorUtilities.DefaultStream());

        TensorUtilities.CheckStatus(status, "conv_transpose2d");

        if (this.Bias is { } bias)
        {
            var output = result.Add(bias.Value);
            MlxArray.Free(result);

            return output;
        }

        return result;
    }
}

/// <summary>
/// 3D transposed convolution over NDHWC-formatted tensors.
/// </summary>
public sealed class ConvTranspose3D : ConvolutionBase
{
    private readonly IntTriple dilation;
    private readonly int groups;
    private readonly IntTriple outputPadding;
    private readonly IntTriple padding;
    private readonly IntTriple stride;

    public ConvTranspose3D(
        int inputChannels,
        int outputChannels,
        IntTriple kernelSize,
        IntTriple? stride = null,
        IntTriple? padding = null,
        IntTriple? dilation = null,
        int groups = 1,
        IntTriple? outputPadding = null,
        bool bias = true) : base(
        TensorFactory.Uniform(
            -ComputeScale(inputChannels * kernelSize.First * kernelSize.Second * kernelSize.Third),
            ComputeScale(inputChannels * kernelSize.First * kernelSize.Second * kernelSize.Third),
            [outputChannels, kernelSize.First, kernelSize.Second, kernelSize.Third, inputChannels]),
        bias)
    {
        if (groups <= 0)
            throw new ArgumentOutOfRangeException(nameof(groups));

        this.stride = stride ?? new IntTriple(1, 1, 1);
        this.padding = padding ?? new IntTriple(0, 0, 0);
        this.dilation = dilation ?? new IntTriple(1, 1, 1);
        this.outputPadding = outputPadding ?? new IntTriple(0, 0, 0);
        this.groups = groups;
    }

    public override MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var status = MlxOps.ConvTranspose3D(
            out var result,
            input,
            this.Weight.Value,
            this.stride.First,
            this.stride.Second,
            this.stride.Third,
            this.padding.First,
            this.padding.Second,
            this.padding.Third,
            this.dilation.First,
            this.dilation.Second,
            this.dilation.Third,
            this.outputPadding.First,
            this.outputPadding.Second,
            this.outputPadding.Third,
            this.groups,
            TensorUtilities.DefaultStream());

        TensorUtilities.CheckStatus(status, "conv_transpose3d");

        if (this.Bias is { } bias)
        {
            var output = result.Add(bias.Value);
            MlxArray.Free(result);

            return output;
        }

        return result;
    }
}
