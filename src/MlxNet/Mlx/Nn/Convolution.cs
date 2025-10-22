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

    protected static float ComputeScale(int denominator)
        => MathF.Sqrt(1f / denominator);
}

/// <summary>
/// 1D convolution over NLC-formatted tensors.
/// </summary>
public sealed class Conv1d : ConvolutionBase
{
    private readonly int _stride;
    private readonly int _padding;
    private readonly int _dilation;
    private readonly int _groups;

    public Conv1d(
        int inputChannels,
        int outputChannels,
        int kernelSize,
        int stride = 1,
        int padding = 0,
        int dilation = 1,
        int groups = 1,
        bool bias = true)
        : base(
            TensorFactory.Uniform(
                -ComputeScale(inputChannels * kernelSize),
                ComputeScale(inputChannels * kernelSize),
                [outputChannels, kernelSize, inputChannels / groups]),
            bias)
    {
        if (inputChannels % groups != 0)
            throw new ArgumentException("Input channels must be divisible by number of groups.", nameof(groups));

        this._stride = stride;
        this._padding = padding;
        this._dilation = dilation;
        this._groups = groups;
    }

    public override MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var status = MlxOps.Conv1d(
            out var result,
            input,
            this.Weight.Value,
            this._stride,
            this._padding,
            this._dilation,
            this._groups,
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
public sealed class Conv2d : ConvolutionBase
{
    private readonly IntPair _stride;
    private readonly IntPair _padding;
    private readonly IntPair _dilation;
    private readonly int _groups;

    public Conv2d(
        int inputChannels,
        int outputChannels,
        IntPair kernelSize,
        IntPair? stride = null,
        IntPair? padding = null,
        IntPair? dilation = null,
        int groups = 1,
        bool bias = true)
        : base(
            TensorFactory.Uniform(
                -ComputeScale(inputChannels * kernelSize.First * kernelSize.Second),
                ComputeScale(inputChannels * kernelSize.First * kernelSize.Second),
                [outputChannels, kernelSize.First, kernelSize.Second, inputChannels / groups]),
            bias)
    {
        if (inputChannels % groups != 0)
            throw new ArgumentException("Input channels must be divisible by number of groups.", nameof(groups));

        this._stride = stride ?? new IntPair(1, 1);
        this._padding = padding ?? new IntPair(0, 0);
        this._dilation = dilation ?? new IntPair(1, 1);
        this._groups = groups;
    }

    public override MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var status = MlxOps.Conv2d(
            out var result,
            input,
            this.Weight.Value,
            this._stride.First,
            this._stride.Second,
            this._padding.First,
            this._padding.Second,
            this._dilation.First,
            this._dilation.Second,
            this._groups,
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
public sealed class Conv3d(
    int inputChannels,
    int outputChannels,
    IntTriple kernelSize,
    IntTriple? stride = null,
    IntTriple? padding = null,
    IntTriple? dilation = null,
    int groups = 1,
    bool bias = true)
    : ConvolutionBase(
        CreateWeight(inputChannels, outputChannels, kernelSize, groups),
        bias)
{
    private readonly IntTriple _stride = stride ?? new IntTriple(1, 1, 1);
    private readonly IntTriple _padding = padding ?? new IntTriple(0, 0, 0);
    private readonly IntTriple _dilation = dilation ?? new IntTriple(1, 1, 1);
    private readonly int _groups = groups;

    public override MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var status = MlxOps.Conv3d(
            out var result,
            input,
            this.Weight.Value,
            this._stride.First,
            this._stride.Second,
            this._stride.Third,
            this._padding.First,
            this._padding.Second,
            this._padding.Third,
            this._dilation.First,
            this._dilation.Second,
            this._dilation.Third,
            this._groups,
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

        return TensorFactory.Uniform(
            -limit,
            limit,
            [outputChannels, kernelSize.First, kernelSize.Second, kernelSize.Third, inputChannels / groups]);
    }
}

/// <summary>
/// 1D transposed convolution over NLC-formatted tensors.
/// </summary>
public sealed class ConvTranspose1d : ConvolutionBase
{
    private readonly int _stride;
    private readonly int _padding;
    private readonly int _dilation;
    private readonly int _outputPadding;
    private readonly int _groups;

    public ConvTranspose1d(
        int inputChannels,
        int outputChannels,
        int kernelSize,
        int stride = 1,
        int padding = 0,
        int dilation = 1,
        int groups = 1,
        int outputPadding = 0,
        bool bias = true)
        : base(
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

        this._stride = stride;
        this._padding = padding;
        this._dilation = dilation;
        this._outputPadding = outputPadding;
        this._groups = groups;
    }

    public override MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var status = MlxOps.ConvTranspose1d(
            out var result,
            input,
            this.Weight.Value,
            this._stride,
            this._padding,
            this._dilation,
            this._outputPadding,
            this._groups,
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
public sealed class ConvTranspose2d : ConvolutionBase
{
    private readonly IntPair _stride;
    private readonly IntPair _padding;
    private readonly IntPair _dilation;
    private readonly IntPair _outputPadding;
    private readonly int _groups;

    public ConvTranspose2d(
        int inputChannels,
        int outputChannels,
        IntPair kernelSize,
        IntPair? stride = null,
        IntPair? padding = null,
        IntPair? dilation = null,
        int groups = 1,
        IntPair? outputPadding = null,
        bool bias = true)
        : base(
            TensorFactory.Uniform(
                -ComputeScale(inputChannels * kernelSize.First * kernelSize.Second),
                ComputeScale(inputChannels * kernelSize.First * kernelSize.Second),
                [outputChannels, kernelSize.First, kernelSize.Second, inputChannels]),
            bias)
    {
        if (groups <= 0)
            throw new ArgumentOutOfRangeException(nameof(groups));

        this._stride = stride ?? new IntPair(1, 1);
        this._padding = padding ?? new IntPair(0, 0);
        this._dilation = dilation ?? new IntPair(1, 1);
        this._outputPadding = outputPadding ?? new IntPair(0, 0);
        this._groups = groups;
    }

    public override MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var status = MlxOps.ConvTranspose2d(
            out var result,
            input,
            this.Weight.Value,
            this._stride.First,
            this._stride.Second,
            this._padding.First,
            this._padding.Second,
            this._dilation.First,
            this._dilation.Second,
            this._outputPadding.First,
            this._outputPadding.Second,
            this._groups,
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
public sealed class ConvTranspose3d : ConvolutionBase
{
    private readonly IntTriple _stride;
    private readonly IntTriple _padding;
    private readonly IntTriple _dilation;
    private readonly IntTriple _outputPadding;
    private readonly int _groups;

    public ConvTranspose3d(
        int inputChannels,
        int outputChannels,
        IntTriple kernelSize,
        IntTriple? stride = null,
        IntTriple? padding = null,
        IntTriple? dilation = null,
        int groups = 1,
        IntTriple? outputPadding = null,
        bool bias = true)
        : base(
            TensorFactory.Uniform(
                -ComputeScale(inputChannels * kernelSize.First * kernelSize.Second * kernelSize.Third),
                ComputeScale(inputChannels * kernelSize.First * kernelSize.Second * kernelSize.Third),
                [outputChannels, kernelSize.First, kernelSize.Second, kernelSize.Third, inputChannels]),
            bias)
    {
        if (groups <= 0)
            throw new ArgumentOutOfRangeException(nameof(groups));

        this._stride = stride ?? new IntTriple(1, 1, 1);
        this._padding = padding ?? new IntTriple(0, 0, 0);
        this._dilation = dilation ?? new IntTriple(1, 1, 1);
        this._outputPadding = outputPadding ?? new IntTriple(0, 0, 0);
        this._groups = groups;
    }

    public override MlxArrayHandle Forward(MlxArrayHandle input)
    {
        var status = MlxOps.ConvTranspose3d(
            out var result,
            input,
            this.Weight.Value,
            this._stride.First,
            this._stride.Second,
            this._stride.Third,
            this._padding.First,
            this._padding.Second,
            this._padding.Third,
            this._dilation.First,
            this._dilation.Second,
            this._dilation.Third,
            this._outputPadding.First,
            this._outputPadding.Second,
            this._outputPadding.Third,
            this._groups,
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
