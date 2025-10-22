// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Collections.Generic;
using Itexoft.Mlx;

namespace Itexoft.Mlx.Nn;

/// <summary>
/// Simple Elman recurrent layer.
/// </summary>
public sealed class Rnn : Module
{
    private readonly ModuleParameter wxh;
    private readonly ModuleParameter whh;
    private readonly ModuleParameter? bias;
    private readonly Func<MlxArrayHandle, MlxArrayHandle> nonLinearity;

    public Rnn(int inputSize, int hiddenSize, bool bias = true, Func<MlxArrayHandle, MlxArrayHandle>? nonLinearity = null)
    {
        this.nonLinearity = nonLinearity ?? (x => x.Tanh());

        var scale = 1f / MathF.Sqrt(hiddenSize);
        this.wxh = this.RegisterParameter("wxh", TensorFactory.Uniform(-scale, scale, [hiddenSize, inputSize]));
        this.whh = this.RegisterParameter("whh", TensorFactory.Uniform(-scale, scale, [hiddenSize, hiddenSize]));
        if (bias)
            this.bias = this.RegisterParameter("bias", TensorFactory.Uniform(-scale, scale, [hiddenSize]));
    }

    public MlxArrayHandle Forward(MlxArrayHandle input, MlxArrayHandle? hidden = null)
    {
        var wxhT = this.wxh.Value.Transpose();
        var projected = input.Matmul(wxhT);
        MlxArray.Free(wxhT);

        projected = RecurrentHelpers.AddBiasIfNeeded(projected, this.bias);

        var whhT = this.whh.Value.Transpose();

        var timeAxis = projected.Rank() - 2;
        var steps = projected.Dim(timeAxis);

        var outputs = new List<MlxArrayHandle>(steps);
        var hasHidden = hidden.HasValue;
        var previousHidden = hidden;

        for (var index = 0; index < steps; index++)
        {
            var xStep = RecurrentHelpers.ExtractTimeStep(projected, timeAxis, index);

            if (hasHidden)
            {
                var recurrent = previousHidden!.Value.Matmul(whhT);
                var sum = xStep.Add(recurrent);
                MlxArray.Free(recurrent);
                MlxArray.Free(xStep);
                xStep = sum;
            }

            var activated = this.nonLinearity(xStep);
            MlxArray.Free(xStep);

            outputs.Add(activated);
            previousHidden = activated;
            hasHidden = true;
        }

        MlxArray.Free(projected);
        MlxArray.Free(whhT);

        var stacked = outputs.Stack(timeAxis);
        foreach (var h in outputs)
            MlxArray.Free(h);

        return stacked;
    }
}

/// <summary>
/// Gated recurrent unit layer.
/// </summary>
public sealed class Gru : Module
{
    private readonly int _hiddenSize;
    private readonly ModuleParameter _wx;
    private readonly ModuleParameter _wh;
    private readonly ModuleParameter? _bias;
    private readonly ModuleParameter? _bhn;

    public Gru(int inputSize, int hiddenSize, bool bias = true)
    {
        this._hiddenSize = hiddenSize;
        var scale = 1f / MathF.Sqrt(hiddenSize);
        this._wx = this.RegisterParameter("wx", TensorFactory.Uniform(-scale, scale, [3 * hiddenSize, inputSize]));
        this._wh = this.RegisterParameter("wh", TensorFactory.Uniform(-scale, scale, [3 * hiddenSize, hiddenSize]));
        if (bias)
        {
            this._bias = this.RegisterParameter("bias", TensorFactory.Uniform(-scale, scale, [3 * hiddenSize]));
            this._bhn = this.RegisterParameter("bias_hidden", TensorFactory.Uniform(-scale, scale, [hiddenSize]));
        }
    }

    public MlxArrayHandle Forward(MlxArrayHandle input, MlxArrayHandle? hidden = null)
    {
        var wxT = this._wx.Value.Transpose();
        var projected = input.Matmul(wxT);
        MlxArray.Free(wxT);

        projected = RecurrentHelpers.AddBiasIfNeeded(projected, this._bias);

        var lastAxis = projected.Rank() - 1;
        var total = projected.Dim(lastAxis);
        var xRz = projected.Slice(lastAxis, 0, total - this._hiddenSize);
        var xN = projected.Slice(lastAxis, total - this._hiddenSize, total);
        MlxArray.Free(projected);

        var whT = this._wh.Value.Transpose();

        var timeAxis = xRz.Rank() - 2;
        var steps = xRz.Dim(timeAxis);

        var outputs = new List<MlxArrayHandle>(steps);
        var hasHidden = hidden.HasValue;
        var currentHidden = hidden;

        for (var index = 0; index < steps; index++)
        {
            var rzInput = RecurrentHelpers.ExtractTimeStep(xRz, timeAxis, index);
            var nInput = RecurrentHelpers.ExtractTimeStep(xN, timeAxis, index);

            MlxArrayHandle? projN = null;

            if (hasHidden)
            {
                var hProj = currentHidden!.Value.Matmul(whT);
                var hProjRz = hProj.Slice(hProj.Rank() - 1, 0, 2 * this._hiddenSize);
                projN = hProj.Slice(hProj.Rank() - 1, 2 * this._hiddenSize, 3 * this._hiddenSize);
                MlxArray.Free(hProj);

                var rzSum = rzInput.Add(hProjRz);
                MlxArray.Free(rzInput);
                MlxArray.Free(hProjRz);
                rzInput = rzSum;

                if (this._bhn is not null)
                {
                    var broadcast = RecurrentHelpers.BroadcastBias(this._bhn.Value, projN.Value.Shape());
                    var withBias = projN.Value.Add(broadcast);
                    MlxArray.Free(projN.Value);
                    MlxArray.Free(broadcast);
                    projN = withBias;
                }
            }

            var rzActivated = rzInput.Sigmoid();
            MlxArray.Free(rzInput);

            var gates = rzActivated.Split(2, -1);
            var r = gates[0];
            var z = gates[1];
            MlxArray.Free(rzActivated);

            if (projN.HasValue)
            {
                var rHidden = r.Multiply(projN.Value);
                MlxArray.Free(projN.Value);
                var nSum = nInput.Add(rHidden);
                MlxArray.Free(nInput);
                MlxArray.Free(rHidden);
                nInput = nSum;
            }

            var nActivated = nInput.Tanh();
            MlxArray.Free(nInput);

            MlxArrayHandle newHidden;
            if (hasHidden)
            {
                var one = TensorFactory.ScalarLike(z, 1f);
                var oneMinusZ = one.Subtract(z);
                MlxArray.Free(one);

                var term1 = oneMinusZ.Multiply(nActivated);
                var term2 = z.Multiply(currentHidden!.Value);
                newHidden = term1.Add(term2);
                MlxArray.Free(oneMinusZ);
                MlxArray.Free(term1);
                MlxArray.Free(term2);
            }
            else
            {
                var one = TensorFactory.ScalarLike(z, 1f);
                var oneMinusZ = one.Subtract(z);
                MlxArray.Free(one);
                newHidden = oneMinusZ.Multiply(nActivated);
                MlxArray.Free(oneMinusZ);
            }

            MlxArray.Free(nActivated);

            outputs.Add(newHidden);
            currentHidden = newHidden;
            hasHidden = true;

            MlxArray.Free(r);
            MlxArray.Free(z);
        }

        MlxArray.Free(xRz);
        MlxArray.Free(xN);
        MlxArray.Free(whT);

        var stacked = outputs.Stack(timeAxis);
        foreach (var h in outputs)
            MlxArray.Free(h);

        return stacked;
    }
}

/// <summary>
/// Long short-term memory layer.
/// </summary>
public sealed class Lstm : Module
{
    private readonly ModuleParameter _wx;
    private readonly ModuleParameter _wh;
    private readonly ModuleParameter? _bias;

    public Lstm(int inputSize, int hiddenSize, bool bias = true)
    {
        var scale = 1f / MathF.Sqrt(hiddenSize);
        this._wx = this.RegisterParameter("wx", TensorFactory.Uniform(-scale, scale, [4 * hiddenSize, inputSize]));
        this._wh = this.RegisterParameter("wh", TensorFactory.Uniform(-scale, scale, [4 * hiddenSize, hiddenSize]));
        if (bias)
            this._bias = this.RegisterParameter("bias", TensorFactory.Uniform(-scale, scale, [4 * hiddenSize]));
    }

    public (MlxArrayHandle Hidden, MlxArrayHandle Cell) Forward(
        MlxArrayHandle input,
        MlxArrayHandle? hidden = null,
        MlxArrayHandle? cell = null)
    {
        var wxT = this._wx.Value.Transpose();
        var projected = input.Matmul(wxT);
        MlxArray.Free(wxT);

        projected = RecurrentHelpers.AddBiasIfNeeded(projected, this._bias);

        var whT = this._wh.Value.Transpose();

        var timeAxis = projected.Rank() - 2;
        var steps = projected.Dim(timeAxis);

        var hiddenStates = new List<MlxArrayHandle>(steps);
        var cellStates = new List<MlxArrayHandle>(steps);

        var hasHidden = hidden.HasValue;
        var currentHidden = hidden;
        var currentCell = cell;

        for (var index = 0; index < steps; index++)
        {
            var ifgo = RecurrentHelpers.ExtractTimeStep(projected, timeAxis, index);
            if (hasHidden)
            {
                var mat = currentHidden!.Value.Matmul(whT);
                var sum = ifgo.Add(mat);
                MlxArray.Free(mat);
                MlxArray.Free(ifgo);
                ifgo = sum;
            }

            var gates = ifgo.Split(4, -1);
            MlxArray.Free(ifgo);

            var i = gates[0].Sigmoid();
            var f = gates[1].Sigmoid();
            var g = gates[2].Tanh();
            var o = gates[3].Sigmoid();

            MlxArray.Free(gates[0]);
            MlxArray.Free(gates[1]);
            MlxArray.Free(gates[2]);
            MlxArray.Free(gates[3]);

            MlxArrayHandle newCell;
            if (currentCell.HasValue)
            {
                var forget = f.Multiply(currentCell.Value);
                var inputGate = i.Multiply(g);
                newCell = forget.Add(inputGate);
                MlxArray.Free(forget);
                MlxArray.Free(inputGate);
            }
            else
            {
                newCell = i.Multiply(g);
            }

            var tanhCell = newCell.Tanh();
            var newHidden = o.Multiply(tanhCell);

            MlxArray.Free(tanhCell);
            MlxArray.Free(i);
            MlxArray.Free(f);
            MlxArray.Free(g);
            MlxArray.Free(o);

            hiddenStates.Add(newHidden);
            cellStates.Add(newCell);

            currentHidden = newHidden;
            currentCell = newCell;
            hasHidden = true;
        }

        MlxArray.Free(projected);
        MlxArray.Free(whT);

        var stackedHidden = hiddenStates.Stack(timeAxis);
        var stackedCell = cellStates.Stack(timeAxis);

        foreach (var h in hiddenStates)
            MlxArray.Free(h);
        foreach (var c in cellStates)
            MlxArray.Free(c);

        return (stackedHidden, stackedCell);
    }
}

internal static class RecurrentHelpers
{
    internal static MlxArrayHandle AddBiasIfNeeded(MlxArrayHandle array, ModuleParameter? bias)
    {
        if (bias is null)
            return array;

        var targetShape = array.Shape();
        var broadcast = BroadcastBias(bias.Value, targetShape);
        var result = array.Add(broadcast);
        MlxArray.Free(array);
        MlxArray.Free(broadcast);

        return result;
    }

    internal static MlxArrayHandle BroadcastBias(MlxArrayHandle bias, int[] targetShape)
    {
        var temporaries = new List<MlxArrayHandle>();
        var handle = bias;
        for (var i = 0; i < targetShape.Length - 1; i++)
        {
            var expanded = handle.ExpandedDimension(0);
            temporaries.Add(expanded);
            handle = expanded;
        }

        var broadcast = handle.BroadcastTo(targetShape);
        foreach (var temp in temporaries)
            MlxArray.Free(temp);

        return broadcast;
    }

    internal static MlxArrayHandle ExtractTimeStep(MlxArrayHandle array, int axis, int index)
    {
        var slice = array.Slice(axis, index, index + 1);
        var squeezed = slice.Squeezed(axis);
        MlxArray.Free(slice);

        return squeezed;
    }
}