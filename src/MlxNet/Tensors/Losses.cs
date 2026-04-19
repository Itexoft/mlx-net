// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using Itexoft.Mlx;
using Itexoft.Tensors.Internal;

namespace Itexoft.Tensors;

public enum LossReduction
{
    None,
    Mean,
    Sum,
}

public readonly struct Losses
{
    public static Tensor CrossEntropy(
        Tensor logits,
        Tensor targets,
        int axis = -1,
        float labelSmoothing = 0f,
        LossReduction reduction = LossReduction.None) =>
        CrossEntropyCore(logits, targets, false, default, axis, labelSmoothing, reduction);

    public static Tensor CrossEntropy(
        Tensor logits,
        Tensor targets,
        Tensor weights,
        int axis = -1,
        float labelSmoothing = 0f,
        LossReduction reduction = LossReduction.None) =>
        CrossEntropyCore(logits, targets, true, weights, axis, labelSmoothing, reduction);

    public static Tensor BinaryCrossEntropy(Tensor logits, Tensor targets, bool withLogits = true, LossReduction reduction = LossReduction.Mean) =>
        BinaryCrossEntropyCore(logits, targets, false, default, withLogits, reduction);

    public static Tensor BinaryCrossEntropy(
        Tensor logits,
        Tensor targets,
        Tensor weights,
        bool withLogits = true,
        LossReduction reduction = LossReduction.Mean) =>
        BinaryCrossEntropyCore(logits, targets, true, weights, withLogits, reduction);

    public static Tensor L1Loss(Tensor predictions, Tensor targets, LossReduction reduction = LossReduction.Mean) =>
        ApplyReduction((predictions - targets).Abs, reduction);

    public static Tensor MseLoss(Tensor predictions, Tensor targets, LossReduction reduction = LossReduction.Mean)
    {
        var diff = predictions - targets;

        return ApplyReduction(diff * diff, reduction);
    }

    public static Tensor NllLoss(Tensor inputs, Tensor targets, int axis = -1, LossReduction reduction = LossReduction.None)
    {
        var axisNorm = NormalizeAxis(inputs, axis);
        var expandedTargets = targets.ExpandDims(axisNorm);
        var gathered = inputs.TakeAlong(expandedTargets, axisNorm);

        return ApplyReduction(-SqueezeAxis(gathered, axisNorm), reduction);
    }

    public static Tensor KlDivLoss(Tensor inputs, Tensor targets, int axis = -1, LossReduction reduction = LossReduction.None)
    {
        var axisNorm = NormalizeAxis(inputs, axis);
        var expTargets = targets.Exp;
        var diff = targets - inputs;
        var product = expTargets * diff;

        return ApplyReduction(product.Sum(new Index(axisNorm)), reduction);
    }

    public static Tensor SmoothL1Loss(Tensor predictions, Tensor targets, float beta = 1f, LossReduction reduction = LossReduction.Mean)
    {
        var absDiff = (predictions - targets).Abs;
        var betaTensor = FullLike(absDiff, beta);
        var mask = absDiff.Lt(betaTensor);
        var quadratic = absDiff * absDiff * 0.5f / beta;
        var linear = absDiff - beta * 0.5f;
        var loss = mask.Where(quadratic, linear);

        return ApplyReduction(loss, reduction);
    }

    public static Tensor TripletLoss(
        Tensor anchors,
        Tensor positives,
        Tensor negatives,
        int axis = -1,
        int p = 2,
        float margin = 1f,
        float eps = 1e-6f,
        LossReduction reduction = LossReduction.None)
    {
        var axisNorm = NormalizeAxis(anchors, axis);
        var posNorm = LpNorm(anchors - positives, axisNorm, p, eps);
        var negNorm = LpNorm(anchors - negatives, axisNorm, p, eps);
        var marginLoss = posNorm - negNorm + margin;
        var clipped = marginLoss.Max(FullLike(marginLoss, 0f));

        return ApplyReduction(clipped, reduction);
    }

    public static Tensor HingeLoss(Tensor inputs, Tensor targets, LossReduction reduction = LossReduction.None)
    {
        var loss = (1f - inputs * targets).Max(FullLike(inputs, 0f));

        return ApplyReduction(loss, reduction);
    }

    public static Tensor HuberLoss(Tensor inputs, Tensor targets, float delta = 1f, LossReduction reduction = LossReduction.None)
    {
        var absErrors = (inputs - targets).Abs;
        var deltaTensor = FullLike(absErrors, delta);
        var quadratic = absErrors.Min(deltaTensor);
        var linear = absErrors - quadratic;
        var loss = quadratic * quadratic * 0.5f + linear * deltaTensor;

        return ApplyReduction(loss, reduction);
    }

    public static Tensor LogCoshLoss(Tensor inputs, Tensor targets, LossReduction reduction = LossReduction.None)
    {
        var errors = inputs - targets;
        var loss = LogAddExp(errors, -errors) - MathF.Log(2f);

        return ApplyReduction(loss, reduction);
    }

    public static Tensor CosineSimilarityLoss(Tensor x1, Tensor x2, int axis = 1, float eps = 1e-8f, LossReduction reduction = LossReduction.None)
    {
        var axisNorm = NormalizeAxis(x1, axis);
        var numerator = (x1 * x2).Sum(new Index(axisNorm));
        var x1Norm = L2Norm(x1, axisNorm);
        var x2Norm = L2Norm(x2, axisNorm);
        var denominator = (x1Norm * x2Norm).Max(FullLike(x1Norm, eps));
        var fraction = numerator / denominator;

        return ApplyReduction(fraction, reduction);
    }

    private static Tensor CrossEntropyCore(
        Tensor logits,
        Tensor targets,
        bool hasWeights,
        Tensor weights,
        int axis,
        float labelSmoothing,
        LossReduction reduction)
    {
        if (labelSmoothing < 0f || labelSmoothing >= 1f)
            throw new ArgumentOutOfRangeException(nameof(labelSmoothing), "labelSmoothing must be in [0, 1).");

        var axisNorm = NormalizeAxis(logits, axis);
        var targetsAsProbabilities = targets.Rank == logits.Rank;
        Tensor score;

        if (targetsAsProbabilities)
            score = (logits * targets).Sum(new Index(axisNorm));
        else
        {
            var expandedTargets = targets.ExpandDims(axisNorm);
            var gathered = logits.TakeAlong(expandedTargets, axisNorm);
            score = SqueezeAxis(gathered, axisNorm);
        }

        var logSumExp = LogSumExp(logits, axisNorm);
        Tensor loss;

        if (labelSmoothing > 0f)
        {
            var adjustedScore = score * (1f - labelSmoothing);
            var smoothedLoss = logits.Mean(new Index(axisNorm)) * -labelSmoothing;
            loss = logSumExp - adjustedScore + smoothedLoss;
        }
        else
            loss = logSumExp - score;

        if (hasWeights)
            loss = loss * weights;

        return ApplyReduction(loss, reduction);
    }

    private static Tensor BinaryCrossEntropyCore(
        Tensor logits,
        Tensor targets,
        bool hasWeights,
        Tensor weights,
        bool withLogits,
        LossReduction reduction)
    {
        Tensor loss;

        if (withLogits)
        {
            var zero = FullLike(logits, 0f);
            var maxTerm = logits.Max(zero);
            var logTerm = ((-logits.Abs).Exp + 1f).Log;
            loss = maxTerm - logits * targets + logTerm;
        }
        else
        {
            const float minLog = -100f;
            var minProbability = MathF.Exp(minLog);
            var clippedInputs = Clip(logits, minProbability, 1f - minProbability);
            var oneMinusInputs = FullLike(logits, 1f) - clippedInputs;
            var lossPositive = targets * clippedInputs.Log;
            var lossNegative = (FullLike(targets, 1f) - targets) * oneMinusInputs.Log;
            loss = -(lossPositive + lossNegative);
        }

        if (hasWeights)
            loss = loss * weights;

        return ApplyReduction(loss, reduction);
    }

    private static Tensor ApplyReduction(Tensor loss, LossReduction reduction) => reduction switch
    {
        LossReduction.None => loss,
        LossReduction.Mean => loss.Mean(),
        LossReduction.Sum => loss.Sum(),
        _ => throw new ArgumentOutOfRangeException(nameof(reduction)),
    };

    private static int NormalizeAxis(Tensor tensor, int axis) => TensorRuntime.NormalizeAxis(axis, tensor.Rank);

    private static Tensor FullLike(Tensor reference, float value) => reference * 0f + value;

    private static Tensor SqueezeAxis(Tensor tensor, int axis)
    {
        Span<int> axes = stackalloc int[1] { axis };

        return tensor.Squeeze(axes);
    }

    private static Tensor Clip(Tensor tensor, float min, float max)
    {
        var lowerBound = FullLike(tensor, min);
        var upperBound = FullLike(tensor, max);

        return tensor.Max(lowerBound).Min(upperBound);
    }

    private static Tensor LogAddExp(Tensor left, Tensor right)
    {
        var max = left.Max(right);
        var sum = (left - max).Exp + (right - max).Exp;

        return max + sum.Log;
    }

    private static Tensor LogSumExp(Tensor tensor, int axis)
    {
        var maxKeep = tensor.Max(new Index(axis), true);
        var sumExp = (tensor - maxKeep).Exp.Sum(new Index(axis), true);
        var resultKeep = maxKeep + sumExp.Log;

        return SqueezeAxis(resultKeep, axis);
    }

    private static Tensor LpNorm(Tensor tensor, int axis, int p, float eps)
    {
        var powers = tensor.Pow(p);
        var summed = powers.Sum(new Index(axis));
        var stabilized = summed + eps;

        return stabilized.Sqrt;
    }

    private static Tensor L2Norm(Tensor tensor, int axis)
    {
        var values = tensor.DType == MlxDType.MlxComplex64 ? tensor.Abs : tensor;
        var squared = values * values;
        var summed = squared.Sum(new Index(axis));

        return summed.Sqrt;
    }
}
