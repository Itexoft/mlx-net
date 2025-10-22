// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;

namespace Itexoft.Mlx.Nn;

/// <summary>
/// Specifies how element-wise losses should be reduced.
/// </summary>
public enum LossReduction
{
    None,
    Mean,
    Sum
}

internal static class LossReductionExtensions
{
    public static MlxArrayHandle Reduce(this LossReduction reduction, MlxArrayHandle loss)
    {
        return reduction switch
        {
            LossReduction.None => loss,
            LossReduction.Mean => ReduceMean(loss),
            LossReduction.Sum => ReduceSum(loss),
            _ => loss
        };
    }

    private static MlxArrayHandle ReduceMean(MlxArrayHandle loss)
    {
        var rank = loss.Rank();

        if (rank == 0)
            return loss;

        var axes = AllAxes(rank);
        var reduced = loss.Mean(axes, false);
        MlxArray.Free(loss);

        return reduced;
    }

    private static MlxArrayHandle ReduceSum(MlxArrayHandle loss)
    {
        var rank = loss.Rank();

        if (rank == 0)
            return loss;

        var axes = AllAxes(rank);
        var reduced = loss.Sum(axes, false);
        MlxArray.Free(loss);

        return reduced;
    }

    private static int[] AllAxes(int rank)
    {
        var result = new int[rank];
        for (var i = 0; i < rank; i++)
            result[i] = i;

        return result;
    }
}

/// <summary>
/// Collection of high-level loss functions mirroring <c>mlx.nn.losses</c>.
/// </summary>
public static class Losses
{
    public static MlxArrayHandle CrossEntropy(
        MlxArrayHandle logits,
        MlxArrayHandle targets,
        MlxArrayHandle weights = default,
        int axis = -1,
        float labelSmoothing = 0f,
        LossReduction reduction = LossReduction.None)
    {
        if (labelSmoothing < 0f || labelSmoothing >= 1f)
            throw new ArgumentOutOfRangeException(nameof(labelSmoothing), "labelSmoothing must be in [0, 1).");

        var logitsRank = logits.Rank();
        var axisNorm = TensorExtensions.NormalizeAxis(axis, logitsRank);
        var targetsAsProbabilities = targets.Rank() == logitsRank;

        MlxArrayHandle score;
        if (targetsAsProbabilities)
        {
            var product = logits.Multiply(targets);
            score = product.Sum(axisNorm);
            MlxArray.Free(product);
        }
        else
        {
            var expandedTargets = targets.ExpandedDimension(-1);
            var gathered = logits.TakeAlong(expandedTargets, axisNorm);
            score = gathered.Squeezed(-1);
            MlxArray.Free(gathered);
            MlxArray.Free(expandedTargets);
        }

        var logSumExp = logits.LogSumExp(axisNorm);
        MlxArrayHandle loss;

        if (labelSmoothing > 0f)
        {
            var adjustedScore = score.MultiplyScalar(1f - labelSmoothing);
            MlxArray.Free(score);

            var meanLogits = logits.Mean(axisNorm);
            var smoothedLoss = meanLogits.MultiplyScalar(-labelSmoothing);
            MlxArray.Free(meanLogits);

            var temp = logSumExp.Subtract(adjustedScore);
            MlxArray.Free(adjustedScore);
            var combined = temp.Add(smoothedLoss);
            MlxArray.Free(temp);
            MlxArray.Free(smoothedLoss);
            loss = combined;
        }
        else
        {
            var temp = logSumExp.Subtract(score);
            MlxArray.Free(score);
            loss = temp;
        }

        MlxArray.Free(logSumExp);

        if (!TensorUtilities.IsNull(weights))
        {
            var weighted = loss.Multiply(weights);
            MlxArray.Free(loss);
            loss = weighted;
        }

        return reduction.Reduce(loss);
    }

    public static MlxArrayHandle BinaryCrossEntropy(
        MlxArrayHandle logits,
        MlxArrayHandle targets,
        MlxArrayHandle weights = default,
        bool withLogits = true,
        LossReduction reduction = LossReduction.Mean)
    {
        var dtype = MlxArray.DType(logits);
        MlxArrayHandle loss;

        if (withLogits)
        {
            var zero = TensorFactory.Scalar(0f, dtype);
            var logAdd = logits.LogAddExp(zero);
            MlxArray.Free(zero);

            var targetsMul = targets.Multiply(logits);
            var difference = logAdd.Subtract(targetsMul);
            MlxArray.Free(logAdd);
            MlxArray.Free(targetsMul);
            loss = difference;
        }
        else
        {
            var logInputs = logits.Log();
            var logInputsClip = logInputs.Clip(min: -100f);
            MlxArray.Free(logInputs);

            var shape = logits.Shape();
            var ones = TensorFactory.Ones(shape, dtype);
            var oneMinusLogits = ones.Subtract(logits);
            MlxArray.Free(ones);

            var logOneMinus = oneMinusLogits.Log();
            var logOneMinusClip = logOneMinus.Clip(min: -100f);
            MlxArray.Free(oneMinusLogits);
            MlxArray.Free(logOneMinus);

            var term1 = targets.Multiply(logInputsClip);
            var oneScalar = TensorFactory.Scalar(1f, dtype);
            var oneMinusTargets = oneScalar.Subtract(targets);
            MlxArray.Free(oneScalar);

            var term2 = oneMinusTargets.Multiply(logOneMinusClip);
            MlxArray.Free(oneMinusTargets);

            var sumTerms = term1.Add(term2);
            MlxArray.Free(term1);
            MlxArray.Free(term2);

            var negative = sumTerms.MultiplyScalar(-1f);
            MlxArray.Free(sumTerms);
            loss = negative;

            MlxArray.Free(logInputsClip);
            MlxArray.Free(logOneMinusClip);
        }

        if (!TensorUtilities.IsNull(weights))
        {
            var weighted = loss.Multiply(weights);
            MlxArray.Free(loss);
            loss = weighted;
        }

        return reduction.Reduce(loss);
    }

    public static MlxArrayHandle L1Loss(
        MlxArrayHandle predictions,
        MlxArrayHandle targets,
        LossReduction reduction = LossReduction.Mean)
    {
        var diff = predictions.Subtract(targets);
        var abs = diff.Abs();
        MlxArray.Free(diff);

        return reduction.Reduce(abs);
    }

    public static MlxArrayHandle MseLoss(
        MlxArrayHandle predictions,
        MlxArrayHandle targets,
        LossReduction reduction = LossReduction.Mean)
    {
        var diff = predictions.Subtract(targets);
        var sq = diff.Square();
        MlxArray.Free(diff);

        return reduction.Reduce(sq);
    }

    public static MlxArrayHandle NllLoss(
        MlxArrayHandle inputs,
        MlxArrayHandle targets,
        int axis = -1,
        LossReduction reduction = LossReduction.None)
    {
        var axisNorm = TensorExtensions.NormalizeAxis(axis, inputs.Rank());
        var expandedTargets = targets.ExpandedDimension(-1);
        var gathered = inputs.TakeAlong(expandedTargets, axisNorm);
        var squeezed = gathered.Squeezed(-1);
        MlxArray.Free(gathered);
        MlxArray.Free(expandedTargets);

        var neg = squeezed.MultiplyScalar(-1f);
        MlxArray.Free(squeezed);

        return reduction.Reduce(neg);
    }

    public static MlxArrayHandle KlDivLoss(
        MlxArrayHandle inputs,
        MlxArrayHandle targets,
        int axis = -1,
        LossReduction reduction = LossReduction.None)
    {
        var axisNorm = TensorExtensions.NormalizeAxis(axis, inputs.Rank());
        var expTargets = targets.Exp();
        var diff = targets.Subtract(inputs);
        var product = expTargets.Multiply(diff);
        MlxArray.Free(expTargets);
        MlxArray.Free(diff);

        var summed = product.Sum(axisNorm);
        MlxArray.Free(product);

        return reduction.Reduce(summed);
    }

    public static MlxArrayHandle SmoothL1Loss(
        MlxArrayHandle predictions,
        MlxArrayHandle targets,
        float beta = 1f,
        LossReduction reduction = LossReduction.Mean)
    {
        var diff = predictions.Subtract(targets);
        var absDiff = diff.Abs();
        MlxArray.Free(diff);

        var betaScalar = TensorFactory.ScalarLike(absDiff, beta);
        var mask = absDiff.LessThan(betaScalar);

        var squared = absDiff.Square();
        var numerator = squared.MultiplyScalar(0.5f);
        var quadratic = numerator.Divide(betaScalar);
        MlxArray.Free(numerator);

        var betaHalf = TensorFactory.ScalarLike(absDiff, beta * 0.5f);
        var linear = absDiff.Subtract(betaHalf);
        MlxArray.Free(betaHalf);

        var loss = mask.Where(quadratic, linear);

        MlxArray.Free(mask);
        MlxArray.Free(quadratic);
        MlxArray.Free(linear);
        MlxArray.Free(squared);
        MlxArray.Free(absDiff);
        MlxArray.Free(betaScalar);

        return reduction.Reduce(loss);
    }

    public static MlxArrayHandle TripletLoss(
        MlxArrayHandle anchors,
        MlxArrayHandle positives,
        MlxArrayHandle negatives,
        int axis = -1,
        int p = 2,
        float margin = 1f,
        float eps = 1e-6f,
        LossReduction reduction = LossReduction.None)
    {
        var axisNorm = TensorExtensions.NormalizeAxis(axis, anchors.Rank());

        var diffPos = anchors.Subtract(positives);
        var diffNeg = anchors.Subtract(negatives);

        var powPos = diffPos.Pow(p);
        var powNeg = diffNeg.Pow(p);

        var sumPos = powPos.Sum(axisNorm);
        var sumNeg = powNeg.Sum(axisNorm);

        MlxArray.Free(diffPos);
        MlxArray.Free(diffNeg);
        MlxArray.Free(powPos);
        MlxArray.Free(powNeg);

        var epsPos = TensorFactory.ScalarLike(sumPos, eps);
        var posWithEps = sumPos.Add(epsPos);
        MlxArray.Free(epsPos);
        MlxArray.Free(sumPos);
        var posNorm = posWithEps.Sqrt();
        MlxArray.Free(posWithEps);

        var epsNeg = TensorFactory.ScalarLike(sumNeg, eps);
        var negWithEps = sumNeg.Add(epsNeg);
        MlxArray.Free(epsNeg);
        MlxArray.Free(sumNeg);
        var negNorm = negWithEps.Sqrt();
        MlxArray.Free(negWithEps);

        var diffNorm = posNorm.Subtract(negNorm);
        MlxArray.Free(posNorm);
        MlxArray.Free(negNorm);

        var marginScalar = TensorFactory.ScalarLike(diffNorm, margin);
        var added = diffNorm.Add(marginScalar);
        MlxArray.Free(diffNorm);
        MlxArray.Free(marginScalar);

        var zeroScalar = TensorFactory.ScalarLike(added, 0f);
        var lossRaw = added.Maximum(zeroScalar);
        MlxArray.Free(added);
        MlxArray.Free(zeroScalar);

        return reduction.Reduce(lossRaw);
    }

    public static MlxArrayHandle HingeLoss(
        MlxArrayHandle inputs,
        MlxArrayHandle targets,
        LossReduction reduction = LossReduction.None)
    {
        var product = inputs.Multiply(targets);
        var one = TensorFactory.ScalarLike(product, 1f);
        var diff = one.Subtract(product);
        MlxArray.Free(one);
        MlxArray.Free(product);

        var zero = TensorFactory.ScalarLike(diff, 0f);
        var loss = diff.Maximum(zero);
        MlxArray.Free(diff);
        MlxArray.Free(zero);

        return reduction.Reduce(loss);
    }

    public static MlxArrayHandle HuberLoss(
        MlxArrayHandle inputs,
        MlxArrayHandle targets,
        float delta = 1f,
        LossReduction reduction = LossReduction.None)
    {
        var errors = inputs.Subtract(targets);
        var absErrors = errors.Abs();
        MlxArray.Free(errors);

        var deltaScalar = TensorFactory.ScalarLike(absErrors, delta);
        var quadratic = absErrors.Minimum(deltaScalar);
        var linear = absErrors.Subtract(quadratic);

        var quadraticSquared = quadratic.Square();
        var quadraticTerm = quadraticSquared.MultiplyScalar(0.5f);
        MlxArray.Free(quadraticSquared);

        var deltaLinear = linear.Multiply(deltaScalar);
        var lossRaw = quadraticTerm.Add(deltaLinear);

        MlxArray.Free(quadratic);
        MlxArray.Free(linear);
        MlxArray.Free(deltaScalar);
        MlxArray.Free(quadraticTerm);
        MlxArray.Free(deltaLinear);
        MlxArray.Free(absErrors);

        return reduction.Reduce(lossRaw);
    }

    public static MlxArrayHandle LogCoshLoss(
        MlxArrayHandle inputs,
        MlxArrayHandle targets,
        LossReduction reduction = LossReduction.None)
    {
        var errors = inputs.Subtract(targets);
        var negativeErrors = errors.Negative();
        var logAdd = errors.LogAddExp(negativeErrors);
        MlxArray.Free(errors);
        MlxArray.Free(negativeErrors);

        var logTwo = TensorFactory.ScalarLike(logAdd, MathF.Log(2f));
        var lossRaw = logAdd.Subtract(logTwo);
        MlxArray.Free(logAdd);
        MlxArray.Free(logTwo);

        return reduction.Reduce(lossRaw);
    }

    public static MlxArrayHandle CosineSimilarityLoss(
        MlxArrayHandle x1,
        MlxArrayHandle x2,
        int axis = 1,
        float eps = 1e-8f,
        LossReduction reduction = LossReduction.None)
    {
        var axisNorm = TensorExtensions.NormalizeAxis(axis, x1.Rank());

        var product = x1.Multiply(x2);
        var numerator = product.Sum(axisNorm);
        MlxArray.Free(product);

        var x1Norm = L2Norm(x1, axisNorm);
        var x2Norm = L2Norm(x2, axisNorm);

        var denominator = x1Norm.Multiply(x2Norm);
        MlxArray.Free(x1Norm);
        MlxArray.Free(x2Norm);

        var epsScalar = TensorFactory.ScalarLike(denominator, eps);
        var denomClamped = denominator.Maximum(epsScalar);
        MlxArray.Free(denominator);
        MlxArray.Free(epsScalar);

        var fraction = numerator.Divide(denomClamped);
        MlxArray.Free(numerator);
        MlxArray.Free(denomClamped);

        return reduction.Reduce(fraction);
    }

    private static MlxArrayHandle L2Norm(MlxArrayHandle array, int axis)
    {
        if (MlxArray.DType(array) == MlxDType.MLX_COMPLEX64)
        {
            var abs = array.Abs();
            var squared = abs.Multiply(abs);
            MlxArray.Free(abs);
            var summed = squared.Sum(axis);
            MlxArray.Free(squared);
            var result = summed.Sqrt();
            MlxArray.Free(summed);

            return result;
        }
        else
        {
            var squared = array.Square();
            var summed = squared.Sum(axis);
            MlxArray.Free(squared);
            var result = summed.Sqrt();
            MlxArray.Free(summed);

            return result;
        }
    }
}