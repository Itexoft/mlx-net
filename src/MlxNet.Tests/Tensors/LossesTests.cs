// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using Itexoft.Tensors;
using NUnit.Framework;

namespace MlxNet.Tests.Tensors;

[TestFixture]
public sealed class LossesTests
{
    [Test]
    public void CrossEntropy_WithClassTargets_Works()
    {
        TestHelpers.RequireNativeOrIgnore();

        var logits = Tensor.Zeros((2, 3));
        var targets = Tensor.From([0, 2], 2);
        var loss = Losses.CrossEntropy(logits, targets, reduction: LossReduction.None);
        var mean = Losses.CrossEntropy(logits, targets, reduction: LossReduction.Mean);
        var expected = MathF.Log(3f);

        Assert.That(ReadFlatFloat(loss), Is.EqualTo(new[] { expected, expected }).Within(1e-5f));
        Assert.That((float)mean, Is.EqualTo(expected).Within(1e-5f));
    }

    [Test]
    public void CrossEntropy_WithProbabilityTargets_Works()
    {
        TestHelpers.RequireNativeOrIgnore();

        var logits = Tensor.Zeros((2, 3));
        var targets = Tensor.From([1f, 0f, 0f, 0f, 0f, 1f], (2, 3));
        var sum = Losses.CrossEntropy(logits, targets, reduction: LossReduction.Sum);
        var expected = 2f * MathF.Log(3f);

        Assert.That((float)sum, Is.EqualTo(expected).Within(1e-5f));
    }

    [Test]
    public void MseLoss_And_BinaryCrossEntropy_Work()
    {
        TestHelpers.RequireNativeOrIgnore();

        var predictions = Tensor.From([1f, 2f, 3f], 3);
        var targets = Tensor.From([0f, 2f, 4f], 3);
        var mse = Losses.MseLoss(predictions, targets);
        var logits = Tensor.From([0f, 0f], 2);
        var binaryTargets = Tensor.From([0f, 1f], 2);
        var bce = Losses.BinaryCrossEntropy(logits, binaryTargets, true, LossReduction.Mean);

        Assert.That((float)mse, Is.EqualTo(2f / 3f).Within(1e-6f));
        Assert.That((float)bce, Is.EqualTo(MathF.Log(2f)).Within(1e-6f));
    }

    [Test]
    public void WeightedCrossEntropy_And_ProbabilityBinaryCrossEntropy_Work()
    {
        TestHelpers.RequireNativeOrIgnore();

        var logits = Tensor.Zeros((2, 3));
        var targets = Tensor.From([0, 2], 2);
        var weights = Tensor.From([2f, 0.5f], 2);
        var weighted = Losses.CrossEntropy(logits, targets, weights, reduction: LossReduction.Sum);
        var probabilities = Tensor.From([0.5f, 0.5f], 2);
        var binaryTargets = Tensor.From([0f, 1f], 2);
        var binary = Losses.BinaryCrossEntropy(probabilities, binaryTargets, false, LossReduction.Mean);
        var expected = 2.5f * MathF.Log(3f);

        Assert.That((float)weighted, Is.EqualTo(expected).Within(1e-5f));
        Assert.That((float)binary, Is.EqualTo(MathF.Log(2f)).Within(1e-6f));
    }

    [Test]
    public void PointwiseFunctionalLosses_AreZeroOnEqualInputs()
    {
        TestHelpers.RequireNativeOrIgnore();

        var values = Tensor.From([1f, 2f, 3f], 3);
        var l1 = Losses.L1Loss(values, values, LossReduction.Mean);
        var smoothL1 = Losses.SmoothL1Loss(values, values, reduction: LossReduction.Mean);
        var huber = Losses.HuberLoss(values, values, reduction: LossReduction.Mean);
        var logCosh = Losses.LogCoshLoss(values, values, LossReduction.Mean);

        Assert.That((float)l1, Is.EqualTo(0f).Within(1e-6f));
        Assert.That((float)smoothL1, Is.EqualTo(0f).Within(1e-6f));
        Assert.That((float)huber, Is.EqualTo(0f).Within(1e-6f));
        Assert.That((float)logCosh, Is.EqualTo(0f).Within(1e-6f));
    }

    [Test]
    public void Nll_KlDiv_Triplet_Hinge_And_CosineSimilarity_Work()
    {
        TestHelpers.RequireNativeOrIgnore();

        var logProbabilitiesData = new[]
        {
            MathF.Log(0.25f), MathF.Log(0.75f),
            MathF.Log(0.80f), MathF.Log(0.20f),
        };

        var logProbabilities = Tensor.From(logProbabilitiesData, (2, 2));
        var classTargets = Tensor.From([1, 0], 2);
        var nll = Losses.NllLoss(logProbabilities, classTargets, reduction: LossReduction.None);
        var kl = Losses.KlDivLoss(logProbabilities, logProbabilities, reduction: LossReduction.Sum);
        var anchors = Tensor.From([0f, 0f, 0f, 0f], (2, 2));
        var positives = Tensor.From([0f, 0f, 0f, 0f], (2, 2));
        var negatives = Tensor.From([0f, 0f, 0f, 0f], (2, 2));
        var triplet = Losses.TripletLoss(anchors, positives, negatives, reduction: LossReduction.Mean);
        var hingeInputs = Tensor.From([0f, 0f], 2);
        var hingeTargets = Tensor.From([1f, -1f], 2);
        var hinge = Losses.HingeLoss(hingeInputs, hingeTargets, LossReduction.Mean);
        var x1 = Tensor.From([1f, 0f, 0f, 1f], (2, 2));
        var x2 = Tensor.From([1f, 0f, 0f, 1f], (2, 2));
        var cosine = Losses.CosineSimilarityLoss(x1, x2, 1, reduction: LossReduction.Mean);

        Assert.That(ReadFlatFloat(nll), Is.EqualTo(new[] { -MathF.Log(0.75f), -MathF.Log(0.80f) }).Within(1e-6f));
        Assert.That((float)kl, Is.EqualTo(0f).Within(1e-6f));
        Assert.That((float)triplet, Is.EqualTo(1f).Within(1e-6f));
        Assert.That((float)hinge, Is.EqualTo(1f).Within(1e-6f));
        Assert.That((float)cosine, Is.EqualTo(1f).Within(1e-6f));
    }

    private static float[] ReadFlatFloat(Tensor tensor)
    {
        tensor.Eval();
        var result = new float[tensor.Dim(0)];

        for (var i = 0; i < result.Length; i++)
            result[i] = (float)tensor[i];

        return result;
    }
}
