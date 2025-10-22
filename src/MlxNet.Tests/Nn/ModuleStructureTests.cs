// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Collections.Generic;
using System.Linq;
using Itexoft.Mlx;
using Itexoft.Mlx.Nn;
using NUnit.Framework;

namespace Itexoft.Mlx.Nn.Tests;

[TestFixture]
public sealed class ModuleStructureTests
{
    [Test]
    public void Parameters_IncludeNestedModules()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var module = new CompositeModule();
        var parameters = module.Parameters();

        var keys = parameters.Select(kv => kv.Key).ToArray();

        Assert.That(parameters.Count, Is.EqualTo(4), "Unexpected parameter count.");
        Assert.That(keys, Is.EquivalentTo(new[] { "rootWeight", "left.weight", "right.weight", "frozen.weight" }));

        Assert.That(parameters["rootWeight"].Trainable, Is.True);
        Assert.That(parameters["left.weight"].Trainable, Is.True);
        Assert.That(parameters["right.weight"].Trainable, Is.True);
        Assert.That(parameters["frozen.weight"].Trainable, Is.False);
    }

    [Test]
    public void TrainableParameters_ExcludesFrozen()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var module = new CompositeModule();
        var trainable = module.TrainableParameters();

        Assert.That(trainable.Count, Is.EqualTo(3));
        Assert.That(trainable.Keys, Is.EquivalentTo(new[] { "rootWeight", "left.weight", "right.weight" }));
    }

    [Test]
    public void UpdateParameters_ReplacesNestedValue()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var module = new CompositeModule();

        var updates = new ParameterCollection();
        var replacement = TensorFactory.Full(7f, [2]);
        updates.AddOrUpdate("left.weight", new(replacement, true));

        module.UpdateParameters(updates, false);

        var refreshed = module.Parameters()["left.weight"].Value;
        TestHelpers.Ok(MlxArray.Eval(refreshed), "eval updated parameter");

        var data = TestHelpers.ToFloat32(refreshed);
        Assert.That(data, Is.EqualTo(new[] { 7f, 7f }).Within(1e-6));
    }

    [Test]
    public void Children_ExposeRegisteredModules()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var module = new CompositeModule();
        var children = module.Children;

        Assert.That(children.Count, Is.EqualTo(3));
        Assert.That(children.Keys, Is.EquivalentTo(new[] { "left", "right", "frozen" }));
        Assert.That(children.Values.All(child => child is TestLeaf), Is.True);
    }

    [Test]
    public void FlattenModules_IncludesSelfAndDescendants()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var module = new CompositeModule();
        var flattened = module.FlattenModules(includeSelf: true);

        Assert.That(flattened.Count, Is.EqualTo(4));
        Assert.That(flattened.ContainsValue(module), Is.True);
        Assert.That(flattened.Keys, Does.Contain("left"));
        Assert.That(flattened.Keys, Does.Contain("right"));
        Assert.That(flattened.Keys, Does.Contain("frozen"));
    }

    [Test]
    public void Sequential_RegistersLayerModules()
    {
        TestHelpers.RequireNativeOrIgnore();

        using var sequential = new Sequential(
            new Tanh(),
            new Sigmoid(),
            new Linear(4, 8),
            new Linear(8, 2));

        Assert.That(sequential.Layers.Count, Is.EqualTo(4));

        var parameters = sequential.Parameters();
        Assert.That(parameters.Count, Is.EqualTo(4));
        Assert.That(parameters.Keys, Does.Contain("2.weight"));
        Assert.That(parameters.Keys, Does.Contain("2.bias"));
        Assert.That(parameters.Keys, Does.Contain("3.weight"));
        Assert.That(parameters.Keys, Does.Contain("3.bias"));
    }

    private sealed class TestLeaf : Module
    {
        public TestLeaf(bool trainable = true) => this.Weight = this.RegisterParameter("weight", TensorFactory.Zeros([2]), trainable);

        public ModuleParameter Weight { get; }
    }

    private sealed class CompositeModule : Module
    {
        public CompositeModule()
        {
            this.Root = this.RegisterParameter("rootWeight", TensorFactory.Zeros([2]));
            this.Left = this.RegisterModule("left", new TestLeaf());
            this.Right = this.RegisterModule("right", new TestLeaf());
            this.Frozen = this.RegisterModule("frozen", new TestLeaf(trainable: false));
        }

        public ModuleParameter Root { get; }

        public TestLeaf Left { get; }

        public TestLeaf Right { get; }

        public TestLeaf Frozen { get; }
    }
}