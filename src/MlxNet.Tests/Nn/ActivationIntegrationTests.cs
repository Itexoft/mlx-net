// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Itexoft.Mlx;
using Itexoft.Mlx.Nn;
using NUnit.Framework;

namespace Itexoft.Mlx.Nn.Tests;

[TestFixture]
public unsafe class ActivationIntegrationTests
{
    private static readonly string DataDirectory =
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "TestData", "IntegrationActivations"));

    private static readonly string DataFile = Path.Combine(DataDirectory, "activations.json");

    public static IEnumerable<TestCaseData> Cases
    {
        get
        {
            if (!File.Exists(DataFile))
            {
                yield return new TestCaseData(ActivationTestCase.Missing()).SetName("ActivationDataMissing");

                yield break;
            }

            var json = File.ReadAllText(DataFile);
            var suite = JsonSerializer.Deserialize<ActivationTestSuite>(
                            json,
                            new JsonSerializerOptions
                            {
                                PropertyNameCaseInsensitive = true
                            })
                        ?? new ActivationTestSuite();

            foreach (var test in suite.Tests)
            {
                var name = string.IsNullOrWhiteSpace(test.Name)
                    ? $"Layer_{test.Layer}"
                    : $"Layer_{test.Layer}_{test.Name}";

                yield return new TestCaseData(test).SetName(name);
            }
        }
    }

    [TestCaseSource(nameof(Cases))]
    public void Execute(ActivationTestCase testCase)
    {
        if (testCase.IsMissing)
            Assert.Ignore("Activation integration data not found. Run ./generate-integration-test-data.sh on macOS to produce it.");

        TestHelpers.RequireNativeOrIgnore();

        using var module = CreateModule(testCase);
        var unaryLayer = (IUnaryLayer)module;
        var input = CreateArray(testCase.Input);
        try
        {
            var output = unaryLayer.Forward(input);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(output), "eval output");
                var actual = TestHelpers.ToFloat32(output);
                var expected = testCase.Output.AsFloatArray();

                Assert.That(actual.Length, Is.EqualTo(expected.Length), "element count mismatch");

                for (var i = 0; i < actual.Length; i++)
                    Assert.That(actual[i], Is.EqualTo(expected[i]).Within(1e-5f), $"mismatch at index {i}");
            }
            finally
            {
                if (output.ctx != 0)
                    MlxArray.Free(output);
            }
        }
        finally
        {
            if (input.ctx != 0)
                MlxArray.Free(input);
        }
    }

    private static Module CreateModule(ActivationTestCase testCase)
    {
        return testCase.Layer switch
        {
            "Sigmoid" => new Sigmoid(),
            "Tanh" => new Tanh(),
            "ReLU" => new ReLU(),
            "SiLU" => new SiLU(),
            "Gelu" => new Gelu(),
            "LeakyReLU" => new LeakyReLU((float)testCase.GetParameterOrDefault("negative_slope", 0.01)),
            "Softmax" => new Softmax((int)testCase.GetParameterOrDefault("axis", -1)),
            _ => throw new NotSupportedException($"Layer '{testCase.Layer}' is not supported by the integration tests.")
        };
    }

    private static MlxArrayHandle CreateArray(TensorPayload tensor)
    {
        if (!string.Equals(tensor.Dtype, "float32", StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException($"Only float32 tensors are supported by the activation tests. Received '{tensor.Dtype}'.");

        var data = tensor.AsFloatArray();
        fixed (float* ptr = data)
        fixed (int* shape = tensor.Shape)
        {
            return MlxArray.NewData(ptr, shape, tensor.Shape.Length, MlxDType.MLX_FLOAT32);
        }
    }

    public sealed class ActivationTestSuite
    {
        public List<ActivationTestCase> Tests { get; set; } = [];
    }

    public sealed class ActivationTestCase
    {
        public string Name { get; set; } = string.Empty;
        public string Layer { get; set; } = string.Empty;
        public Dictionary<string, double>? Parameters { get; set; }
        public TensorPayload Input { get; set; } = new();
        public TensorPayload Output { get; set; } = new();

        public bool IsMissing => string.Equals(this.Name, "__missing__", StringComparison.Ordinal);

        public double GetParameterOrDefault(string key, double fallback)
        {
            if (this.Parameters is null)
                return fallback;

            return this.Parameters.TryGetValue(key, out var value) ? value : fallback;
        }

        public static ActivationTestCase Missing() => new() { Name = "__missing__" };
    }

    public sealed class TensorPayload
    {
        public string Dtype { get; set; } = "float32";
        public int[] Shape { get; set; } = [];
        public List<double> Data { get; set; } = [];

        public float[] AsFloatArray()
            => this.Data.Select(v => (float)v).ToArray();
    }
}