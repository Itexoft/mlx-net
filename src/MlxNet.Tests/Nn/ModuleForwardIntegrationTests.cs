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
public unsafe class ModuleForwardIntegrationTests
{
    private static readonly string DataDirectory =
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "TestData", "IntegrationModules"));

    private static readonly string DataFile = Path.Combine(DataDirectory, "modules.json");

    public static IEnumerable<TestCaseData> Cases
    {
        get
        {
            if (!File.Exists(DataFile))
            {
                yield return new TestCaseData(ModuleTestCase.Missing()).SetName("ModuleDataMissing");

                yield break;
            }

            var json = File.ReadAllText(DataFile);
            var suite = JsonSerializer.Deserialize<ModuleTestSuite>(
                            json,
                            new JsonSerializerOptions
                            {
                                PropertyNameCaseInsensitive = true
                            })
                        ?? new ModuleTestSuite();

            foreach (var test in suite.Tests)
            {
                var name = string.IsNullOrWhiteSpace(test.Name)
                    ? $"Module_{test.Layer}"
                    : $"Module_{test.Layer}_{test.Name}";

                yield return new TestCaseData(test).SetName(name);
            }
        }
    }

    [TestCaseSource(nameof(Cases))]
    public void Execute(ModuleTestCase testCase)
    {
        if (testCase.IsMissing)
            Assert.Ignore("Module integration data not found. Run ./generate-integration-test-data.sh on macOS to produce it.");

        TestHelpers.RequireNativeOrIgnore();

        using var module = CreateModule(testCase);
        ApplyParameters(module, testCase.Parameters);

        if (module is not IUnaryLayer unary)
        {
            Assert.Fail($"Module '{testCase.Layer}' does not implement IUnaryLayer.");

            return;
        }

        var input = CreateArray(testCase.Input);
        try
        {
            var output = unary.Forward(input);
            try
            {
                TestHelpers.Ok(MlxArray.Eval(output), "eval output");
                var actual = TestHelpers.ToFloat32(output);
                var expected = testCase.Output.AsFloatArray();

                Assert.That(actual.Length, Is.EqualTo(expected.Length), "Element count mismatch.");

                for (var i = 0; i < actual.Length; i++)
                    Assert.That(actual[i], Is.EqualTo(expected[i]).Within(1e-4f), $"Mismatch at index {i}");
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

    private static Module CreateModule(ModuleTestCase testCase)
    {
        return testCase.Layer switch
        {
            "Linear" => new Linear(
                testCase.Settings.RequireInt("inputDimensions"),
                testCase.Settings.RequireInt("outputDimensions"),
                testCase.Settings.RequireBool("bias", true)),
            "Conv1d" => new Conv1d(
                testCase.Settings.RequireInt("in_channels"),
                testCase.Settings.RequireInt("out_channels"),
                testCase.Settings.RequireInt("kernel_size"),
                testCase.Settings.RequireInt("stride", 1),
                testCase.Settings.RequireInt("padding", 0),
                testCase.Settings.RequireInt("dilation", 1),
                testCase.Settings.RequireInt("groups", 1),
                testCase.Settings.RequireBool("bias", true)),
            "Conv2d" => new Conv2d(
                testCase.Settings.RequireInt("in_channels"),
                testCase.Settings.RequireInt("out_channels"),
                testCase.Settings.RequirePair("kernel_size"),
                testCase.Settings.RequirePair("stride", new IntPair(1, 1)),
                testCase.Settings.RequirePair("padding", new IntPair(0, 0)),
                testCase.Settings.RequirePair("dilation", new IntPair(1, 1)),
                testCase.Settings.RequireInt("groups", 1),
                testCase.Settings.RequireBool("bias", true)),
            _ => throw new NotSupportedException($"Module '{testCase.Layer}' is not supported.")
        };
    }

    private static void ApplyParameters(Module module, List<ModuleParameterPayload> parameters)
    {
        var updates = new ParameterCollection();
        foreach (var parameter in parameters)
        {
            var handle = CreateArray(parameter.Tensor);
            updates.AddOrUpdate(parameter.Path, new(handle, parameter.Trainable));
        }

        module.UpdateParameters(updates, true, true);
    }

    private static MlxArrayHandle CreateArray(TensorPayload tensor)
    {
        if (!string.Equals(tensor.Dtype, "float32", StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException($"Tensor dtype '{tensor.Dtype}' is not supported in module tests.");

        var data = tensor.AsFloatArray();
        fixed (float* ptr = data)
        fixed (int* shape = tensor.Shape)
        {
            return MlxArray.NewData(ptr, shape, tensor.Shape.Length, MlxDType.MLX_FLOAT32);
        }
    }

    public sealed class ModuleTestSuite
    {
        public List<ModuleTestCase> Tests { get; set; } = [];
    }

    public sealed class ModuleTestCase
    {
        public string Name { get; set; } = string.Empty;
        public string Layer { get; set; } = string.Empty;
        public Dictionary<string, JsonElement> Settings { get; set; } = new();
        public TensorPayload Input { get; set; } = new();
        public TensorPayload Output { get; set; } = new();
        public List<ModuleParameterPayload> Parameters { get; set; } = [];

        public bool IsMissing => string.Equals(this.Name, "__missing__", StringComparison.Ordinal);

        public static ModuleTestCase Missing() => new() { Name = "__missing__" };
    }

    public sealed class ModuleParameterPayload
    {
        public string Path { get; set; } = string.Empty;
        public TensorPayload Tensor { get; set; } = new();
        public bool Trainable { get; set; } = true;
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

file static class JsonSettingsExtensions
{
    public static int RequireInt(this Dictionary<string, JsonElement> settings, string key, int? defaultValue = null)
    {
        if (!settings.TryGetValue(key, out var element))
        {
            if (defaultValue.HasValue)
                return defaultValue.Value;

            throw new KeyNotFoundException($"Setting '{key}' is missing.");
        }

        return element.ValueKind switch
        {
            JsonValueKind.Number => element.GetInt32(),
            JsonValueKind.Array => ReadArray(element) switch
            {
                { Length: 1 } values => values[0],
                _ => throw new InvalidOperationException($"Setting '{key}' must contain a single integer.")
            },
            _ => throw new InvalidOperationException($"Setting '{key}' is not an integer.")
        };
    }

    public static bool RequireBool(this Dictionary<string, JsonElement> settings, string key, bool defaultValue = false)
    {
        if (!settings.TryGetValue(key, out var element))
            return defaultValue;

        return element.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.Number => Math.Abs(element.GetDouble()) > double.Epsilon,
            _ => defaultValue
        };
    }

    public static IntPair RequirePair(this Dictionary<string, JsonElement> settings, string key, IntPair? defaultValue = null)
    {
        if (!settings.TryGetValue(key, out var element))
            return defaultValue ?? throw new KeyNotFoundException($"Setting '{key}' is missing.");

        if (element.ValueKind == JsonValueKind.Array)
        {
            var values = ReadArray(element);

            if (values.Length == 2)
                return new(values[0], values[1]);
        }

        throw new InvalidOperationException($"Setting '{key}' must be an array of two integers.");
    }

    private static int[] ReadArray(JsonElement element)
    {
        var list = new List<int>();
        foreach (var item in element.EnumerateArray())
            list.Add(item.GetInt32());

        return list.ToArray();
    }
}