// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Collections.Generic;

namespace Itexoft.Mlx.Nn;

public interface IQuantizable
{
    Module ToQuantized(int groupSize, int bits, QuantizationMode mode = QuantizationMode.Affine);
}

public interface IQuantized
{
    int GroupSize { get; }
    int Bits { get; }
    QuantizationMode Mode { get; }
}

public static class Quantization
{
    public static Module? QuantizeSingle(Module module, int groupSize = 64, int bits = 4, QuantizationMode mode = QuantizationMode.Affine)
    {
        if (module is IQuantized)
            return null;

        return module is IQuantizable quantizable
            ? quantizable.ToQuantized(groupSize, bits, mode)
            : null;
    }

    public static void Quantize(
        Module model,
        int groupSize = 64,
        int bits = 4,
        QuantizationMode mode = QuantizationMode.Affine,
        Func<string, Module, bool>? filter = null,
        Func<Module, int, int, QuantizationMode, Module?>? apply = null)
    {
        filter ??= (_, module) => module is IQuantizable && module is not IQuantized;
        apply ??= QuantizeSingle;

        var replacements = new Dictionary<string, Module>(StringComparer.Ordinal);
        foreach (var (path, module) in model.LeafModules())
        {
            if (!filter(path, module))
                continue;

            var replacement = apply(module, groupSize, bits, mode);
            if (replacement is not null)
                replacements[path] = replacement;
        }

        if (replacements.Count > 0)
            model.UpdateModules(replacements, false);
    }

    public static void Quantize(
        Module model,
        Func<string, Module, (int groupSize, int bits, QuantizationMode mode)?> selector,
        Func<Module, int, int, QuantizationMode, Module?>? apply = null)
    {
        apply ??= QuantizeSingle;

        var replacements = new Dictionary<string, Module>(StringComparer.Ordinal);
        foreach (var (path, module) in model.FlattenModules())
        {
            var selection = selector(path, module);

            if (!selection.HasValue)
                continue;

            var (groupSize, bits, mode) = selection.Value;
            var replacement = apply(module, groupSize, bits, mode);
            if (replacement is not null)
                replacements[path] = replacement;
        }

        if (replacements.Count > 0)
            model.UpdateModules(replacements, false);
    }
}

internal static class QuantizationModeExtensions
{
    internal static string ToNativeString(this QuantizationMode mode)
        => mode switch
        {
            QuantizationMode.Affine => "affine",
            QuantizationMode.Mxfp4 => "mxfp4",
            _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, "Unsupported quantization mode.")
        };
}
