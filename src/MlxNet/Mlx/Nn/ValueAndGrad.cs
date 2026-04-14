// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace Itexoft.Mlx.Nn;

public static class ValueAndGrad
{
    public static Func<Module, MlxArrayHandle, MlxArrayHandle, (MlxArrayHandle, ParameterCollection)> Build(
        Module model,
        Func<Module, MlxArrayHandle, MlxArrayHandle, MlxArrayHandle> loss)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(loss);

        var compiled = BuildInternal<(MlxArrayHandle first, MlxArrayHandle second)>(
            model,
            (m, _, args) =>
            {
                var (first, second) = args;

                return [loss(m, first, second)];
            },
            true);

        return (current, first, second) =>
        {
            EnsureSameInstance(model, current);

            var parameters = current.TrainableParameters();
            var (values, gradients) = compiled(parameters, (first, second));

            if (values.Length != 1)
                throw new InvalidOperationException("Loss function must return exactly one tensor.");

            return (values[0], gradients);
        };
    }

    public static Func<Module, IReadOnlyList<MlxArrayHandle>, (MlxArrayHandle[], ParameterCollection)> Build(
        Module model,
        Func<Module, IReadOnlyList<MlxArrayHandle>?, MlxArrayHandle[]> loss)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(loss);

        var compiled = BuildInternal<IReadOnlyList<MlxArrayHandle>>(model, (m, _, arrays) => loss(m, arrays), true);

        return (current, arrays) =>
        {
            EnsureSameInstance(model, current);

            var parameters = current.TrainableParameters();

            return compiled(parameters, arrays ?? []);
        };
    }

    public static Func<Module, TArguments, (MlxArrayHandle[], ParameterCollection)> Build<TArguments>(
        Module model,
        Func<Module, TArguments?, MlxArrayHandle[]> loss)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(loss);

        var compiled = BuildInternal<TArguments>(model, (m, _, args) => loss(m, args), false);

        return (current, args) =>
        {
            EnsureSameInstance(model, current);

            var parameters = current.TrainableParameters();

            return compiled(parameters, args);
        };
    }

    private static Func<ParameterCollection, TArgs, (MlxArrayHandle[] Values, ParameterCollection Gradients)> BuildInternal<TArgs>(
        Module model,
        Func<Module, ParameterCollection, TArgs?, MlxArrayHandle[]> loss,
        bool cacheKernel)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(loss);

        LossAdapter adapter = (m, parameters, state) => loss(m, parameters, state is null ? default : (TArgs)state);

        if (!cacheKernel)
        {
            return (parameters, args) =>
            {
                ArgumentNullException.ThrowIfNull(parameters);

                var layout = ParameterLayout.From(parameters);
                using var kernel = ValueAndGradKernel.Create(model, adapter, layout);

                return kernel.Invoke(parameters, args);
            };
        }

        var gate = new object();
        ValueAndGradKernel? kernel = null;
        ParameterLayout? cachedLayout = null;

        return (parameters, args) =>
        {
            ArgumentNullException.ThrowIfNull(parameters);

            lock (gate)
            {
                if (kernel is null || cachedLayout is null || !cachedLayout.Matches(parameters))
                {
                    kernel?.Dispose();
                    cachedLayout = ParameterLayout.From(parameters);
                    kernel = ValueAndGradKernel.Create(model, adapter, cachedLayout);
                }

                return kernel.Invoke(parameters, args);
            }
        };
    }

    private static void EnsureSameInstance(Module expected, Module actual)
    {
        if (!ReferenceEquals(expected, actual))
            throw new InvalidOperationException("ValueAndGrad closure is bound to the model used during construction.");
    }

    private delegate MlxArrayHandle[] LossAdapter(Module model, ParameterCollection parameters, object? state);

    private unsafe sealed class ValueAndGradKernel : IDisposable
    {
        private readonly Lock gate = new();
        private readonly ParameterLayout layout;
        private readonly LossAdapter loss;
        private readonly Module model;

        private MlxClosureHandle closure;
        private MlxClosureValueAndGradHandle closureValueAndGrad;
        private object? currentArguments;
        private bool disposed;
        private Exception? pendingException;

        private ValueAndGradKernel(Module model, LossAdapter loss, ParameterLayout layout)
        {
            this.model = model;
            this.loss = loss;
            this.layout = layout;
            var argumentNumbers1 = CreateArgumentNumbers(layout.Count);

            var handle = GCHandle.Alloc(this);
            var payload = (void*)GCHandle.ToIntPtr(handle);

            try
            {
                this.closure = MlxClosure.NewFuncPayload(&Invoke, payload, &Destroy);

                fixed (int* numbers = argumentNumbers1)
                {
                    var ptr = argumentNumbers1.Length > 0 ? numbers : null;
                    var status = MlxTransforms.ValueAndGrad(out this.closureValueAndGrad, this.closure, ptr, (nuint)argumentNumbers1.Length);
                    TensorUtilities.CheckStatus(status, "value_and_grad");
                }
            }
            catch
            {
                if (this.closure.ctx != 0)
                {
                    MlxClosure.Free(this.closure);
                    this.closure = default;
                }
                else if (handle.IsAllocated)
                    handle.Free();

                throw;
            }
        }

        public void Dispose()
        {
            this.Dispose(true);
            GC.SuppressFinalize(this);
        }

        ~ValueAndGradKernel() => this.Dispose(false);

        internal static ValueAndGradKernel Create(Module model, LossAdapter loss, ParameterLayout layout) => new(model, loss, layout);

        internal (MlxArrayHandle[] Values, ParameterCollection Gradients) Invoke<TArgs>(ParameterCollection parameters, TArgs args)
        {
            this.EnsureNotDisposed();

            var primals = this.layout.ExtractHandles(parameters);
            var boxedArgs = (object?)args;

            lock (this.gate)
            {
                this.currentArguments = boxedArgs;
                var inputVector = TensorVectorUtilities.Create(primals);

                try
                {
                    var status = MlxClosure.ValueAndGradApply(out var valuesHandle, out var gradientsHandle, this.closureValueAndGrad, inputVector);

                    if (status != 0)
                    {
                        var captured = this.pendingException;
                        this.pendingException = null;

                        if (captured is not null)
                            throw captured;

                        TensorUtilities.CheckStatus(status, "closure_value_and_grad_apply");
                    }

                    this.pendingException = null;
                    var values = TensorVectorUtilities.Consume(valuesHandle);
                    var gradientArrays = TensorVectorUtilities.Consume(gradientsHandle);
                    var gradients = this.layout.Rehydrate(gradientArrays);

                    return (values, gradients);
                }
                finally
                {
                    this.currentArguments = null;

                    if (inputVector.ctx != 0)
                        MlxVector.ArrayFree(inputVector);
                }
            }
        }

        private static int[] CreateArgumentNumbers(int count)
        {
            if (count == 0)
                return [];

            var result = new int[count];

            for (var i = 0; i < count; i++)
                result[i] = i;

            return result;
        }

        private void EnsureNotDisposed()
        {
            if (this.disposed)
                throw new ObjectDisposedException(nameof(ValueAndGradKernel));
        }

        [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
        private static int Invoke(MlxVectorArrayHandle* result, MlxVectorArrayHandle input, void* payload)
        {
            if (payload == null)
                return -1;

            var handle = GCHandle.FromIntPtr((nint)payload);

            if (!handle.IsAllocated || handle.Target is not ValueAndGradKernel kernel)
                return -1;

            return kernel.Execute(result, input);
        }

        private int Execute(MlxVectorArrayHandle* result, MlxVectorArrayHandle input)
        {
            try
            {
                var primals = ReadHandles(input);
                var parameters = this.layout.Rehydrate(primals);

                this.model.UpdateParameters(parameters, true, false);

                var outputs = this.loss(this.model, parameters, this.currentArguments);
                *result = TensorVectorUtilities.Create(outputs);
                this.pendingException = null;

                return 0;
            }
            catch (Exception ex)
            {
                this.pendingException = ex;

                return -1;
            }
        }

        private static MlxArrayHandle[] ReadHandles(MlxVectorArrayHandle vector)
        {
            var count = (int)MlxVector.ArraySize(vector);

            if (count == 0)
                return [];

            var result = new MlxArrayHandle[count];

            for (var i = 0; i < count; i++)
            {
                var status = MlxVector.ArrayGet(out result[i], vector, (nuint)i);
                TensorUtilities.CheckStatus(status, "vector_array_get");
            }

            return result;
        }

        [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
        private static void Destroy(void* payload)
        {
            if (payload == null)
                return;

            try
            {
                var handle = GCHandle.FromIntPtr((nint)payload);

                if (handle.IsAllocated)
                    handle.Free();
            }
            catch
            {
                // Ignore handle release failures during native teardown.
            }
        }

        private void Dispose(bool disposing)
        {
            if (this.disposed)
                return;

            try
            {
                if (this.closureValueAndGrad.ctx != 0)
                {
                    var status = MlxClosure.ValueAndGradFree(this.closureValueAndGrad);

                    if (disposing)
                        TensorUtilities.CheckStatus(status, "closure_value_and_grad_free");
                }

                if (this.closure.ctx != 0)
                {
                    var status = MlxClosure.Free(this.closure);

                    if (disposing)
                        TensorUtilities.CheckStatus(status, "closure_free");
                }
            }
            catch when (!disposing)
            {
                // Suppress exceptions during finalization.
            }
            finally
            {
                this.closureValueAndGrad = default;
                this.closure = default;
                this.disposed = true;
            }
        }
    }

    private sealed class ParameterLayout
    {
        private readonly string[] paths;
        private readonly bool[] trainable;

        private ParameterLayout(string[] paths, bool[] trainable)
        {
            this.paths = paths;
            this.trainable = trainable;
        }

        internal int Count => this.paths.Length;

        internal static ParameterLayout From(ParameterCollection parameters)
        {
            ArgumentNullException.ThrowIfNull(parameters);

            var entries = new List<(string Path, ParameterEntry Entry)>(parameters.Count);

            foreach (var (path, entry) in parameters)
                entries.Add((path, entry));

            entries.Sort(static (left, right) => string.CompareOrdinal(left.Path, right.Path));

            var count = entries.Count;
            var paths = new string[count];
            var trainable = new bool[count];

            for (var i = 0; i < count; i++)
            {
                paths[i] = entries[i].Path;
                trainable[i] = entries[i].Entry.Trainable;
            }

            return new(paths, trainable);
        }

        internal bool Matches(ParameterCollection parameters)
        {
            ArgumentNullException.ThrowIfNull(parameters);

            if (parameters.Count != this.paths.Length)
                return false;

            for (var i = 0; i < this.paths.Length; i++)
            {
                if (!parameters.TryGetValue(this.paths[i], out var entry))
                    return false;

                if (entry.Trainable != this.trainable[i])
                    return false;
            }

            return true;
        }

        internal MlxArrayHandle[] ExtractHandles(ParameterCollection parameters)
        {
            ArgumentNullException.ThrowIfNull(parameters);

            var result = new MlxArrayHandle[this.paths.Length];

            for (var i = 0; i < this.paths.Length; i++)
            {
                if (!parameters.TryGetValue(this.paths[i], out var entry))
                    throw new InvalidOperationException($"Parameter '{this.paths[i]}' is missing from the supplied collection.");

                result[i] = entry.Value;
            }

            return result;
        }

        internal ParameterCollection Rehydrate(IReadOnlyList<MlxArrayHandle> arrays)
        {
            ArgumentNullException.ThrowIfNull(arrays);

            if (arrays.Count != this.paths.Length)
                throw new InvalidOperationException("Gradient count does not match parameter layout.");

            var collection = new ParameterCollection();

            for (var i = 0; i < this.paths.Length; i++)
                collection.AddOrUpdate(this.paths[i], new(arrays[i], this.trainable[i]));

            return collection;
        }
    }
}
