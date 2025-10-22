// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Collections.Generic;

namespace Itexoft.Mlx.Nn;

/// <summary>
/// Base type for neural network components that carry parameters and submodules.
/// </summary>
public abstract class Module : IDisposable
{
    private readonly Dictionary<string, ModuleParameter> _parameters = new(StringComparer.Ordinal);
    private readonly Dictionary<string, ModuleBuffer> _buffers = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Module> _submodules = new(StringComparer.Ordinal);

    private bool training = true;
    private bool disposed;

    /// <summary>
    /// Gets a value indicating whether the module is in training mode.
    /// </summary>
    public bool Training => this.training;

    /// <summary>
    /// Marks the module as training or evaluation and cascades to all registered children.
    /// </summary>
    public virtual void Train(bool mode = true)
    {
        if (this.training == mode)
            return;

        this.training = mode;
        this.DidSetTrain(mode);

        foreach (var child in this._submodules.Values)
            child.Train(mode);
    }

    /// <summary>
    /// Convenience helper that places the module in evaluation mode.
    /// </summary>
    public void Eval() => this.Train(false);

    /// <summary>
    /// Called when <see cref="Training"/> is updated.
    /// </summary>
    /// <param name="mode">The new training flag.</param>
    protected virtual void DidSetTrain(bool mode) { }

    /// <summary>
    /// Registers a parameter that belongs to this module.
    /// </summary>
    /// <param name="name">Local name of the parameter.</param>
    /// <param name="value">Array handle representing the parameter tensor.</param>
    /// <param name="trainable">Whether automatic differentiation should produce gradients for this parameter.</param>
    protected ModuleParameter RegisterParameter(string name, MlxArrayHandle value, bool trainable = true)
    {
        this.EnsureNotDisposed();

        if (this._parameters.ContainsKey(name))
            throw new InvalidOperationException($"Parameter '{name}' is already registered on module '{this.GetType().Name}'.");

        var parameter = new ModuleParameter(name, value, trainable);
        this._parameters[name] = parameter;

        return parameter;
    }

    /// <summary>
    /// Registers a persistent non-trainable buffer.
    /// </summary>
    protected ModuleBuffer RegisterBuffer(string name, MlxArrayHandle value)
    {
        this.EnsureNotDisposed();

        if (this._buffers.ContainsKey(name))
            throw new InvalidOperationException($"Buffer '{name}' is already registered on module '{this.GetType().Name}'.");

        var buffer = new ModuleBuffer(name, value);
        this._buffers[name] = buffer;

        return buffer;
    }

    /// <summary>
    /// Registers a child module.
    /// </summary>
    /// <typeparam name="TModule">Type of the child module.</typeparam>
    /// <param name="name">Local identifier for the child.</param>
    /// <param name="module">Module instance to register.</param>
    /// <returns>The provided <paramref name="module"/>.</returns>
    protected TModule RegisterModule<TModule>(string name, TModule module)
        where TModule : Module
    {
        this.EnsureNotDisposed();

        if (this._submodules.ContainsKey(name))
            throw new InvalidOperationException($"Submodule '{name}' is already registered on module '{this.GetType().Name}'.");

        this._submodules[name] = module ?? throw new ArgumentNullException(nameof(module));
        module.Train(this.training);

        return module;
    }

    /// <summary>
    /// Enumerates child modules that were explicitly registered via <see cref="RegisterModule{TModule}(string, TModule)"/>.
    /// </summary>
    public IReadOnlyDictionary<string, Module> Children => this._submodules;

    /// <summary>
    /// Returns all descendant modules keyed by their dotted path.
    /// </summary>
    public Dictionary<string, Module> FlattenModules(bool includeSelf = false)
    {
        var result = new Dictionary<string, Module>(StringComparer.Ordinal);

        if (includeSelf)
            result[string.Empty] = this;

        void Traverse(Module current, string prefix)
        {
            foreach (var (name, child) in current._submodules)
            {
                var path = string.IsNullOrEmpty(prefix) ? name : $"{prefix}.{name}";
                result[path] = child;
                Traverse(child, path);
            }
        }

        Traverse(this, string.Empty);

        return result;
    }

    /// <summary>
    /// Returns only the leaf modules (those without registered children).
    /// </summary>
    public Dictionary<string, Module> LeafModules()
    {
        var result = new Dictionary<string, Module>(StringComparer.Ordinal);

        void Traverse(Module current, string prefix)
        {
            foreach (var (name, child) in current._submodules)
            {
                var path = string.IsNullOrEmpty(prefix) ? name : $"{prefix}.{name}";
                if (child._submodules.Count == 0)
                    result[path] = child;
                else
                    Traverse(child, path);
            }
        }

        Traverse(this, string.Empty);

        return result;
    }

    /// <summary>
    /// Returns a flattened collection of all parameters belonging to this module.
    /// </summary>
    /// <param name="recursive">When <c>true</c>, traverses registered submodules.</param>
    /// <param name="includeFrozen">When <c>false</c>, excludes parameters with <see cref="ModuleParameter.Trainable"/> set to <c>false</c>.</param>
    public ParameterCollection Parameters(bool recursive = true, bool includeFrozen = true)
    {
        var result = new ParameterCollection();
        CollectParameters(this, result, string.Empty, recursive, includeFrozen);

        return result;
    }

    /// <summary>
    /// Returns the subset of parameters that require gradients.
    /// </summary>
    public ParameterCollection TrainableParameters(bool recursive = true)
        => this.Parameters(recursive, false);

    private static void CollectParameters(Module module, ParameterCollection target, string prefix, bool recursive, bool includeFrozen)
    {
        foreach (var (name, parameter) in module._parameters)
        {
            if (!includeFrozen && !parameter.Trainable)
                continue;

            var path = string.IsNullOrEmpty(prefix) ? name : $"{prefix}.{name}";
            target.AddOrUpdate(path, new(parameter.Value, parameter.Trainable));
        }

        if (!recursive)
            return;

        foreach (var (childName, child) in module._submodules)
        {
            var nextPrefix = string.IsNullOrEmpty(prefix) ? childName : $"{prefix}.{childName}";
            CollectParameters(child, target, nextPrefix, recursive, includeFrozen);
        }
    }

    /// <summary>
    /// Returns a flattened set of buffers registered with this module hierarchy.
    /// </summary>
    public ParameterCollection Buffers(bool recursive = true)
    {
        var result = new ParameterCollection();
        CollectBuffers(this, result, string.Empty, recursive);

        return result;
    }

    private static void CollectBuffers(Module module, ParameterCollection target, string prefix, bool recursive)
    {
        foreach (var (name, buffer) in module._buffers)
        {
            var path = string.IsNullOrEmpty(prefix) ? name : $"{prefix}.{name}";
            target.AddOrUpdate(path, new(buffer.Value, false));
        }

        if (!recursive)
            return;

        foreach (var (childName, child) in module._submodules)
        {
            var nextPrefix = string.IsNullOrEmpty(prefix) ? childName : $"{prefix}.{childName}";
            CollectBuffers(child, target, nextPrefix, recursive);
        }
    }

    /// <summary>
    /// Updates parameter tensors with the provided values.
    /// </summary>
    /// <param name="replacement">New values keyed by fully-qualified parameter path.</param>
    /// <param name="strict">When <c>true</c>, throws if a parameter path cannot be resolved.</param>
    /// <param name="disposeReplaced">When <c>true</c>, existing tensors are released.</param>
    public void UpdateParameters(ParameterCollection replacement, bool strict = true, bool disposeReplaced = true)
    {
        this.EnsureNotDisposed();

        var map = BuildParameterMap(this, string.Empty);
        foreach (var (path, entry) in replacement)
        {
            if (!map.TryGetValue(path, out var parameter))
            {
                if (strict)
                    throw new KeyNotFoundException($"Parameter '{path}' could not be resolved on module '{this.GetType().Name}'.");

                continue;
            }

            parameter.SetValue(entry.Value, disposeReplaced);
            parameter.Trainable = entry.Trainable;
        }

        if (strict)
            foreach (var path in map.Keys)
                if (!replacement.ContainsKey(path))
                    throw new InvalidOperationException($"No replacement value was supplied for parameter '{path}'.");
    }

    /// <summary>
    /// Updates registered buffers with new values.
    /// </summary>
    public void UpdateBuffers(ParameterCollection replacement, bool strict = true, bool disposeReplaced = true)
    {
        this.EnsureNotDisposed();

        var map = BuildBufferMap(this, string.Empty);
        foreach (var (path, entry) in replacement)
        {
            if (!map.TryGetValue(path, out var buffer))
            {
                if (strict)
                    throw new KeyNotFoundException($"Buffer '{path}' could not be resolved on module '{this.GetType().Name}'.");

                continue;
            }

            buffer.SetValue(entry.Value, disposeReplaced);
        }

        if (strict)
            foreach (var path in map.Keys)
                if (!replacement.ContainsKey(path))
                    throw new InvalidOperationException($"No replacement value was supplied for buffer '{path}'.");
    }

    /// <summary>
    /// Replaces child modules using their dotted path.
    /// </summary>
    public void UpdateModules(IDictionary<string, Module> replacements, bool strict = true, bool disposeReplaced = true)
    {
        this.EnsureNotDisposed();

        var map = BuildModulePathMap(this, string.Empty);
        foreach (var (path, module) in replacements)
        {
            if (!map.TryGetValue(path, out var entry))
            {
                if (strict)
                    throw new KeyNotFoundException($"Module path '{path}' could not be resolved on '{this.GetType().Name}'.");

                continue;
            }

            entry.Parent.ReplaceModule(entry.Name, module, disposeReplaced);
        }
    }

    private static Dictionary<string, ModuleParameter> BuildParameterMap(Module module, string prefix)
    {
        var result = new Dictionary<string, ModuleParameter>(StringComparer.Ordinal);
        PopulateParameterMap(module, prefix, result);

        return result;
    }

    private static void PopulateParameterMap(Module module, string prefix, Dictionary<string, ModuleParameter> result)
    {
        foreach (var (name, parameter) in module._parameters)
        {
            var path = string.IsNullOrEmpty(prefix) ? name : $"{prefix}.{name}";
            result[path] = parameter;
        }

        foreach (var (childName, child) in module._submodules)
        {
            var nextPrefix = string.IsNullOrEmpty(prefix) ? childName : $"{prefix}.{childName}";
            PopulateParameterMap(child, nextPrefix, result);
        }
    }

    private static Dictionary<string, ModuleBuffer> BuildBufferMap(Module module, string prefix)
    {
        var result = new Dictionary<string, ModuleBuffer>(StringComparer.Ordinal);
        PopulateBufferMap(module, prefix, result);

        return result;
    }

    private static void PopulateBufferMap(Module module, string prefix, Dictionary<string, ModuleBuffer> result)
    {
        foreach (var (name, buffer) in module._buffers)
        {
            var path = string.IsNullOrEmpty(prefix) ? name : $"{prefix}.{name}";
            result[path] = buffer;
        }

        foreach (var (childName, child) in module._submodules)
        {
            var nextPrefix = string.IsNullOrEmpty(prefix) ? childName : $"{prefix}.{childName}";
            PopulateBufferMap(child, nextPrefix, result);
        }
    }

    /// <summary>
    /// Marks parameters as frozen (non-trainable).
    /// </summary>
    /// <param name="paths">Optional set of parameter paths. When omitted, all parameters within the module are frozen.</param>
    /// <param name="recursive">When <c>true</c>, traverses child modules. Ignored when <paramref name="paths"/> are provided.</param>
    public void Freeze(IEnumerable<string>? paths = null, bool recursive = true)
    {
        this.EnsureNotDisposed();

        var parameterMap = BuildParameterMap(this, string.Empty);

        if (paths is null)
        {
            foreach (var key in parameterMap.Keys)
            {
                if (!recursive && !key.Contains('.'))
                {
                    parameterMap[key].Trainable = false;

                    continue;
                }

                if (recursive)
                    parameterMap[key].Trainable = false;
            }

            return;
        }

        foreach (var path in paths)
            if (parameterMap.TryGetValue(path, out var parameter))
                parameter.Trainable = false;
    }

    /// <summary>
    /// Marks parameters as trainable.
    /// </summary>
    public void Unfreeze(IEnumerable<string>? paths = null, bool recursive = true)
    {
        this.EnsureNotDisposed();

        var parameterMap = BuildParameterMap(this, string.Empty);

        if (paths is null)
        {
            foreach (var (key, parameter) in parameterMap)
            {
                if (!recursive && key.Contains('.'))
                    continue;

                parameter.Trainable = true;
            }

            return;
        }

        foreach (var path in paths)
            if (parameterMap.TryGetValue(path, out var parameter))
                parameter.Trainable = true;
    }

    private void EnsureNotDisposed()
    {
        if (this.disposed)
            throw new ObjectDisposedException(this.GetType().FullName);
    }

    private void ReplaceModule(string name, Module module, bool disposeExisting)
    {
        if (!this._submodules.TryGetValue(name, out var existing))
            throw new KeyNotFoundException($"Module '{name}' is not registered on '{this.GetType().Name}'.");

        var replacement = module ?? throw new ArgumentNullException(nameof(module));
        this._submodules[name] = replacement;
        replacement.Train(this.training);

        if (disposeExisting && !ReferenceEquals(existing, replacement))
            existing.Dispose();
    }

    private static Dictionary<string, ModulePathEntry> BuildModulePathMap(Module module, string prefix)
    {
        var result = new Dictionary<string, ModulePathEntry>(StringComparer.Ordinal);

        foreach (var (name, child) in module._submodules)
        {
            var path = string.IsNullOrEmpty(prefix) ? name : $"{prefix}.{name}";
            result[path] = new(module, name);

            foreach (var kv in BuildModulePathMap(child, path))
                result[kv.Key] = kv.Value;
        }

        return result;
    }

    private readonly struct ModulePathEntry(Module parent, string name)
    {
        public Module Parent { get; } = parent;
        public string Name { get; } = name;
    }

    /// <inheritdoc/>
    public virtual void Dispose()
    {
        if (this.disposed)
            return;

        foreach (var parameter in this._parameters.Values)
            parameter.Dispose();

        this._parameters.Clear();

        foreach (var buffer in this._buffers.Values)
            buffer.Dispose();

        this._buffers.Clear();

        foreach (var module in this._submodules.Values)
            module.Dispose();

        this._submodules.Clear();

        this.disposed = true;
    }
}