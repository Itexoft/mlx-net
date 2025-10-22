// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Collections;
using System.Collections.Generic;

namespace Itexoft.Mlx.Nn;

/// <summary>
/// Represents a flattened view of module parameters keyed by hierarchical path.
/// </summary>
public sealed class ParameterCollection : IReadOnlyDictionary<string, ParameterEntry>
{
    private readonly Dictionary<string, ParameterEntry> _entries;

    /// <summary>
    /// Initializes a new empty collection.
    /// </summary>
    public ParameterCollection() => this._entries = new(StringComparer.Ordinal);

    private ParameterCollection(Dictionary<string, ParameterEntry> entries) => this._entries = entries;

    /// <summary>
    /// Gets the number of stored items.
    /// </summary>
    public int Count => this._entries.Count;

    /// <inheritdoc/>
    public IEnumerable<string> Keys => this._entries.Keys;

    /// <inheritdoc/>
    public IEnumerable<ParameterEntry> Values => this._entries.Values;

    /// <inheritdoc/>
    public ParameterEntry this[string key] => this._entries[key];

    /// <summary>
    /// Adds or replaces an entry.
    /// </summary>
    /// <param name="path">Hierarchical parameter path (dot notation).</param>
    /// <param name="entry">Entry for the parameter.</param>
    public void AddOrUpdate(string path, ParameterEntry entry)
    {
        this._entries[path] = entry;
    }

    /// <summary>
    /// Determines whether a parameter with the provided path exists.
    /// </summary>
    public bool ContainsKey(string path) => this._entries.ContainsKey(path);

    /// <inheritdoc/>
    public bool TryGetValue(string key, out ParameterEntry value) => this._entries.TryGetValue(key, out value);

    /// <inheritdoc/>
    public IEnumerator<KeyValuePair<string, ParameterEntry>> GetEnumerator() => this._entries.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => this._entries.GetEnumerator();

    /// <summary>
    /// Returns a shallow copy of the collection.
    /// </summary>
    public ParameterCollection Clone()
        => new(new(this._entries, StringComparer.Ordinal));

    /// <summary>
    /// Returns a filtered collection containing only entries that match the provided predicate.
    /// </summary>
    /// <param name="predicate">Predicate that receives the fully-qualified parameter path.</param>
    public ParameterCollection Where(Func<string, bool> predicate)
    {
        var filtered = new Dictionary<string, ParameterEntry>(StringComparer.Ordinal);
        foreach (var (key, entry) in this._entries)
            if (predicate(key))
                filtered[key] = entry;

        return new(filtered);
    }

    /// <summary>
    /// Projects the collection into a sequence using the provided selector.
    /// </summary>
    public IEnumerable<TResult> Select<TResult>(Func<string, ParameterEntry, TResult> selector)
    {
        foreach (var (key, entry) in this._entries)
            yield return selector(key, entry);
    }
}

/// <summary>
/// Represents a single flattened parameter entry.
/// </summary>
public readonly record struct ParameterEntry(MlxArrayHandle Value, bool Trainable);