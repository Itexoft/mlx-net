using System.Diagnostics.CodeAnalysis;
using NUnit.Framework;

[assembly: LevelOfParallelism(1)]
[assembly: NonParallelizable]
[assembly: SuppressMessage("ReSharper", "UnassignedField.Local")]