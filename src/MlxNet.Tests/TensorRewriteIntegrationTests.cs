// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Diagnostics;
using System.IO;
using NUnit.Framework;

[TestFixture]
public sealed class TensorRewriteIntegrationTests
{
    private static readonly string RepoRoot = FindRepoRoot(AppContext.BaseDirectory);
    private static readonly string MlxNetProjectPath = Path.Combine(RepoRoot, "src/MlxNet/MlxNet.csproj");
    private static readonly string VersionFilePath = Path.Combine(RepoRoot, "VERSION");

    [Test]
    public void ProjectReference_RewriterDiagnostic_IsActive()
    {
        var projectDirectory = CreateTemporaryDirectory(Path.Combine(RepoRoot, ".tmp-tests"));

        try
        {
            File.WriteAllText(Path.Combine(projectDirectory, "ProjectReferenceConsumer.csproj"), $$"""
                <Project Sdk="Microsoft.NET.Sdk">
                    <PropertyGroup>
                        <TargetFramework>net10.0</TargetFramework>
                        <Nullable>enable</Nullable>
                        <LangVersion>latest</LangVersion>
                    </PropertyGroup>

                    <ItemGroup>
                        <ProjectReference Include="{{MlxNetProjectPath}}"/>
                    </ItemGroup>
                </Project>
                """);

            File.WriteAllText(Path.Combine(projectDirectory, "BadTensorMethod.cs"), """
                using Itexoft.Tensors;

                public static class BadTensorMethod
                {
                    public static Tensor Fail(Tensor value)
                    {
                        try
                        {
                            return value;
                        }
                        finally
                        {
                        }
                    }
                }
                """);

            var build = RunDotnet("build ProjectReferenceConsumer.csproj -c Debug", projectDirectory);
            Assert.That(build.ExitCode, Is.Not.EqualTo(0), build.Output);
            Assert.That(build.Output, Does.Contain("MLXT0001"));
        }
        finally
        {
            SafeDeleteDirectory(projectDirectory);
        }
    }

    [Test]
    public void ProjectReference_RewriterRejects_LinkedSourceOutsideProjectDirectory()
    {
        var workDirectory = CreateTemporaryDirectory(Path.Combine(RepoRoot, ".tmp-tests"));
        var projectDirectory = Path.Combine(workDirectory, "project");
        var sharedDirectory = Path.Combine(workDirectory, "shared");
        Directory.CreateDirectory(projectDirectory);
        Directory.CreateDirectory(sharedDirectory);

        try
        {
            File.WriteAllText(Path.Combine(projectDirectory, "ProjectReferenceConsumer.csproj"), $$"""
                <Project Sdk="Microsoft.NET.Sdk">
                    <PropertyGroup>
                        <TargetFramework>net10.0</TargetFramework>
                        <Nullable>enable</Nullable>
                        <LangVersion>latest</LangVersion>
                    </PropertyGroup>

                    <ItemGroup>
                        <ProjectReference Include="{{MlxNetProjectPath}}"/>
                        <Compile Include="../shared/LinkedTensorMethod.cs" Link="LinkedTensorMethod.cs" />
                    </ItemGroup>
                </Project>
                """);

            File.WriteAllText(Path.Combine(sharedDirectory, "LinkedTensorMethod.cs"), """
                using Itexoft.Tensors;

                public static class LinkedTensorMethod
                {
                    public static Tensor Shift(Tensor value)
                    {
                        var shifted = value + 1f;
                        return shifted;
                    }
                }
                """);

            var build = RunDotnet("build ProjectReferenceConsumer.csproj -c Debug", projectDirectory);
            Assert.That(build.ExitCode, Is.Not.EqualTo(0), build.Output);
            Assert.That(build.Output, Does.Contain("MLXT0012"));
        }
        finally
        {
            SafeDeleteDirectory(workDirectory);
        }
    }

    [Test]
    public void ProjectReference_RewriterRejects_BreakStatement()
    {
        var projectDirectory = CreateTemporaryDirectory(Path.Combine(RepoRoot, ".tmp-tests"));

        try
        {
            File.WriteAllText(Path.Combine(projectDirectory, "ProjectReferenceConsumer.csproj"), $$"""
                <Project Sdk="Microsoft.NET.Sdk">
                    <PropertyGroup>
                        <TargetFramework>net10.0</TargetFramework>
                        <Nullable>enable</Nullable>
                        <LangVersion>latest</LangVersion>
                    </PropertyGroup>

                    <ItemGroup>
                        <ProjectReference Include="{{MlxNetProjectPath}}"/>
                    </ItemGroup>
                </Project>
                """);

            File.WriteAllText(Path.Combine(projectDirectory, "BadTensorMethod.cs"), """
                using Itexoft.Tensors;

                public static class BadTensorMethod
                {
                    public static Tensor Fail(Tensor value)
                    {
                        for (;;)
                        {
                            break;
                        }

                        return value;
                    }
                }
                """);

            var build = RunDotnet("build ProjectReferenceConsumer.csproj -c Debug", projectDirectory);
            Assert.That(build.ExitCode, Is.Not.EqualTo(0), build.Output);
            Assert.That(build.Output, Does.Contain("MLXT0001"));
        }
        finally
        {
            SafeDeleteDirectory(projectDirectory);
        }
    }

    [Test]
    public void NuGetPackage_RewriterDiagnostic_IsActive()
    {
        var workDirectory = CreateTemporaryDirectory(Path.GetTempPath());
        var packageDirectory = Path.Combine(workDirectory, "packages");
        Directory.CreateDirectory(packageDirectory);

        try
        {
            var pack = RunDotnet($"pack \"{MlxNetProjectPath}\" -c Debug -o \"{packageDirectory}\"", RepoRoot);
            Assert.That(pack.ExitCode, Is.EqualTo(0), pack.Output);

            var version = File.ReadAllText(VersionFilePath).Trim();
            var consumerDirectory = Path.Combine(workDirectory, "consumer");
            Directory.CreateDirectory(consumerDirectory);
            File.WriteAllText(Path.Combine(consumerDirectory, "nuget.config"), $$"""
                <?xml version="1.0" encoding="utf-8"?>
                <configuration>
                  <packageSources>
                    <clear />
                    <add key="local" value="{{packageDirectory}}" />
                  </packageSources>
                </configuration>
                """);

            File.WriteAllText(Path.Combine(consumerDirectory, "PackageConsumer.csproj"), $$"""
                <Project Sdk="Microsoft.NET.Sdk">
                    <PropertyGroup>
                        <TargetFramework>net10.0</TargetFramework>
                        <Nullable>enable</Nullable>
                        <LangVersion>latest</LangVersion>
                        <RestorePackagesPath>{{Path.Combine(workDirectory, "restore-cache")}}</RestorePackagesPath>
                    </PropertyGroup>

                    <ItemGroup>
                        <PackageReference Include="Itexoft.MLX" Version="{{version}}"/>
                    </ItemGroup>
                </Project>
                """);

            File.WriteAllText(Path.Combine(consumerDirectory, "BadTensorMethod.cs"), """
                using Itexoft.Tensors;

                public static class BadTensorMethod
                {
                    public static Tensor Fail(Tensor value)
                    {
                        try
                        {
                            return value;
                        }
                        finally
                        {
                        }
                    }
                }
                """);

            var build = RunDotnet("build PackageConsumer.csproj -c Debug", consumerDirectory);
            Assert.That(build.ExitCode, Is.Not.EqualTo(0), build.Output);
            Assert.That(build.Output, Does.Contain("MLXT0001"));
        }
        finally
        {
            SafeDeleteDirectory(workDirectory);
        }
    }

    private static (int ExitCode, string Output) RunDotnet(string arguments, string workingDirectory)
    {
        var logPath = Path.Combine(workingDirectory, ".dotnet-output.log");
        var scriptPath = Path.Combine(workingDirectory, ".run-dotnet.zsh");
        File.WriteAllText(scriptPath, $$"""
            cd {{QuoteForShell(workingDirectory)}} || exit 1
            dotnet {{arguments}} > {{QuoteForShell(logPath)}} 2>&1
            """);

        using var process = new Process();
        process.StartInfo.FileName = "/bin/zsh";
        process.StartInfo.ArgumentList.Add(scriptPath);
        process.StartInfo.UseShellExecute = false;
        process.Start();

        process.WaitForExit();
        var output = File.Exists(logPath) ? File.ReadAllText(logPath) : string.Empty;
        return (process.ExitCode, output);
    }

    private static string CreateTemporaryDirectory(string parentDirectory)
    {
        Directory.CreateDirectory(parentDirectory);
        var path = Path.Combine(parentDirectory, "tensor-rewrite-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }

    private static void SafeDeleteDirectory(string path)
    {
        if (!Directory.Exists(path))
            return;

        try
        {
            Directory.Delete(path, recursive: true);
        }
        catch
        {
        }
    }

    private static string QuoteForShell(string value) => "'" + value.Replace("'", "'\"'\"'") + "'";

    private static string FindRepoRoot(string startDirectory)
    {
        var current = new DirectoryInfo(Path.GetFullPath(startDirectory));

        while (current is not null)
        {
            var versionPath = Path.Combine(current.FullName, "VERSION");
            var mlxNetProjectPath = Path.Combine(current.FullName, "src", "MlxNet", "MlxNet.csproj");

            if (File.Exists(versionPath) && File.Exists(mlxNetProjectPath))
                return current.FullName;

            current = current.Parent;
        }

        throw new DirectoryNotFoundException($"Repository root was not found from '{startDirectory}'.");
    }
}
