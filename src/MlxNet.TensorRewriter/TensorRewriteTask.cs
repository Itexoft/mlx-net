// Copyright (c) 2011-2026 Denis Kudelin
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
// If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
// This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using Microsoft.Build.Framework;
using Microsoft.Build.Utilities;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Itexoft.Mlx.TensorRewriter;

public sealed class TensorRewriteTask : Task
{
    [Required] public ITaskItem[] Sources { get; set; } = [];

    [Required] public ITaskItem[] References { get; set; } = [];

    [Required] public string OutputDirectory { get; set; } = string.Empty;

    public string? ProjectDirectory { get; set; }

    public string? ProjectPath { get; set; }

    public string? DefineConstants { get; set; }

    public string? LangVersion { get; set; }

    [Output] public ITaskItem[] RewrittenFiles { get; private set; } = [];

    [Output] public ITaskItem[] RewrittenOriginalFiles { get; private set; } = [];

    public override bool Execute()
    {
        try
        {
            var outputDirectory = Path.GetFullPath(this.OutputDirectory, Environment.CurrentDirectory);
            Directory.CreateDirectory(outputDirectory);

            var parseOptions = CreateParseOptions(this.LangVersion, this.DefineConstants);
            var sourceFiles = this.Sources.Select(item => item.ItemSpec).Where(File.Exists).Distinct(StringComparer.Ordinal).ToArray();
            var trees = sourceFiles.Select(path => CSharpSyntaxTree.ParseText(File.ReadAllText(path), parseOptions, path)).ToArray();
            var references = BuildReferences(this.References);

            var compilation = CSharpCompilation.Create(
                "TensorRewrite",
                trees,
                references,
                new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, allowUnsafe: true));

            var tensorType = compilation.GetTypeByMetadataName("Itexoft.Tensors.Tensor");

            if (tensorType is null)
            {
                this.Log.LogMessage(MessageImportance.Low, "Tensor rewrite skipped: Itexoft.Tensors.Tensor was not found.");
                this.RewrittenFiles = [];
                this.RewrittenOriginalFiles = [];

                return !this.Log.HasLoggedErrors;
            }

            var treeToModel = trees.ToDictionary(tree => tree, tree => compilation.GetSemanticModel(tree), ReferenceEqualityComparer.Instance);
            var diagnostics = new List<RewriteDiagnostic>();
            var rewrittenFiles = new List<ITaskItem>();
            var rewrittenOriginals = new List<ITaskItem>();
            var projectDirectory = ResolveProjectDirectory(this.ProjectDirectory, this.ProjectPath);

            foreach (var tree in trees)
            {
                var model = treeToModel[tree];
                var validator = new TensorProjectValidator(model, tensorType);
                diagnostics.AddRange(validator.Validate(tree.GetRoot()));

                var rewriter = new TensorCompilationRewriter(model, tensorType, diagnostics);
                var rewrittenRoot = rewriter.Visit(tree.GetRoot());

                if (ReferenceEquals(rewrittenRoot, tree.GetRoot()) || rewrittenRoot.ToFullString() == tree.GetRoot().ToFullString())
                    continue;

                if (!TryResolveOutputPath(outputDirectory, projectDirectory, tree.FilePath, out var outputPath, out var pathError))
                {
                    this.Log.LogError("TensorRewrite", "MLXT0012", null, tree.FilePath, 0, 0, 0, 0, pathError);
                    continue;
                }

                Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
                File.WriteAllText(outputPath, rewrittenRoot.NormalizeWhitespace().ToFullString());

                rewrittenFiles.Add(new TaskItem(outputPath));
                rewrittenOriginals.Add(new TaskItem(tree.FilePath));
            }

            foreach (var diagnostic in diagnostics)
                LogDiagnostic(this.Log, diagnostic);

            this.RewrittenFiles = rewrittenFiles.ToArray();
            this.RewrittenOriginalFiles = rewrittenOriginals.ToArray();

            return !this.Log.HasLoggedErrors;
        }
        catch (Exception exception)
        {
            this.Log.LogErrorFromException(exception, true, true, this.ProjectPath);

            return false;
        }
    }

    private static CSharpParseOptions CreateParseOptions(string? langVersionText, string? defineConstants)
    {
        var symbols = string.IsNullOrWhiteSpace(defineConstants)
            ? []
            : defineConstants.Split([';', ','], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        var langVersion = LanguageVersion.Preview;

        if (!string.IsNullOrWhiteSpace(langVersionText) && LanguageVersionFacts.TryParse(langVersionText, out var parsed))
            langVersion = parsed;

        return CSharpParseOptions.Default.WithLanguageVersion(langVersion).WithPreprocessorSymbols(symbols);
    }

    private static PortableExecutableReference[] BuildReferences(IEnumerable<ITaskItem> items) =>
        items.Select(item => item.ItemSpec).Where(path => path.EndsWith(".dll", StringComparison.OrdinalIgnoreCase) && File.Exists(path))
            .Distinct(StringComparer.OrdinalIgnoreCase).Select(path => MetadataReference.CreateFromFile(path)).ToArray();

    private static string ResolveProjectDirectory(string? explicitDirectory, string? projectPath)
    {
        if (!string.IsNullOrWhiteSpace(explicitDirectory))
            return Path.GetFullPath(explicitDirectory, Environment.CurrentDirectory);

        if (!string.IsNullOrWhiteSpace(projectPath))
            return Path.GetDirectoryName(Path.GetFullPath(projectPath, Environment.CurrentDirectory))!;

        return Environment.CurrentDirectory;
    }

    private static string GetRelativePath(string projectDirectory, string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            return Path.GetFileName(filePath);

        var fullPath = Path.GetFullPath(filePath, Environment.CurrentDirectory);

        return Path.GetRelativePath(projectDirectory, fullPath);
    }

    private static bool TryResolveOutputPath(string outputDirectory, string projectDirectory, string filePath, out string outputPath, out string error)
    {
        var relativePath = GetRelativePath(projectDirectory, filePath);
        var candidatePath = Path.GetFullPath(Path.Combine(outputDirectory, relativePath));

        if (IsPathInsideDirectory(outputDirectory, candidatePath))
        {
            outputPath = candidatePath;
            error = string.Empty;

            return true;
        }

        outputPath = string.Empty;
        error =
            $"Tensor rewrite cannot emit '{filePath}' because the rewritten path escapes the intermediate directory '{outputDirectory}'.";

        return false;
    }

    private static bool IsPathInsideDirectory(string directoryPath, string candidatePath)
    {
        var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        var normalizedDirectory = directoryPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;

        return candidatePath.StartsWith(normalizedDirectory, comparison);
    }

    private static void LogDiagnostic(TaskLoggingHelper logger, RewriteDiagnostic diagnostic) => logger.LogError(
        "TensorRewrite",
        diagnostic.Code,
        null,
        diagnostic.FilePath,
        diagnostic.Line,
        diagnostic.Column,
        diagnostic.EndLine,
        diagnostic.EndColumn,
        diagnostic.Message);

    private sealed class TensorCompilationRewriter(SemanticModel model, INamedTypeSymbol tensorType, List<RewriteDiagnostic> diagnostics)
        : CSharpSyntaxRewriter
    {
        public override SyntaxNode VisitMethodDeclaration(MethodDeclarationSyntax node)
        {
            if (!this.MethodRequiresRewrite(node))
                return node;

            if (node.Body is null && node.ExpressionBody is null)
                return node;

            var bodyRewriter = new TensorMethodBodyRewriter(model, tensorType, diagnostics);

            return bodyRewriter.Rewrite(node);
        }

        public override SyntaxNode VisitConstructorDeclaration(ConstructorDeclarationSyntax node)
        {
            if (!this.ConstructorRequiresRewrite(node))
                return node;

            if (node.Body is null && node.ExpressionBody is null)
                return node;

            var bodyRewriter = new TensorMethodBodyRewriter(model, tensorType, diagnostics);

            return bodyRewriter.Rewrite(node);
        }

        public override SyntaxNode VisitPropertyDeclaration(PropertyDeclarationSyntax node)
        {
            if (!this.PropertyRequiresRewrite(node))
                return node;

            var bodyRewriter = new TensorMethodBodyRewriter(model, tensorType, diagnostics);

            return bodyRewriter.Rewrite(node);
        }

        private bool MethodRequiresRewrite(MethodDeclarationSyntax node)
        {
            var symbol = model.GetDeclaredSymbol(node);

            if (symbol is null)
                return false;

            if (this.ContainsTensor(symbol.ReturnType))
                return true;

            if (symbol.Parameters.Any(parameter => this.ContainsTensor(parameter.Type)))
                return true;

            return this.ContainsTensorExpression(node.Body) || this.ContainsTensorExpression(node.ExpressionBody?.Expression);
        }

        private bool ConstructorRequiresRewrite(ConstructorDeclarationSyntax node)
        {
            var symbol = model.GetDeclaredSymbol(node);

            if (symbol is null)
                return false;

            if (symbol.Parameters.Any(parameter => this.ContainsTensor(parameter.Type)))
                return true;

            return this.ContainsTensorExpression(node.Body) || this.ContainsTensorExpression(node.ExpressionBody?.Expression);
        }

        private bool PropertyRequiresRewrite(PropertyDeclarationSyntax node)
        {
            var symbol = model.GetDeclaredSymbol(node);

            if (symbol is null)
                return false;

            if (this.ContainsTensor(symbol.Type))
                return true;

            return this.ContainsTensorExpression(node.ExpressionBody?.Expression)
                || this.ContainsTensorExpression(node.AccessorList);
        }

        private bool ContainsTensor(ITypeSymbol? type)
        {
            if (type is null)
                return false;

            if (SymbolEqualityComparer.Default.Equals(type, tensorType))
                return true;

            return type switch
            {
                IArrayTypeSymbol array => this.ContainsTensor(array.ElementType),
                IPointerTypeSymbol pointer => this.ContainsTensor(pointer.PointedAtType),
                INamedTypeSymbol named => named.IsTupleType
                    ? named.TupleElements.Any(element => this.ContainsTensor(element.Type))
                    : named.TypeArguments.Any(this.ContainsTensor),
                _ => false,
            };
        }

        private bool IsTensorValueFlow(ExpressionSyntax expression)
        {
            foreach (var node in expression.DescendantNodesAndSelf().OfType<ExpressionSyntax>())
            {
                if (!this.IsTensorValueExpression(node))
                    continue;

                if (!this.IsScalarTensorReceiver(node))
                    return true;
            }

            return false;
        }

        private bool ContainsTensorExpression(SyntaxNode? node) =>
            node is not null && node.DescendantNodesAndSelf().OfType<ExpressionSyntax>().Any(this.IsTensorValueExpression);

        private bool IsScalarTensorReceiver(ExpressionSyntax expression) => expression.Parent switch
        {
            MemberAccessExpressionSyntax member when member.Expression == expression => this.MemberReturnsNonTensor(member),
            ElementAccessExpressionSyntax element when element.Expression == expression && !this.IsTensorValueExpression(element) => true,
            _ => false,
        };

        private bool MemberReturnsNonTensor(MemberAccessExpressionSyntax member)
        {
            if (this.IsTensorValueExpression(member))
                return false;

            return model.GetSymbolInfo(member).Symbol switch
            {
                IPropertySymbol property => !this.ContainsTensor(property.Type),
                IFieldSymbol field => !this.ContainsTensor(field.Type),
                IMethodSymbol method => !this.ContainsTensor(method.ReturnType),
                _ => true,
            };
        }

        private bool IsTensorValueExpression(ExpressionSyntax expression)
        {
            if (!ReferenceEquals(expression.SyntaxTree, model.SyntaxTree))
                return false;

            try
            {
                if (expression is IdentifierNameSyntax identifier)
                {
                    var identifierSymbol = model.GetSymbolInfo(identifier).Symbol;

                    if (identifierSymbol is INamedTypeSymbol or IAliasSymbol)
                        return false;
                }

                if (expression is MemberAccessExpressionSyntax memberAccess && model.GetSymbolInfo(memberAccess).Symbol is IMethodSymbol)
                    return false;

                var typeInfo = model.GetTypeInfo(expression);

                return SymbolEqualityComparer.Default.Equals(typeInfo.Type ?? typeInfo.ConvertedType, tensorType);
            }
            catch (ArgumentException)
            {
                return false;
            }
        }
    }

    private sealed class TensorProjectValidator(SemanticModel model, INamedTypeSymbol tensorType)
    {
        public IEnumerable<RewriteDiagnostic> Validate(SyntaxNode root)
        {
            foreach (var field in root.DescendantNodes().OfType<FieldDeclarationSyntax>())
            {
                foreach (var variable in field.Declaration.Variables)
                {
                    if (model.GetDeclaredSymbol(variable) is IFieldSymbol symbol && this.ContainsTensor(symbol.Type))
                        yield return RewriteDiagnostic.Create("MLXT0008", variable, model, "Tensor cannot be stored in fields.");
                }
            }

            foreach (var property in root.DescendantNodes().OfType<PropertyDeclarationSyntax>())
            {
                if (model.GetDeclaredSymbol(property) is not IPropertySymbol symbol)
                    continue;

                if (this.ContainsTensor(symbol.Type) && !IsSupportedComputedTensorProperty(property, symbol))
                {
                    yield return RewriteDiagnostic.Create("MLXT0009", property, model, "Tensor cannot be stored in properties.");
                    continue;
                }

                if (PropertyTouchesTensor(property) && !IsSupportedComputedTensorProperty(property, symbol))
                    yield return RewriteDiagnostic.Create("MLXT0013", property, model, "Properties that touch Tensor must be read-only computed getters.");
            }
        }

        private static bool IsSupportedComputedTensorProperty(PropertyDeclarationSyntax property, IPropertySymbol symbol)
        {
            if (!symbol.IsReadOnly)
                return false;

            if (property.ExpressionBody is not null)
                return true;

            if (property.AccessorList is null)
                return false;

            if (property.AccessorList.Accessors.Count != 1)
                return false;

            var getter = property.AccessorList.Accessors[0];

            if (!getter.IsKind(SyntaxKind.GetAccessorDeclaration))
                return false;

            return getter.Body is not null || getter.ExpressionBody is not null;
        }

        private bool ContainsTensor(ITypeSymbol? type)
        {
            if (type is null)
                return false;

            if (SymbolEqualityComparer.Default.Equals(type, tensorType))
                return true;

            return type switch
            {
                IArrayTypeSymbol array => this.ContainsTensor(array.ElementType),
                INamedTypeSymbol named => named.IsTupleType
                    ? named.TupleElements.Any(element => this.ContainsTensor(element.Type))
                    : named.TypeArguments.Any(this.ContainsTensor),
                _ => false,
            };
        }

        private bool PropertyTouchesTensor(PropertyDeclarationSyntax property) =>
            this.ContainsTensorExpression(property.ExpressionBody?.Expression)
            || this.ContainsTensorExpression(property.AccessorList);

        private bool ContainsTensorExpression(SyntaxNode? node) =>
            node is not null && node.DescendantNodesAndSelf().OfType<ExpressionSyntax>().Any(this.IsTensorValueExpression);

        private bool IsTensorValueExpression(ExpressionSyntax expression)
        {
            if (!ReferenceEquals(expression.SyntaxTree, model.SyntaxTree))
                return false;

            try
            {
                if (expression is IdentifierNameSyntax identifier)
                {
                    var identifierSymbol = model.GetSymbolInfo(identifier).Symbol;

                    if (identifierSymbol is INamedTypeSymbol or IAliasSymbol)
                        return false;
                }

                if (expression is MemberAccessExpressionSyntax memberAccess && model.GetSymbolInfo(memberAccess).Symbol is IMethodSymbol)
                    return false;

                var typeInfo = model.GetTypeInfo(expression);

                return SymbolEqualityComparer.Default.Equals(typeInfo.Type ?? typeInfo.ConvertedType, tensorType);
            }
            catch (ArgumentException)
            {
                return false;
            }
        }
    }

    private sealed class TensorMethodBodyRewriter(SemanticModel model, INamedTypeSymbol tensorType, List<RewriteDiagnostic> diagnostics)
    {
        private static readonly SymbolDisplayFormat nullableFullyQualifiedFormat = SymbolDisplayFormat.FullyQualifiedFormat.WithMiscellaneousOptions(
            SymbolDisplayFormat.FullyQualifiedFormat.MiscellaneousOptions | SymbolDisplayMiscellaneousOptions.IncludeNullableReferenceTypeModifier);

        private readonly Stack<List<string>> scopes = new();
        private ITypeSymbol? currentMethodReturnType;
        private int tempId;

        public MethodDeclarationSyntax Rewrite(MethodDeclarationSyntax node)
        {
            var diagnosticCount = diagnostics.Count;
            this.ValidateMethod(node);

            if (diagnostics.Count != diagnosticCount)
                return node;

            this.currentMethodReturnType = model.GetDeclaredSymbol(node)?.ReturnType;

            if (node.Body is null && node.ExpressionBody is not null)
            {
                var bodyStatements = node.ReturnType is PredefinedTypeSyntax predefined && predefined.Keyword.IsKind(SyntaxKind.VoidKeyword)
                    ? this.RewriteExpressionBody(node.ExpressionBody.Expression)
                    : this.RewriteReturnExpression(node.ExpressionBody.Expression);

                return node.WithBody(SyntaxFactory.Block(bodyStatements)).WithExpressionBody(null).WithSemicolonToken(default);
            }

            var rewrittenBody = this.RewriteBlock(node.Body!);

            return node.WithBody(rewrittenBody).WithExpressionBody(null).WithSemicolonToken(default);
        }

        public ConstructorDeclarationSyntax Rewrite(ConstructorDeclarationSyntax node)
        {
            var diagnosticCount = diagnostics.Count;
            this.ValidateConstructor(node);

            if (diagnostics.Count != diagnosticCount)
                return node;

            this.currentMethodReturnType = null;

            if (node.Body is null && node.ExpressionBody is not null)
            {
                var bodyStatements = this.RewriteExpressionBody(node.ExpressionBody.Expression);
                return node.WithBody(SyntaxFactory.Block(bodyStatements)).WithExpressionBody(null).WithSemicolonToken(default);
            }

            var rewrittenBody = this.RewriteBlock(node.Body!);

            return node.WithBody(rewrittenBody).WithExpressionBody(null).WithSemicolonToken(default);
        }

        public PropertyDeclarationSyntax Rewrite(PropertyDeclarationSyntax node)
        {
            var diagnosticCount = diagnostics.Count;
            this.ValidateProperty(node);

            if (diagnostics.Count != diagnosticCount)
                return node;

            this.currentMethodReturnType = model.GetDeclaredSymbol(node)?.Type ?? model.GetTypeInfo(node.Type).Type;

            if (node.ExpressionBody is not null)
            {
                var getterBody = SyntaxFactory.Block(this.RewriteReturnExpression(node.ExpressionBody.Expression));
                return node.WithAccessorList(CreateGetterAccessorList(getterBody)).WithExpressionBody(null).WithSemicolonToken(default);
            }

            var accessorList = node.AccessorList;
            var getter = accessorList?.Accessors.SingleOrDefault(accessor => accessor.IsKind(SyntaxKind.GetAccessorDeclaration));

            if (getter is null)
                return node;

            var rewrittenGetter = getter;

            if (getter.Body is not null)
            {
                rewrittenGetter = getter.WithBody(this.RewriteBlock(getter.Body)).WithExpressionBody(null).WithSemicolonToken(default);
            }
            else if (getter.ExpressionBody is not null)
            {
                var getterBody = SyntaxFactory.Block(this.RewriteReturnExpression(getter.ExpressionBody.Expression));
                rewrittenGetter = getter.WithBody(getterBody).WithExpressionBody(null).WithSemicolonToken(default);
            }

            return node.WithAccessorList(accessorList!.WithAccessors(SyntaxFactory.SingletonList(rewrittenGetter)))
                .WithExpressionBody(null)
                .WithSemicolonToken(default);
        }

        private void ValidateMethod(MethodDeclarationSyntax node)
        {
            this.ValidateExecutable(node);

            foreach (var parameter in node.ParameterList.Parameters)
            {
                if (model.GetDeclaredSymbol(parameter) is IParameterSymbol symbol
                    && this.ContainsTensor(symbol.Type)
                    && !SymbolEqualityComparer.Default.Equals(symbol.Type, tensorType))
                    diagnostics.Add(RewriteDiagnostic.Create("MLXT0003", parameter, model, "Tensor method parameters must be plain Tensor values."));
            }
        }

        private void ValidateConstructor(ConstructorDeclarationSyntax node)
        {
            this.ValidateExecutable(node);

            foreach (var parameter in node.ParameterList.Parameters)
            {
                if (model.GetDeclaredSymbol(parameter) is IParameterSymbol symbol
                    && this.ContainsTensor(symbol.Type)
                    && !SymbolEqualityComparer.Default.Equals(symbol.Type, tensorType))
                {
                    diagnostics.Add(
                        RewriteDiagnostic.Create(
                            "MLXT0003",
                            parameter,
                            model,
                            "Tensor method parameters must be plain Tensor values."));
                }
            }
        }

        private void ValidateProperty(PropertyDeclarationSyntax node)
        {
            this.ValidateExecutable(node);

            if (node.ExpressionBody is not null)
                return;

            if (node.AccessorList is null)
            {
                diagnostics.Add(RewriteDiagnostic.Create("MLXT0013", node, model, "Properties that touch Tensor must be read-only computed getters."));
                return;
            }

            if (node.AccessorList.Accessors.Count != 1 || !node.AccessorList.Accessors[0].IsKind(SyntaxKind.GetAccessorDeclaration))
                diagnostics.Add(RewriteDiagnostic.Create("MLXT0013", node, model, "Properties that touch Tensor must be read-only computed getters."));
        }

        private void ValidateExecutable(SyntaxNode node)
        {
            foreach (var invalid in node.DescendantNodes().Where(node => node is TryStatementSyntax
                         or UsingStatementSyntax
                         or BreakStatementSyntax
                         or ContinueStatementSyntax
                         or GotoStatementSyntax
                         or LocalFunctionStatementSyntax
                         or AwaitExpressionSyntax
                         or YieldStatementSyntax
                         or SimpleLambdaExpressionSyntax
                         or ParenthesizedLambdaExpressionSyntax
                         or AnonymousMethodExpressionSyntax))
                diagnostics.Add(RewriteDiagnostic.Create("MLXT0001", invalid, model, "Unsupported syntax in Tensor method."));

            foreach (var variable in node.DescendantNodes().OfType<VariableDeclaratorSyntax>())
            {
                if (model.GetDeclaredSymbol(variable) is ILocalSymbol symbol
                    && this.ContainsTensor(symbol.Type)
                    && !SymbolEqualityComparer.Default.Equals(symbol.Type, tensorType))
                {
                    diagnostics.Add(
                        RewriteDiagnostic.Create(
                            "MLXT0002",
                            variable,
                            model,
                            "Tensor cannot be stored in arrays, tuples, object, dynamic, or collections."));
                }
            }
        }

        private BlockSyntax RewriteBlock(BlockSyntax block)
        {
            var statements = this.RewriteScopedStatements(block.Statements);
            return block.WithStatements(SyntaxFactory.List(statements));
        }

        private List<StatementSyntax> RewriteScopedStatements(SyntaxList<StatementSyntax> statements)
        {
            this.scopes.Push([]);

            try
            {
                return this.RewriteStatementList(statements, 0);
            }
            finally
            {
                this.scopes.Pop();
            }
        }

        private List<StatementSyntax> RewriteStatementList(SyntaxList<StatementSyntax> statements, int index)
        {
            if (index >= statements.Count)
                return [];

            if (this.TryRewriteTensorLocalScope(statements, index, out var rewrittenTensorScope))
                return rewrittenTensorScope.ToList();

            var rewritten = this.RewriteStatement(statements[index]).ToList();
            rewritten.AddRange(this.RewriteStatementList(statements, index + 1));

            return rewritten;
        }

        private bool TryRewriteTensorLocalScope(
            SyntaxList<StatementSyntax> statements,
            int index,
            out IReadOnlyList<StatementSyntax> rewrittenScope)
        {
            if (statements[index] is not LocalDeclarationStatementSyntax declaration
                || !this.CanRewriteTensorLocalScope(declaration))
            {
                rewrittenScope = [];
                return false;
            }

            rewrittenScope = this.RewriteTensorLocalScope(declaration, 0, statements, index + 1);
            return true;
        }

        private bool CanRewriteTensorLocalScope(LocalDeclarationStatementSyntax declaration)
        {
            if (declaration.Declaration.Variables.Count == 0)
                return false;

            foreach (var variable in declaration.Declaration.Variables)
            {
                if (model.GetDeclaredSymbol(variable) is not ILocalSymbol symbol
                    || !this.IsTensor(symbol.Type)
                    || variable.Initializer?.Value is null)
                    return false;
            }

            return true;
        }

        private IReadOnlyList<StatementSyntax> RewriteTensorLocalScope(
            LocalDeclarationStatementSyntax declaration,
            int variableIndex,
            SyntaxList<StatementSyntax> statements,
            int nextStatementIndex)
        {
            if (variableIndex >= declaration.Declaration.Variables.Count)
                return this.RewriteStatementList(statements, nextStatementIndex);

            var variable = declaration.Declaration.Variables[variableIndex];
            var localName = variable.Identifier.Text;
            var lowered = this.LowerTensorValue(variable.Initializer!.Value, TensorStoreKind.Local);
            this.scopes.Peek().Add(localName);

            var bodyStatements = new List<StatementSyntax>();
            bodyStatements.AddRange(this.ComposeTensorAssignment(localName, lowered));
            bodyStatements.AddRange(this.RewriteTensorLocalScope(declaration, variableIndex + 1, statements, nextStatementIndex));

            var rewritten = new List<StatementSyntax>();
            var localDeclaration = TensorDefaultDeclaration(localName);

            if (variableIndex == 0)
                localDeclaration = localDeclaration.WithLeadingTrivia(declaration.GetLeadingTrivia());

            rewritten.Add(localDeclaration);
            rewritten.Add(CreateProtectedCleanupStatement(bodyStatements, [ReleaseStatement(localName)]).WithTrailingTrivia(declaration.GetTrailingTrivia()));

            return rewritten;
        }

        private IEnumerable<StatementSyntax> RewriteStatement(StatementSyntax statement) => statement switch
        {
            BlockSyntax block => [this.RewriteBlock(block)],
            LocalDeclarationStatementSyntax local => this.RewriteLocalDeclaration(local),
            ExpressionStatementSyntax expression => this.RewriteExpressionStatement(expression),
            ReturnStatementSyntax @return => this.RewriteReturn(@return),
            IfStatementSyntax @if => this.RewriteIf(@if),
            WhileStatementSyntax @while => this.RewriteWhile(@while),
            DoStatementSyntax @do => this.RewriteDo(@do),
            ForStatementSyntax @for => this.RewriteFor(@for),
            ForEachStatementSyntax @foreach => this.RewriteForeach(@foreach),
            SwitchStatementSyntax @switch => this.RewriteSwitch(@switch),
            ThrowStatementSyntax @throw => this.RewriteThrow(@throw),
            EmptyStatementSyntax empty => [empty],
            _ => [statement],
        };

        private IEnumerable<StatementSyntax> RewriteLocalDeclaration(LocalDeclarationStatementSyntax statement)
        {
            var rewritten = new List<StatementSyntax>();

            foreach (var variable in statement.Declaration.Variables)
            {
                if (model.GetDeclaredSymbol(variable) is not ILocalSymbol symbol)
                {
                    rewritten.Add(statement);
                    continue;
                }

                if (!this.IsTensor(symbol.Type))
                {
                    if (variable.Initializer is { Value: { } value })
                    {
                        var loweredValue = this.LowerNonTensorExpression(value);
                        if (loweredValue.CleanupStatements.Count == 0)
                        {
                            rewritten.AddRange(loweredValue.PrefixStatements);

                            var localDeclaration =
                                SyntaxFactory.LocalDeclarationStatement(
                                        SyntaxFactory.VariableDeclaration(statement.Declaration.Type.WithTriviaFrom(statement.Declaration.Type))
                                            .WithVariables(
                                                SyntaxFactory.SingletonSeparatedList(
                                                    SyntaxFactory.VariableDeclarator(variable.Identifier)
                                                        .WithInitializer(SyntaxFactory.EqualsValueClause(loweredValue.Expression)))))
                                    .WithLeadingTrivia(statement.GetLeadingTrivia()).WithTrailingTrivia(statement.GetTrailingTrivia());

                            if (!statement.UsingKeyword.IsKind(SyntaxKind.None))
                                localDeclaration = localDeclaration.WithUsingKeyword(statement.UsingKeyword);

                            rewritten.Add(localDeclaration);
                        }
                        else
                        {
                            rewritten.Add(ValueDefaultDeclaration(variable.Identifier.Text, symbol.Type).WithLeadingTrivia(statement.GetLeadingTrivia()));
                            rewritten.AddRange(this.CreateProtectedAssignmentStatements(variable.Identifier.Text, loweredValue, releaseActiveLocalsOnThrow: false));
                        }
                    }
                    else
                    {
                        var localDeclaration =
                            SyntaxFactory.LocalDeclarationStatement(
                                    SyntaxFactory.VariableDeclaration(statement.Declaration.Type.WithTriviaFrom(statement.Declaration.Type))
                                        .WithVariables(SyntaxFactory.SingletonSeparatedList(SyntaxFactory.VariableDeclarator(variable.Identifier))))
                                .WithLeadingTrivia(statement.GetLeadingTrivia()).WithTrailingTrivia(statement.GetTrailingTrivia());

                        if (!statement.UsingKeyword.IsKind(SyntaxKind.None))
                            localDeclaration = localDeclaration.WithUsingKeyword(statement.UsingKeyword);

                        rewritten.Add(localDeclaration);
                    }

                    continue;
                }

                if (variable.Initializer?.Value is null)
                {
                    diagnostics.Add(RewriteDiagnostic.Create("MLXT0005", variable, model, "Tensor locals must be initialized explicitly."));
                    rewritten.Add(statement);

                    continue;
                }

                var lowered = this.LowerTensorValue(variable.Initializer.Value, TensorStoreKind.Local);
                rewritten.AddRange(lowered.PrefixStatements);

                rewritten.Add(
                    TensorLocalDeclaration(variable.Identifier.Text, lowered.Expression).WithLeadingTrivia(statement.GetLeadingTrivia())
                        .WithTrailingTrivia(statement.GetTrailingTrivia()));

                rewritten.AddRange(lowered.CleanupStatements);
                this.scopes.Peek().Add(variable.Identifier.Text);
            }

            return rewritten;
        }

        private IEnumerable<StatementSyntax> RewriteExpressionStatement(ExpressionStatementSyntax statement)
        {
            if (statement.Expression is AssignmentExpressionSyntax assignment
                && model.GetSymbolInfo(assignment.Left).Symbol is ILocalSymbol local
                && this.IsTensor(local.Type))
            {
                var lowered = this.LowerTensorValue(assignment.Right, TensorStoreKind.Local);
                var assignedName = this.NextIdentifier("__mlxTensorAssigned");
                var assignedValue = lowered.Expression is ThrowExpressionSyntax ? lowered : lowered with { Expression = this.CreateStoredTensorExpression(lowered) };

                var statements = new List<StatementSyntax>();
                statements.Add(TensorDefaultDeclaration(assignedName).WithLeadingTrivia(statement.GetLeadingTrivia()));
                statements.AddRange(this.CreateProtectedAssignmentStatements(assignedName, assignedValue, releaseActiveLocalsOnThrow: true));
                statements.Add(ReleaseStatement(local.Name));
                statements.Add(
                    SyntaxFactory.ExpressionStatement(
                            SyntaxFactory.AssignmentExpression(
                                SyntaxKind.SimpleAssignmentExpression,
                                SyntaxFactory.IdentifierName(local.Name),
                                TensorCompilerCall(
                                    "AdoptOwned",
                                    TensorCompilerCall("TakeOwned", assignedName, false),
                                    byRef: false)))
                        .WithTrailingTrivia(statement.GetTrailingTrivia()));

                return statements;
            }

            var loweredExpression = this.LowerNonTensorExpression(statement.Expression);
            return this.CreateProtectedExpressionStatements(
                loweredExpression,
                SyntaxFactory.ExpressionStatement(loweredExpression.Expression).WithLeadingTrivia(statement.GetLeadingTrivia())
                    .WithTrailingTrivia(statement.GetTrailingTrivia()));
        }

        private IEnumerable<StatementSyntax> RewriteExpressionBody(ExpressionSyntax expression)
        {
            var loweredExpression = this.LowerNonTensorExpression(expression);
            return this.CreateProtectedExpressionStatements(loweredExpression, SyntaxFactory.ExpressionStatement(loweredExpression.Expression));
        }

        private IEnumerable<StatementSyntax> RewriteReturn(ReturnStatementSyntax statement)
        {
            if (statement.Expression is null)
            {
                var cleanupStatements = this.ReleaseActiveLocals();
                var statements = cleanupStatements.ToList();
                statements.Add(statement);

                return statements;
            }

            if (this.IsTensorValueExpression(statement.Expression))
            {
                var lowered = this.LowerExpression(statement.Expression);

                if (!lowered.IsTensor || lowered.OriginKind == TensorOriginKind.None || lowered.Identifier is null)
                {
                    diagnostics.Add(RewriteDiagnostic.Create("MLXT0010", statement.Expression, model, "Tensor expression could not be lowered."));

                    return [statement];
                }

                var statements = new List<StatementSyntax>();
                var returnHandleName = this.NextIdentifier("__mlxTensorReturnHandle");
                statements.Add(HandleDefaultDeclaration(returnHandleName));
                statements.AddRange(
                    this.CreateProtectedAssignmentStatements(
                        returnHandleName,
                        lowered with { Expression = this.CreateReturnHandleExpression(lowered) },
                        releaseActiveLocalsOnThrow: false));
                statements.AddRange(this.ReleaseActiveLocals());

                statements.Add(
                    SyntaxFactory.ReturnStatement(TensorCompilerCall("AdoptOwned", returnHandleName, byRef: false))
                        .WithLeadingTrivia(statement.GetLeadingTrivia()).WithTrailingTrivia(statement.GetTrailingTrivia()));

                return statements;
            }

            var loweredScalar = this.LowerNonTensorExpression(statement.Expression);
            var returnValueName = this.NextIdentifier("__mlxReturnValue");
            var rewritten = new List<StatementSyntax>();
            rewritten.Add(this.CreateReturnValueDefaultDeclaration(statement, returnValueName));
            rewritten.AddRange(this.CreateProtectedAssignmentStatements(returnValueName, loweredScalar, releaseActiveLocalsOnThrow: true));
            rewritten.AddRange(this.ReleaseActiveLocals());

            rewritten.Add(
                SyntaxFactory.ReturnStatement(SyntaxFactory.IdentifierName(returnValueName)).WithLeadingTrivia(statement.GetLeadingTrivia())
                    .WithTrailingTrivia(statement.GetTrailingTrivia()));

            return rewritten;
        }

        private IEnumerable<StatementSyntax> RewriteReturnExpression(ExpressionSyntax expression)
        {
            if (this.IsTensorValueExpression(expression))
            {
                var lowered = this.LowerExpression(expression);

                if (!lowered.IsTensor || lowered.OriginKind == TensorOriginKind.None || lowered.Identifier is null)
                {
                    diagnostics.Add(RewriteDiagnostic.Create("MLXT0010", expression, model, "Tensor expression could not be lowered."));

                    return [SyntaxFactory.ReturnStatement(expression)];
                }

                var statements = new List<StatementSyntax>();
                var returnHandleName = this.NextIdentifier("__mlxTensorReturnHandle");
                statements.Add(HandleDefaultDeclaration(returnHandleName));
                statements.AddRange(
                    this.CreateProtectedAssignmentStatements(
                        returnHandleName,
                        lowered with { Expression = this.CreateReturnHandleExpression(lowered) },
                        releaseActiveLocalsOnThrow: false));
                statements.AddRange(this.ReleaseActiveLocals());
                statements.Add(SyntaxFactory.ReturnStatement(TensorCompilerCall("AdoptOwned", returnHandleName, byRef: false)));

                return statements;
            }

            var loweredScalar = this.LowerNonTensorExpression(expression);
            var returnValueName = this.NextIdentifier("__mlxReturnValue");
            var rewritten = new List<StatementSyntax>();
            rewritten.Add(this.CreateReturnValueDefaultDeclaration(returnValueName));
            rewritten.AddRange(this.CreateProtectedAssignmentStatements(returnValueName, loweredScalar, releaseActiveLocalsOnThrow: true));
            rewritten.AddRange(this.ReleaseActiveLocals());
            rewritten.Add(SyntaxFactory.ReturnStatement(SyntaxFactory.IdentifierName(returnValueName)));

            return rewritten;
        }

        private IEnumerable<StatementSyntax> RewriteIf(IfStatementSyntax statement)
        {
            var condition = this.LowerCondition(statement.Condition);
            var statements = new List<StatementSyntax>();
            statements.AddRange(condition.PrefixStatements);

            statements.Add(
                SyntaxFactory.IfStatement(
                        condition.Expression,
                        this.RewriteEmbedded(statement.Statement),
                        statement.Else is null ? null : SyntaxFactory.ElseClause(this.RewriteEmbedded(statement.Else.Statement)))
                    .WithLeadingTrivia(statement.GetLeadingTrivia()).WithTrailingTrivia(statement.GetTrailingTrivia()));

            return statements;
        }

        private IEnumerable<StatementSyntax> RewriteWhile(WhileStatementSyntax statement)
        {
            if (this.ContainsTensorExpression(statement.Condition))
            {
                diagnostics.Add(
                    RewriteDiagnostic.Create("MLXT0006", statement.Condition, model, "Tensor expressions are not allowed in loop conditions."));

                return [statement];
            }

            return [statement.WithStatement(this.RewriteEmbedded(statement.Statement))];
        }

        private IEnumerable<StatementSyntax> RewriteDo(DoStatementSyntax statement)
        {
            if (this.ContainsTensorExpression(statement.Condition))
            {
                diagnostics.Add(
                    RewriteDiagnostic.Create("MLXT0006", statement.Condition, model, "Tensor expressions are not allowed in loop conditions."));

                return [statement];
            }

            return [statement.WithStatement(this.RewriteEmbedded(statement.Statement))];
        }

        private IEnumerable<StatementSyntax> RewriteFor(ForStatementSyntax statement)
        {
            if (statement.Initializers.Any(this.ContainsTensorExpression)
                || statement.Incrementors.Any(this.ContainsTensorExpression)
                || (statement.Condition is not null && this.ContainsTensorExpression(statement.Condition)))
            {
                diagnostics.Add(RewriteDiagnostic.Create("MLXT0006", statement, model, "Tensor expressions are not allowed in for-loop headers."));

                return [statement];
            }

            return [statement.WithStatement(this.RewriteEmbedded(statement.Statement))];
        }

        private IEnumerable<StatementSyntax> RewriteSwitch(SwitchStatementSyntax statement)
        {
            if (this.ContainsTensorExpression(statement.Expression))
            {
                diagnostics.Add(
                    RewriteDiagnostic.Create("MLXT0006", statement.Expression, model, "Tensor expressions are not allowed in switch headers."));

                return [statement];
            }

            var rewrittenSections = new List<SwitchSectionSyntax>(statement.Sections.Count);

            foreach (var section in statement.Sections)
                rewrittenSections.Add(section.WithStatements(SyntaxFactory.List(this.RewriteScopedStatements(section.Statements))));

            return [statement.WithSections(SyntaxFactory.List(rewrittenSections))];
        }

        private IEnumerable<StatementSyntax> RewriteThrow(ThrowStatementSyntax statement)
        {
            if (statement.Expression is null)
            {
                var rewritten = this.ReleaseActiveLocals().ToList();
                rewritten.Add(statement);

                return rewritten;
            }

            var loweredException = this.LowerNonTensorExpression(statement.Expression);
            var exceptionName = this.NextIdentifier("__mlxException");
            var exceptionType = model.GetTypeInfo(statement.Expression).ConvertedType
                ?? model.GetTypeInfo(statement.Expression).Type
                ?? model.Compilation.GetTypeByMetadataName("System.Exception")
                ?? model.Compilation.ObjectType;
            var rewrittenStatements = new List<StatementSyntax>();
            rewrittenStatements.Add(ValueDefaultDeclaration(exceptionName, exceptionType));
            rewrittenStatements.AddRange(this.CreateProtectedAssignmentStatements(exceptionName, loweredException, releaseActiveLocalsOnThrow: true));
            rewrittenStatements.AddRange(this.ReleaseActiveLocals());
            rewrittenStatements.Add(
                SyntaxFactory.ThrowStatement(SyntaxFactory.IdentifierName(exceptionName)).WithLeadingTrivia(statement.GetLeadingTrivia())
                    .WithTrailingTrivia(statement.GetTrailingTrivia()));

            return rewrittenStatements;
        }

        private IEnumerable<StatementSyntax> RewriteForeach(ForEachStatementSyntax statement)
        {
            if (this.ContainsTensorExpression(statement.Expression))
            {
                diagnostics.Add(
                    RewriteDiagnostic.Create("MLXT0007", statement.Expression, model, "Tensor cannot be enumerated or stored in collections."));

                return [statement];
            }

            return [statement.WithStatement(this.RewriteEmbedded(statement.Statement))];
        }

        private StatementSyntax RewriteEmbedded(StatementSyntax statement)
        {
            if (statement is BlockSyntax block)
                return this.RewriteBlock(block);

            this.scopes.Push([]);

            try
            {
                if (statement is LocalDeclarationStatementSyntax local && this.CanRewriteTensorLocalScope(local))
                    return SyntaxFactory.Block(this.RewriteTensorLocalScope(local, 0, default, 0));

                return SyntaxFactory.Block(this.RewriteStatement(statement));
            }
            finally
            {
                this.scopes.Pop();
            }
        }

        private LoweredExpression LowerCondition(ExpressionSyntax expression)
        {
            var lowered = this.LowerNonTensorExpression(expression);

            if (lowered.CleanupStatements.Count == 0)
                return lowered;

            var conditionName = this.NextIdentifier("__mlxCondition");
            var boolType = model.Compilation.GetSpecialType(SpecialType.System_Boolean);

            var prefix = new List<StatementSyntax>
            {
                ValueDefaultDeclaration(conditionName, boolType)
            };

            prefix.AddRange(this.CreateProtectedAssignmentStatements(conditionName, lowered, releaseActiveLocalsOnThrow: true));

            return new LoweredExpression(prefix, SyntaxFactory.IdentifierName(conditionName), [], false, TensorOriginKind.None, null);
        }

        private LoweredExpression LowerTensorValue(ExpressionSyntax expression, TensorStoreKind storeKind)
        {
            var lowered = this.LowerExpression(expression);

            if (lowered.Expression is ThrowExpressionSyntax)
                return lowered;

            if (!lowered.IsTensor || lowered.OriginKind == TensorOriginKind.None || lowered.Identifier is null)
            {
                diagnostics.Add(RewriteDiagnostic.Create("MLXT0010", expression, model, "Tensor expression could not be lowered."));

                return lowered;
            }

            var storeExpression = this.CreateStoredTensorExpression(lowered);

            return lowered with { Expression = storeExpression };
        }

        private LoweredExpression LowerNonTensorExpression(ExpressionSyntax expression)
        {
            var lowered = this.LowerExpression(expression);

            if (lowered.IsTensor)
            {
                diagnostics.Add(
                    RewriteDiagnostic.Create(
                        "MLXT0011",
                        expression,
                        model,
                        "Tensor value cannot be used where a scalar/value expression is required."));
            }

            return lowered;
        }

        private LoweredExpression LowerExpression(ExpressionSyntax expression)
        {
            if (expression is null)
            {
                return new LoweredExpression(
                    [],
                    SyntaxFactory.LiteralExpression(SyntaxKind.NullLiteralExpression),
                    [],
                    false,
                    TensorOriginKind.None,
                    null);
            }

            if (this.IsTensorValueExpression(expression))
                return this.LowerTensorExpression(expression);

            return this.LowerValueExpression(expression);
        }

        private LoweredExpression LowerTensorExpression(ExpressionSyntax expression)
        {
            if (expression is IdentifierNameSyntax identifier
                && model.GetSymbolInfo(identifier).Symbol is ISymbol symbol
                && this.IsTensor(GetSymbolType(symbol)))
            {
                return symbol.Kind switch
                {
                    SymbolKind.Parameter => new LoweredExpression([], identifier, [], true, TensorOriginKind.Parameter, identifier.Identifier.Text),
                    SymbolKind.Local => new LoweredExpression([], identifier, [], true, TensorOriginKind.Local, identifier.Identifier.Text),
                    _ => new LoweredExpression([], identifier, [], true, TensorOriginKind.None, null),
                };
            }

            if (expression is ParenthesizedExpressionSyntax parenthesized)
                return this.LowerTensorExpression(parenthesized.Expression);

            if (expression is ThrowExpressionSyntax throwExpression)
                return this.LowerTensorThrow(throwExpression);

            if (expression is ConditionalExpressionSyntax conditional)
                return this.LowerTensorConditional(conditional);

            var children = GetDirectSubExpressions(expression).Select(this.LowerExpression).ToArray();
            var prefix = new List<StatementSyntax>();

            foreach (var child in children)
                prefix.AddRange(child.PrefixStatements);

            var rewrittenExpression = ReplaceDirectSubExpressions(expression, children.Select(child => child.Expression).ToArray());
            var tempName = this.NextIdentifier("__mlxTensorTemp");
            var childCleanup = children.SelectMany(child => child.CleanupStatements).ToArray();
            prefix.Add(TensorLocalDeclaration(tempName, rewrittenExpression));

            var cleanup = new List<StatementSyntax>
            {
                ReleaseStatement(tempName)
            };

            cleanup.AddRange(childCleanup);

            return new LoweredExpression(
                prefix,
                SyntaxFactory.IdentifierName(tempName),
                cleanup,
                true,
                TensorOriginKind.Temp,
                tempName);
        }

        private LoweredExpression LowerTensorThrow(ThrowExpressionSyntax throwExpression)
        {
            var loweredThrow = this.LowerNonTensorExpression(throwExpression.Expression);

            return new LoweredExpression(
                loweredThrow.PrefixStatements,
                SyntaxFactory.ThrowExpression(loweredThrow.Expression),
                loweredThrow.CleanupStatements,
                true,
                TensorOriginKind.None,
                null);
        }

        private LoweredExpression LowerTensorConditional(ConditionalExpressionSyntax conditional)
        {
            var condition = this.LowerCondition(conditional.Condition);
            var tempName = this.NextIdentifier("__mlxTensorTemp");
            var trueValue = this.LowerTensorValue(conditional.WhenTrue, TensorStoreKind.Local);
            var falseValue = this.LowerTensorValue(conditional.WhenFalse, TensorStoreKind.Local);

            var prefix = new List<StatementSyntax>();
            prefix.AddRange(condition.PrefixStatements);
            prefix.Add(TensorDefaultDeclaration(tempName));

            prefix.Add(
                SyntaxFactory.IfStatement(
                    condition.Expression,
                    SyntaxFactory.Block(this.ComposeTensorAssignment(tempName, trueValue)),
                    SyntaxFactory.ElseClause(SyntaxFactory.Block(this.ComposeTensorAssignment(tempName, falseValue)))));

            return new LoweredExpression(
                prefix,
                SyntaxFactory.IdentifierName(tempName),
                [ReleaseStatement(tempName)],
                true,
                TensorOriginKind.Temp,
                tempName);
        }

        private IEnumerable<StatementSyntax> ComposeTensorAssignment(string targetName, LoweredExpression lowered)
            => this.CreateProtectedAssignmentStatements(targetName, lowered, releaseActiveLocalsOnThrow: true);

        private LoweredExpression LowerValueExpression(ExpressionSyntax expression)
        {
            if (expression is ConditionalExpressionSyntax conditional && this.ContainsAnyTensorExpression(conditional))
                return this.LowerConditionalValue(conditional);

            var children = GetDirectSubExpressions(expression).Select(this.LowerExpression).ToArray();
            var prefix = new List<StatementSyntax>();
            var cleanup = new List<StatementSyntax>();

            foreach (var child in children)
            {
                prefix.AddRange(child.PrefixStatements);
                cleanup.AddRange(child.CleanupStatements);
            }

            var rewritten = ReplaceDirectSubExpressions(expression, children.Select(child => child.Expression).ToArray());

            return new LoweredExpression(prefix, rewritten, cleanup, false, TensorOriginKind.None, null);
        }

        private LoweredExpression LowerConditionalValue(ConditionalExpressionSyntax conditional)
        {
            var condition = this.LowerCondition(conditional.Condition);
            var trueValue = this.LowerNonTensorExpression(conditional.WhenTrue);
            var falseValue = this.LowerNonTensorExpression(conditional.WhenFalse);
            var tempName = this.NextIdentifier("__mlxValueTemp");
            var tempType = model.GetTypeInfo(conditional).ConvertedType ?? model.GetTypeInfo(conditional).Type;

            if (tempType is null)
                return new LoweredExpression(condition.PrefixStatements, conditional, [], false, TensorOriginKind.None, null);

            var prefix = new List<StatementSyntax>();
            prefix.AddRange(condition.PrefixStatements);
            prefix.Add(ValueDefaultDeclaration(tempName, tempType));
            prefix.Add(
                SyntaxFactory.IfStatement(
                    condition.Expression,
                    SyntaxFactory.Block(this.ComposeValueAssignment(tempName, trueValue)),
                    SyntaxFactory.ElseClause(SyntaxFactory.Block(this.ComposeValueAssignment(tempName, falseValue)))));

            return new LoweredExpression(prefix, SyntaxFactory.IdentifierName(tempName), [], false, TensorOriginKind.None, null);
        }

        private IEnumerable<StatementSyntax> ComposeValueAssignment(string targetName, LoweredExpression lowered)
            => this.CreateProtectedAssignmentStatements(targetName, lowered, releaseActiveLocalsOnThrow: true);

        private bool ContainsTensorExpression(ExpressionSyntax expression)
        {
            if (this.IsTensorValueExpression(expression))
                return true;

            return expression.DescendantNodesAndSelf().OfType<ExpressionSyntax>().Any(this.IsMaterialTensorExpression);
        }

        private bool ContainsAnyTensorExpression(SyntaxNode node) =>
            node.DescendantNodesAndSelf().OfType<ExpressionSyntax>().Any(this.IsTensorValueExpression);

        private bool IsTensor(ITypeSymbol? type) => SymbolEqualityComparer.Default.Equals(type, tensorType);

        private bool ContainsTensor(ITypeSymbol? type)
        {
            if (type is null)
                return false;

            if (this.IsTensor(type))
                return true;

            return type switch
            {
                IArrayTypeSymbol array => this.ContainsTensor(array.ElementType),
                INamedTypeSymbol named => named.IsTupleType
                    ? named.TupleElements.Any(element => this.ContainsTensor(element.Type))
                    : named.TypeArguments.Any(this.ContainsTensor),
                _ => false,
            };
        }

        private static ITypeSymbol? GetSymbolType(ISymbol symbol) => symbol switch
        {
            ILocalSymbol local => local.Type,
            IParameterSymbol parameter => parameter.Type,
            IFieldSymbol field => field.Type,
            IPropertySymbol property => property.Type,
            _ => null,
        };

        private bool IsMaterialTensorExpression(ExpressionSyntax expression)
        {
            if (!this.IsTensorValueExpression(expression))
                return false;

            return expression.Parent switch
            {
                MemberAccessExpressionSyntax member when member.Expression == expression => this.MemberAccessProducesTensor(member),
                ElementAccessExpressionSyntax element when element.Expression == expression => this.IsTensorValueExpression(element),
                _ => true,
            };
        }

        private bool MemberAccessProducesTensor(MemberAccessExpressionSyntax member)
        {
            if (this.IsTensorValueExpression(member))
                return true;

            if (!ReferenceEquals(member.SyntaxTree, model.SyntaxTree))
                return false;

            try
            {
                return model.GetSymbolInfo(member).Symbol switch
                {
                    IPropertySymbol property => this.ContainsTensor(property.Type),
                    IFieldSymbol field => this.ContainsTensor(field.Type),
                    IMethodSymbol method => this.ContainsTensor(method.ReturnType),
                    _ => false,
                };
            }
            catch (ArgumentException)
            {
                return false;
            }
        }

        private bool IsTensorValueExpression(ExpressionSyntax expression)
        {
            if (!ReferenceEquals(expression.SyntaxTree, model.SyntaxTree))
                return false;

            try
            {
                if (expression is IdentifierNameSyntax identifier)
                {
                    var identifierSymbol = model.GetSymbolInfo(identifier).Symbol;

                    if (identifierSymbol is INamedTypeSymbol or IAliasSymbol)
                        return false;
                }

                if (expression is MemberAccessExpressionSyntax memberAccess && model.GetSymbolInfo(memberAccess).Symbol is IMethodSymbol)
                    return false;

                var typeInfo = model.GetTypeInfo(expression);

                return this.IsTensor(typeInfo.Type ?? typeInfo.ConvertedType);
            }
            catch (ArgumentException)
            {
                return false;
            }
        }

        private static bool EndsControlFlow(StatementSyntax? statement) => statement switch
        {
            null => false,
            ReturnStatementSyntax => true,
            ThrowStatementSyntax => true,
            BreakStatementSyntax => true,
            ContinueStatementSyntax => true,
            GotoStatementSyntax => true,
            _ => false,
        };

        private IReadOnlyList<StatementSyntax> ReleaseActiveLocals(string? exceptName = null) =>
            this.scopes.SelectMany(scope => scope).Reverse().Where(local => !string.Equals(local, exceptName, StringComparison.Ordinal))
                .Select(ReleaseStatement).ToArray();

        private string NextIdentifier(string prefix) => prefix + this.tempId++.ToString(CultureInfo.InvariantCulture);

        private ExpressionSyntax CreateStoredTensorExpression(LoweredExpression lowered) => lowered.OriginKind switch
        {
            TensorOriginKind.Parameter => TensorCompilerCall(
                "AdoptOwned",
                TensorCompilerCall("RetainBorrowed", lowered.Identifier!, false, false),
                byRef: false),
            TensorOriginKind.Local => TensorCompilerCall(
                "AdoptOwned",
                TensorCompilerCall("RetainBorrowed", lowered.Identifier!, false, false),
                byRef: false),
            TensorOriginKind.Temp => TensorCompilerCall("AdoptOwned", TensorCompilerCall("TakeOwned", lowered.Identifier!, false), byRef: false),
            _ => lowered.Expression,
        };

        private ExpressionSyntax CreateReturnHandleExpression(LoweredExpression lowered) => lowered.OriginKind switch
        {
            TensorOriginKind.Parameter => TensorCompilerCall("RetainBorrowed", lowered.Identifier!, false, false),
            TensorOriginKind.Local => TensorCompilerCall("TakeOwned", lowered.Identifier!, false),
            TensorOriginKind.Temp => TensorCompilerCall("TakeOwned", lowered.Identifier!, false),
            _ => lowered.Expression,
        };

        private static LocalDeclarationStatementSyntax TensorLocalDeclaration(string name, ExpressionSyntax initializer) =>
            SyntaxFactory.LocalDeclarationStatement(
                SyntaxFactory.VariableDeclaration(SyntaxFactory.IdentifierName("Tensor")).WithVariables(
                    SyntaxFactory.SingletonSeparatedList(
                        SyntaxFactory.VariableDeclarator(SyntaxFactory.Identifier(name))
                            .WithInitializer(SyntaxFactory.EqualsValueClause(initializer)))));

        private static LocalDeclarationStatementSyntax HandleLocalDeclaration(string name, ExpressionSyntax initializer) =>
            SyntaxFactory.LocalDeclarationStatement(
                SyntaxFactory.VariableDeclaration(SyntaxFactory.ParseTypeName("global::Itexoft.Mlx.MlxArrayHandle")).WithVariables(
                    SyntaxFactory.SingletonSeparatedList(
                        SyntaxFactory.VariableDeclarator(SyntaxFactory.Identifier(name))
                            .WithInitializer(SyntaxFactory.EqualsValueClause(initializer)))));

        private static LocalDeclarationStatementSyntax HandleDefaultDeclaration(string name) =>
            SyntaxFactory.LocalDeclarationStatement(
                SyntaxFactory.VariableDeclaration(SyntaxFactory.ParseTypeName("global::Itexoft.Mlx.MlxArrayHandle")).WithVariables(
                    SyntaxFactory.SingletonSeparatedList(
                        SyntaxFactory.VariableDeclarator(SyntaxFactory.Identifier(name))
                            .WithInitializer(
                                SyntaxFactory.EqualsValueClause(SyntaxFactory.LiteralExpression(SyntaxKind.DefaultLiteralExpression))))));

        private static LocalDeclarationStatementSyntax TensorDefaultDeclaration(string name) =>
            TensorLocalDeclaration(name, SyntaxFactory.LiteralExpression(SyntaxKind.DefaultLiteralExpression));

        private static LocalDeclarationStatementSyntax VarLocalDeclaration(string name, ExpressionSyntax initializer) =>
            SyntaxFactory.LocalDeclarationStatement(
                SyntaxFactory.VariableDeclaration(SyntaxFactory.IdentifierName("var")).WithVariables(
                    SyntaxFactory.SingletonSeparatedList(
                        SyntaxFactory.VariableDeclarator(SyntaxFactory.Identifier(name))
                            .WithInitializer(SyntaxFactory.EqualsValueClause(initializer)))));

        private static LocalDeclarationStatementSyntax ValueDefaultDeclaration(string name, ITypeSymbol type)
        {
            var typeName = SyntaxFactory.ParseTypeName(type.ToDisplayString(nullableFullyQualifiedFormat));
            var defaultExpression = SyntaxFactory.PostfixUnaryExpression(
                SyntaxKind.SuppressNullableWarningExpression,
                SyntaxFactory.LiteralExpression(SyntaxKind.DefaultLiteralExpression));

            return SyntaxFactory.LocalDeclarationStatement(
                SyntaxFactory.VariableDeclaration(typeName).WithVariables(
                    SyntaxFactory.SingletonSeparatedList(
                        SyntaxFactory.VariableDeclarator(SyntaxFactory.Identifier(name))
                            .WithInitializer(SyntaxFactory.EqualsValueClause(defaultExpression)))));
        }

        private static AccessorListSyntax CreateGetterAccessorList(BlockSyntax getterBody) =>
            SyntaxFactory.AccessorList(
                SyntaxFactory.SingletonList(
                    SyntaxFactory.AccessorDeclaration(SyntaxKind.GetAccessorDeclaration).WithBody(getterBody)));

        private LocalDeclarationStatementSyntax CreateReturnValueDeclaration(
            ReturnStatementSyntax statement,
            string name,
            ExpressionSyntax initializer)
        {
            var returnType = this.currentMethodReturnType ?? (model.GetEnclosingSymbol(statement.SpanStart) as IMethodSymbol)?.ReturnType;

            if (returnType is null)
                return VarLocalDeclaration(name, initializer);

            var typeName = SyntaxFactory.ParseTypeName(returnType.ToDisplayString(nullableFullyQualifiedFormat));

            return SyntaxFactory.LocalDeclarationStatement(
                SyntaxFactory.VariableDeclaration(typeName).WithVariables(
                    SyntaxFactory.SingletonSeparatedList(
                        SyntaxFactory.VariableDeclarator(SyntaxFactory.Identifier(name))
                            .WithInitializer(SyntaxFactory.EqualsValueClause(initializer)))));
        }

        private LocalDeclarationStatementSyntax CreateReturnValueDeclaration(string name, ExpressionSyntax initializer)
        {
            if (this.currentMethodReturnType is null)
                return VarLocalDeclaration(name, initializer);

            var typeName = SyntaxFactory.ParseTypeName(this.currentMethodReturnType.ToDisplayString(nullableFullyQualifiedFormat));

            return SyntaxFactory.LocalDeclarationStatement(
                SyntaxFactory.VariableDeclaration(typeName).WithVariables(
                    SyntaxFactory.SingletonSeparatedList(
                        SyntaxFactory.VariableDeclarator(SyntaxFactory.Identifier(name))
                            .WithInitializer(SyntaxFactory.EqualsValueClause(initializer)))));
        }

        private LocalDeclarationStatementSyntax CreateReturnValueDefaultDeclaration(ReturnStatementSyntax statement, string name)
        {
            var returnType = this.currentMethodReturnType ?? (model.GetEnclosingSymbol(statement.SpanStart) as IMethodSymbol)?.ReturnType;
            return ValueDefaultDeclaration(name, returnType ?? model.Compilation.ObjectType);
        }

        private LocalDeclarationStatementSyntax CreateReturnValueDefaultDeclaration(string name) =>
            ValueDefaultDeclaration(name, this.currentMethodReturnType ?? model.Compilation.ObjectType);

        private IEnumerable<StatementSyntax> CreateProtectedExpressionStatements(LoweredExpression lowered, StatementSyntax body)
        {
            var statements = new List<StatementSyntax>();
            statements.AddRange(lowered.PrefixStatements);
            statements.AddRange(WrapWithCleanup([body], lowered.CleanupStatements));

            return statements;
        }

        private IEnumerable<StatementSyntax> CreateProtectedAssignmentStatements(
            string targetName,
            LoweredExpression lowered,
            bool releaseActiveLocalsOnThrow)
        {
            var statements = new List<StatementSyntax>();
            statements.AddRange(lowered.PrefixStatements);

            var bodyStatements = new List<StatementSyntax>();

            if (lowered.Expression is ThrowExpressionSyntax throwExpression)
            {
                if (releaseActiveLocalsOnThrow)
                    bodyStatements.AddRange(this.ReleaseActiveLocals());

                bodyStatements.Add(SyntaxFactory.ThrowStatement(throwExpression.Expression));
            }
            else
            {
                bodyStatements.Add(
                    SyntaxFactory.ExpressionStatement(
                        SyntaxFactory.AssignmentExpression(
                            SyntaxKind.SimpleAssignmentExpression,
                            SyntaxFactory.IdentifierName(targetName),
                            lowered.Expression)));
            }

            statements.AddRange(WrapWithCleanup(bodyStatements, lowered.CleanupStatements));

            return statements;
        }

        private static IEnumerable<StatementSyntax> WrapWithCleanup(
            IReadOnlyList<StatementSyntax> bodyStatements,
            IReadOnlyList<StatementSyntax> cleanupStatements)
        {
            if (cleanupStatements.Count == 0)
                return bodyStatements;

            return [CreateProtectedCleanupStatement(bodyStatements, cleanupStatements)];
        }

        private static TryStatementSyntax CreateProtectedCleanupStatement(
            IReadOnlyList<StatementSyntax> bodyStatements,
            IReadOnlyList<StatementSyntax> cleanupStatements) =>
            SyntaxFactory.TryStatement(
                SyntaxFactory.Block(bodyStatements),
                SyntaxFactory.List<CatchClauseSyntax>(),
                SyntaxFactory.FinallyClause(SyntaxFactory.Block(cleanupStatements)));

        private static StatementSyntax ReleaseStatement(string name) =>
            SyntaxFactory.ExpressionStatement(TensorCompilerCall("Release", name, false));

        private static InvocationExpressionSyntax TensorCompilerCall(string method, string name, bool returnsTensor = true, bool byRef = true) =>
            TensorCompilerCall(method, SyntaxFactory.IdentifierName(name), returnsTensor, byRef);

        private static InvocationExpressionSyntax TensorCompilerCall(
            string method,
            ExpressionSyntax argumentExpression,
            bool returnsTensor = true,
            bool byRef = true)
        {
            var member = SyntaxFactory.MemberAccessExpression(
                SyntaxKind.SimpleMemberAccessExpression,
                SyntaxFactory.ParseName("global::Itexoft.Tensors.CompilerServices.TensorCompiler"),
                SyntaxFactory.IdentifierName(method));

            var argument = SyntaxFactory.Argument(argumentExpression);

            if (byRef)
                argument = argument.WithRefOrOutKeyword(SyntaxFactory.Token(SyntaxKind.RefKeyword));

            return SyntaxFactory.InvocationExpression(member)
                .WithArgumentList(SyntaxFactory.ArgumentList(SyntaxFactory.SingletonSeparatedList(argument)));
        }

        private static ExpressionSyntax[] GetDirectSubExpressions(ExpressionSyntax expression) => expression switch
        {
            ParenthesizedExpressionSyntax parenthesized => [parenthesized.Expression],
            PrefixUnaryExpressionSyntax prefix => [prefix.Operand],
            PostfixUnaryExpressionSyntax postfix => [postfix.Operand],
            BinaryExpressionSyntax binary => [binary.Left, binary.Right],
            MemberAccessExpressionSyntax member => [member.Expression],
            InvocationExpressionSyntax invocation =>
            [
                .. invocation.Expression is null ? [] : new[] { invocation.Expression },
                .. invocation.ArgumentList.Arguments.Select(argument => argument.Expression),
            ],
            ElementAccessExpressionSyntax element => [element.Expression, .. element.ArgumentList.Arguments.Select(argument => argument.Expression)],
            CastExpressionSyntax cast => [cast.Expression],
            RangeExpressionSyntax range => [.. EnumerateMaybe(range.LeftOperand), .. EnumerateMaybe(range.RightOperand)],
            CollectionExpressionSyntax collection => [.. collection.Elements.OfType<ExpressionElementSyntax>().Select(element => element.Expression)],
            InitializerExpressionSyntax initializer => [.. initializer.Expressions],
            ConditionalExpressionSyntax conditional => [conditional.Condition, conditional.WhenTrue, conditional.WhenFalse],
            _ => [],
        };

        private static IEnumerable<ExpressionSyntax> EnumerateMaybe(ExpressionSyntax? expression)
        {
            if (expression is not null)
                yield return expression;
        }

        private static ExpressionSyntax ReplaceDirectSubExpressions(ExpressionSyntax expression, ExpressionSyntax[] replacements) => expression switch
        {
            ParenthesizedExpressionSyntax parenthesized => parenthesized.WithExpression(replacements[0]),
            PrefixUnaryExpressionSyntax prefix => prefix.WithOperand(replacements[0]),
            PostfixUnaryExpressionSyntax postfix => postfix.WithOperand(replacements[0]),
            BinaryExpressionSyntax binary => binary.WithLeft(replacements[0]).WithRight(replacements[1]),
            MemberAccessExpressionSyntax member => member.WithExpression(replacements[0]),
            InvocationExpressionSyntax invocation => ReplaceInvocation(invocation, replacements),
            ElementAccessExpressionSyntax element => ReplaceElementAccess(element, replacements),
            CastExpressionSyntax cast => cast.WithExpression(replacements[0]),
            RangeExpressionSyntax range => ReplaceRange(range, replacements),
            CollectionExpressionSyntax collection => ReplaceCollection(collection, replacements),
            InitializerExpressionSyntax initializer => initializer.WithExpressions(SyntaxFactory.SeparatedList(replacements)),
            ConditionalExpressionSyntax conditional => conditional.WithCondition(replacements[0]).WithWhenTrue(replacements[1])
                .WithWhenFalse(replacements[2]),
            _ => expression,
        };

        private static ExpressionSyntax ReplaceInvocation(InvocationExpressionSyntax invocation, ExpressionSyntax[] replacements)
        {
            var index = 0;
            var rewrittenTarget = invocation.Expression;

            if (invocation.Expression is not null)
                rewrittenTarget = replacements[index++];

            var arguments = invocation.ArgumentList.Arguments.Select(argument => argument.WithExpression(replacements[index++]));

            return invocation.WithExpression(rewrittenTarget)
                .WithArgumentList(invocation.ArgumentList.WithArguments(SyntaxFactory.SeparatedList(arguments)));
        }

        private static ExpressionSyntax ReplaceElementAccess(ElementAccessExpressionSyntax element, ExpressionSyntax[] replacements)
        {
            var index = 0;
            var target = replacements[index++];
            var arguments = element.ArgumentList.Arguments.Select(argument => argument.WithExpression(replacements[index++]));

            return element.WithExpression(target).WithArgumentList(element.ArgumentList.WithArguments(SyntaxFactory.SeparatedList(arguments)));
        }

        private static ExpressionSyntax ReplaceRange(RangeExpressionSyntax range, ExpressionSyntax[] replacements)
        {
            var index = 0;
            var left = range.LeftOperand is null ? null : replacements[index++];
            var right = range.RightOperand is null ? null : replacements[index++];

            return range.WithLeftOperand(left).WithRightOperand(right);
        }

        private static ExpressionSyntax ReplaceCollection(CollectionExpressionSyntax collection, ExpressionSyntax[] replacements)
        {
            var index = 0;
            var elements = new List<CollectionElementSyntax>(collection.Elements.Count);

            foreach (var element in collection.Elements)
            {
                if (element is ExpressionElementSyntax expressionElement)
                    elements.Add(expressionElement.WithExpression(replacements[index++]));
                else
                    elements.Add(element);
            }

            return collection.WithElements(SyntaxFactory.SeparatedList(elements));
        }
    }

    private enum TensorStoreKind
    {
        Local,
    }

    private enum TensorOriginKind
    {
        None,
        Parameter,
        Local,
        Temp,
    }

    private sealed record LoweredExpression(
        IReadOnlyList<StatementSyntax> PrefixStatements,
        ExpressionSyntax Expression,
        IReadOnlyList<StatementSyntax> CleanupStatements,
        bool IsTensor,
        TensorOriginKind OriginKind,
        string? Identifier);

    private sealed record RewriteDiagnostic(
        string Code,
        string Message,
        string FilePath,
        int Line,
        int Column,
        int EndLine,
        int EndColumn,
        Location MethodLocation)
    {
        public static RewriteDiagnostic Create(string code, SyntaxNode node, SemanticModel model, string message)
        {
            var span = node.GetLocation().GetLineSpan();

            return new RewriteDiagnostic(
                code,
                message,
                span.Path,
                span.StartLinePosition.Line + 1,
                span.StartLinePosition.Character + 1,
                span.EndLinePosition.Line + 1,
                span.EndLinePosition.Character + 1,
                model.GetEnclosingSymbol(node.SpanStart)?.Locations.FirstOrDefault() ?? node.GetLocation());
        }
    }

    private sealed class ReferenceEqualityComparer : IEqualityComparer<SyntaxTree>
    {
        public static ReferenceEqualityComparer Instance { get; } = new();

        public bool Equals(SyntaxTree? x, SyntaxTree? y) => ReferenceEquals(x, y);

        public int GetHashCode(SyntaxTree obj) => RuntimeHelpers.GetHashCode(obj);
    }
}
