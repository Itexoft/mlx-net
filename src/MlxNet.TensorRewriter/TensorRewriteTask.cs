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
                var ctx = new TensorContext(model, tensorType);
                diagnostics.AddRange(new TensorProjectValidator(ctx).Validate(tree.GetRoot()));

                var rewriter = new TensorCompilationRewriter(ctx, diagnostics);
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

    private static bool TryResolveOutputPath(string outputDirectory, string projectDirectory, string filePath, out string outputPath, out string error)
    {
        var relativePath = string.IsNullOrWhiteSpace(filePath)
            ? Path.GetFileName(filePath)
            : Path.GetRelativePath(projectDirectory, Path.GetFullPath(filePath, Environment.CurrentDirectory));
        var candidatePath = Path.GetFullPath(Path.Combine(outputDirectory, relativePath));

        var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        var normalizedDirectory = outputDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;

        if (candidatePath.StartsWith(normalizedDirectory, comparison))
        {
            outputPath = candidatePath;
            error = string.Empty;
            return true;
        }

        outputPath = string.Empty;
        error = $"Tensor rewrite cannot emit '{filePath}' because the rewritten path escapes the intermediate directory '{outputDirectory}'.";
        return false;
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

    private sealed record TensorContext(SemanticModel Model, INamedTypeSymbol TensorType)
    {
        public bool IsTensor(ITypeSymbol? type) => SymbolEqualityComparer.Default.Equals(type, this.TensorType);

        public bool ContainsTensor(ITypeSymbol? type) => type switch
        {
            null => false,
            _ when this.IsTensor(type) => true,
            IArrayTypeSymbol array => this.ContainsTensor(array.ElementType),
            IPointerTypeSymbol pointer => this.ContainsTensor(pointer.PointedAtType),
            INamedTypeSymbol named => named.IsTupleType
                ? named.TupleElements.Any(element => this.ContainsTensor(element.Type))
                : named.TypeArguments.Any(this.ContainsTensor),
            _ => false,
        };

        public bool ContainsTensorExpression(SyntaxNode? node) =>
            node is not null && node.DescendantNodesAndSelf().OfType<ExpressionSyntax>().Any(this.IsTensorValueExpression);

        public bool IsTensorValueExpression(ExpressionSyntax expression)
        {
            if (!ReferenceEquals(expression.SyntaxTree, this.Model.SyntaxTree))
                return false;

            try
            {
                if (expression is IdentifierNameSyntax identifier
                    && this.Model.GetSymbolInfo(identifier).Symbol is INamedTypeSymbol or IAliasSymbol)
                    return false;

                if (expression is MemberAccessExpressionSyntax memberAccess
                    && this.Model.GetSymbolInfo(memberAccess).Symbol is IMethodSymbol)
                    return false;

                var typeInfo = this.Model.GetTypeInfo(expression);
                return this.IsTensor(typeInfo.Type ?? typeInfo.ConvertedType);
            }
            catch (ArgumentException)
            {
                return false;
            }
        }

        public static ITypeSymbol? GetSymbolType(ISymbol symbol) => symbol switch
        {
            ILocalSymbol local => local.Type,
            IParameterSymbol parameter => parameter.Type,
            IFieldSymbol field => field.Type,
            IPropertySymbol property => property.Type,
            _ => null,
        };
    }

    private sealed class TensorCompilationRewriter(TensorContext ctx, List<RewriteDiagnostic> diagnostics) : CSharpSyntaxRewriter
    {
        public override SyntaxNode VisitMethodDeclaration(MethodDeclarationSyntax node) =>
            this.RequiresRewrite(node) && (node.Body is not null || node.ExpressionBody is not null)
                ? new TensorMethodBodyRewriter(ctx, diagnostics).Rewrite(node)
                : node;

        public override SyntaxNode VisitConstructorDeclaration(ConstructorDeclarationSyntax node) =>
            this.RequiresRewrite(node) && (node.Body is not null || node.ExpressionBody is not null)
                ? new TensorMethodBodyRewriter(ctx, diagnostics).Rewrite(node)
                : node;

        public override SyntaxNode VisitPropertyDeclaration(PropertyDeclarationSyntax node) =>
            this.RequiresRewrite(node)
                ? new TensorMethodBodyRewriter(ctx, diagnostics).Rewrite(node)
                : node;

        private bool RequiresRewrite(MethodDeclarationSyntax node)
        {
            var symbol = ctx.Model.GetDeclaredSymbol(node);
            return symbol is not null
                   && (ctx.ContainsTensor(symbol.ReturnType)
                       || symbol.Parameters.Any(p => ctx.ContainsTensor(p.Type))
                       || ctx.ContainsTensorExpression(node.Body)
                       || ctx.ContainsTensorExpression(node.ExpressionBody?.Expression));
        }

        private bool RequiresRewrite(ConstructorDeclarationSyntax node)
        {
            var symbol = ctx.Model.GetDeclaredSymbol(node);
            return symbol is not null
                   && (symbol.Parameters.Any(p => ctx.ContainsTensor(p.Type))
                       || ctx.ContainsTensorExpression(node.Body)
                       || ctx.ContainsTensorExpression(node.ExpressionBody?.Expression));
        }

        private bool RequiresRewrite(PropertyDeclarationSyntax node)
        {
            var symbol = ctx.Model.GetDeclaredSymbol(node);
            return symbol is not null
                   && (ctx.ContainsTensor(symbol.Type)
                       || ctx.ContainsTensorExpression(node.ExpressionBody?.Expression)
                       || ctx.ContainsTensorExpression(node.AccessorList));
        }
    }

    private sealed class TensorProjectValidator(TensorContext ctx)
    {
        public IEnumerable<RewriteDiagnostic> Validate(SyntaxNode root)
        {
            foreach (var field in root.DescendantNodes().OfType<FieldDeclarationSyntax>())
            {
                foreach (var variable in field.Declaration.Variables)
                {
                    if (ctx.Model.GetDeclaredSymbol(variable) is IFieldSymbol symbol && ctx.ContainsTensor(symbol.Type))
                        yield return RewriteDiagnostic.Create("MLXT0008", variable, ctx.Model, "Tensor cannot be stored in fields.");
                }
            }

            foreach (var property in root.DescendantNodes().OfType<PropertyDeclarationSyntax>())
            {
                if (ctx.Model.GetDeclaredSymbol(property) is not IPropertySymbol symbol)
                    continue;

                var supported = IsSupportedComputedTensorProperty(property, symbol);

                if (ctx.ContainsTensor(symbol.Type) && !supported)
                {
                    yield return RewriteDiagnostic.Create("MLXT0009", property, ctx.Model, "Tensor cannot be stored in properties.");
                    continue;
                }

                if (ctx.ContainsTensorExpression(property.ExpressionBody?.Expression) || ctx.ContainsTensorExpression(property.AccessorList))
                {
                    if (!supported)
                        yield return RewriteDiagnostic.Create("MLXT0013", property, ctx.Model, "Properties that touch Tensor must be read-only computed getters.");
                }
            }
        }

        private static bool IsSupportedComputedTensorProperty(PropertyDeclarationSyntax property, IPropertySymbol symbol)
        {
            if (!symbol.IsReadOnly)
                return false;

            if (property.ExpressionBody is not null)
                return true;

            if (property.AccessorList is null || property.AccessorList.Accessors.Count != 1)
                return false;

            var getter = property.AccessorList.Accessors[0];
            return getter.IsKind(SyntaxKind.GetAccessorDeclaration) && (getter.Body is not null || getter.ExpressionBody is not null);
        }
    }

    private sealed class TensorMethodBodyRewriter(TensorContext ctx, List<RewriteDiagnostic> diagnostics)
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

            this.currentMethodReturnType = ctx.Model.GetDeclaredSymbol(node)?.ReturnType;
            var returnsVoid = node.ReturnType is PredefinedTypeSyntax predefined && predefined.Keyword.IsKind(SyntaxKind.VoidKeyword);

            return node.WithBody(this.RewriteMemberBody(node.Body, node.ExpressionBody, returnsVoid))
                .WithExpressionBody(null).WithSemicolonToken(default);
        }

        public ConstructorDeclarationSyntax Rewrite(ConstructorDeclarationSyntax node)
        {
            var diagnosticCount = diagnostics.Count;
            this.ValidateConstructor(node);

            if (diagnostics.Count != diagnosticCount)
                return node;

            this.currentMethodReturnType = null;

            return node.WithBody(this.RewriteMemberBody(node.Body, node.ExpressionBody, returnsVoid: true))
                .WithExpressionBody(null).WithSemicolonToken(default);
        }

        public PropertyDeclarationSyntax Rewrite(PropertyDeclarationSyntax node)
        {
            var diagnosticCount = diagnostics.Count;
            this.ValidateProperty(node);

            if (diagnostics.Count != diagnosticCount)
                return node;

            this.currentMethodReturnType = ctx.Model.GetDeclaredSymbol(node)?.Type ?? ctx.Model.GetTypeInfo(node.Type).Type;

            if (node.ExpressionBody is not null)
            {
                var getterBody = SyntaxFactory.Block(this.RewriteReturnExpression(node.ExpressionBody.Expression));
                return node.WithAccessorList(CreateGetterAccessorList(getterBody)).WithExpressionBody(null).WithSemicolonToken(default);
            }

            var accessorList = node.AccessorList;
            var getter = accessorList?.Accessors.SingleOrDefault(accessor => accessor.IsKind(SyntaxKind.GetAccessorDeclaration));

            if (getter is null)
                return node;

            var rewrittenGetter = getter.Body is not null
                ? getter.WithBody(this.RewriteBlock(getter.Body)).WithExpressionBody(null).WithSemicolonToken(default)
                : getter.ExpressionBody is not null
                    ? getter.WithBody(SyntaxFactory.Block(this.RewriteReturnExpression(getter.ExpressionBody.Expression)))
                        .WithExpressionBody(null).WithSemicolonToken(default)
                    : getter;

            return node.WithAccessorList(accessorList!.WithAccessors(SyntaxFactory.SingletonList(rewrittenGetter)))
                .WithExpressionBody(null).WithSemicolonToken(default);
        }

        private BlockSyntax RewriteMemberBody(BlockSyntax? body, ArrowExpressionClauseSyntax? expressionBody, bool returnsVoid)
        {
            if (body is null && expressionBody is not null)
            {
                var statements = returnsVoid
                    ? this.RewriteExpressionBody(expressionBody.Expression)
                    : this.RewriteReturnExpression(expressionBody.Expression);
                return SyntaxFactory.Block(statements);
            }

            return this.RewriteBlock(body!);
        }

        private void ValidateMethod(MethodDeclarationSyntax node)
        {
            this.ValidateExecutable(node);
            this.ValidateTensorReturnEscape(ctx.Model.GetDeclaredSymbol(node)?.ReturnType, node.ExpressionBody?.Expression);

            var returnType = ctx.Model.GetDeclaredSymbol(node)?.ReturnType;
            if (returnType is not null)
            {
                foreach (var @return in node.DescendantNodes().OfType<ReturnStatementSyntax>())
                    this.ValidateTensorReturnEscape(returnType, @return.Expression);
            }

            this.ValidatePlainTensorParameters(node.ParameterList);
        }

        private void ValidateConstructor(ConstructorDeclarationSyntax node)
        {
            this.ValidateExecutable(node);
            this.ValidatePlainTensorParameters(node.ParameterList);
        }

        private void ValidateProperty(PropertyDeclarationSyntax node)
        {
            this.ValidateExecutable(node);
            var propertyType = ctx.Model.GetDeclaredSymbol(node)?.Type ?? ctx.Model.GetTypeInfo(node.Type).Type;
            this.ValidateTensorReturnEscape(propertyType, node.ExpressionBody?.Expression);

            if (node.ExpressionBody is not null)
                return;

            if (node.AccessorList is null
                || node.AccessorList.Accessors.Count != 1
                || !node.AccessorList.Accessors[0].IsKind(SyntaxKind.GetAccessorDeclaration))
                diagnostics.Add(RewriteDiagnostic.Create("MLXT0013", node, ctx.Model, "Properties that touch Tensor must be read-only computed getters."));

            if (propertyType is null || node.AccessorList is null)
                return;

            foreach (var getter in node.AccessorList.Accessors.Where(accessor => accessor.IsKind(SyntaxKind.GetAccessorDeclaration)))
            {
                foreach (var @return in getter.DescendantNodes().OfType<ReturnStatementSyntax>())
                    this.ValidateTensorReturnEscape(propertyType, @return.Expression);
            }
        }

        private void ValidatePlainTensorParameters(ParameterListSyntax parameterList)
        {
            foreach (var parameter in parameterList.Parameters)
            {
                if (ctx.Model.GetDeclaredSymbol(parameter) is IParameterSymbol symbol
                    && ctx.ContainsTensor(symbol.Type)
                    && !ctx.IsTensor(symbol.Type))
                    diagnostics.Add(RewriteDiagnostic.Create("MLXT0003", parameter, ctx.Model, "Tensor method parameters must be plain Tensor values."));
            }
        }

        private void ValidateExecutable(SyntaxNode node)
        {
            foreach (var invalid in node.DescendantNodes().Where(n => n is TryStatementSyntax
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
                diagnostics.Add(RewriteDiagnostic.Create("MLXT0001", invalid, ctx.Model, "Unsupported syntax in Tensor method."));

            foreach (var variable in node.DescendantNodes().OfType<VariableDeclaratorSyntax>())
            {
                if (ctx.Model.GetDeclaredSymbol(variable) is ILocalSymbol symbol
                    && ctx.ContainsTensor(symbol.Type)
                    && !ctx.IsTensor(symbol.Type))
                    diagnostics.Add(RewriteDiagnostic.Create("MLXT0002", variable, ctx.Model,
                        "Tensor cannot be stored in arrays, tuples, object, dynamic, or collections."));
            }

            foreach (var statement in node.DescendantNodes().OfType<StatementSyntax>())
            {
                if (IsLegacyUnsupportedStatement(statement) || IsSupportedTensorStatement(statement))
                    continue;

                diagnostics.Add(RewriteDiagnostic.Create("MLXT0015", statement, ctx.Model,
                    $"Unsupported statement syntax '{statement.Kind()}' in Tensor method."));
            }

            foreach (var expression in node.DescendantNodes().OfType<ExpressionSyntax>().Where(IsEvaluatedExpressionRoot))
                this.ValidateExpressionTree(expression);
        }

        private static bool IsEvaluatedExpressionRoot(ExpressionSyntax expression) => expression.Parent is
            ArrowExpressionClauseSyntax or
            EqualsValueClauseSyntax or
            ExpressionStatementSyntax or
            ReturnStatementSyntax or
            ThrowStatementSyntax or
            IfStatementSyntax or
            WhileStatementSyntax or
            DoStatementSyntax or
            ForStatementSyntax or
            SwitchStatementSyntax or
            LockStatementSyntax or
            UsingStatementSyntax or
            YieldStatementSyntax;

        private void ValidateExpressionTree(ExpressionSyntax expression)
        {
            if (!this.IsSupportedValueExpression(expression))
            {
                diagnostics.Add(RewriteDiagnostic.Create("MLXT0014", expression, ctx.Model,
                    $"Unsupported expression syntax '{expression.Kind()}' in Tensor method."));
                return;
            }

            if (this.ContainsAnyTensorExpression(expression))
            {
                if (this.ContainsCompositeTensorEscape(expression))
                {
                    diagnostics.Add(RewriteDiagnostic.Create("MLXT0016", expression, ctx.Model,
                        "Tensor values cannot be embedded into tuples, arrays, object creation, or collection expressions."));
                    return;
                }

                if (!this.IsSupportedTensorExpression(expression))
                {
                    diagnostics.Add(RewriteDiagnostic.Create("MLXT0014", expression, ctx.Model,
                        $"Unsupported expression syntax '{expression.Kind()}' in Tensor method."));
                    return;
                }
            }

            foreach (var child in GetDirectSubExpressions(expression))
                this.ValidateExpressionTree(child);
        }

        private void ValidateTensorReturnEscape(ITypeSymbol? returnType, ExpressionSyntax? expression)
        {
            if (returnType is null || expression is null || ctx.ContainsTensor(returnType) || !ctx.IsTensorValueExpression(expression))
                return;

            diagnostics.Add(RewriteDiagnostic.Create("MLXT0017", expression, ctx.Model,
                "Tensor values cannot be returned through non-Tensor return types."));
        }

        private bool IsSupportedValueExpression(ExpressionSyntax expression) => expression switch
        {
            IdentifierNameSyntax => true,
            GenericNameSyntax => true,
            LiteralExpressionSyntax => true,
            PredefinedTypeSyntax => true,
            ThisExpressionSyntax => true,
            BaseExpressionSyntax => true,
            ParenthesizedExpressionSyntax => true,
            PrefixUnaryExpressionSyntax => true,
            PostfixUnaryExpressionSyntax => true,
            BinaryExpressionSyntax => true,
            CheckedExpressionSyntax => true,
            ArrayCreationExpressionSyntax => true,
            ImplicitArrayCreationExpressionSyntax => true,
            StackAllocArrayCreationExpressionSyntax => true,
            ImplicitStackAllocArrayCreationExpressionSyntax => true,
            TupleExpressionSyntax => true,
            MemberAccessExpressionSyntax => true,
            MemberBindingExpressionSyntax => true,
            InvocationExpressionSyntax => true,
            ElementAccessExpressionSyntax => true,
            ElementBindingExpressionSyntax => true,
            ConditionalAccessExpressionSyntax conditionalAccess => !this.ContainsAnyTensorExpression(conditionalAccess),
            CastExpressionSyntax => true,
            DeclarationExpressionSyntax => true,
            RangeExpressionSyntax => true,
            IsPatternExpressionSyntax => true,
            ObjectCreationExpressionSyntax => true,
            ImplicitObjectCreationExpressionSyntax => true,
            CollectionExpressionSyntax collection => collection.Elements.All(element => element is ExpressionElementSyntax),
            InitializerExpressionSyntax => true,
            InterpolatedStringExpressionSyntax => true,
            ConditionalExpressionSyntax => true,
            DefaultExpressionSyntax => true,
            ThrowExpressionSyntax => true,
            AssignmentExpressionSyntax assignment => IsSupportedValueAssignment(assignment),
            _ => false,
        };

        private bool IsSupportedTensorExpression(ExpressionSyntax expression) =>
            this.IsSupportedValueExpression(expression)
            && (expression is not AssignmentExpressionSyntax assignment || this.IsSupportedTensorAssignment(assignment));

        private bool ContainsCompositeTensorEscape(ExpressionSyntax expression) => expression switch
        {
            ArrayCreationExpressionSyntax arrayCreation => this.InitializerContainsTensorEscape(arrayCreation.Initializer),
            ImplicitArrayCreationExpressionSyntax implicitArrayCreation => this.InitializerContainsTensorEscape(implicitArrayCreation.Initializer),
            TupleExpressionSyntax tuple => tuple.Arguments.Any(argument => ctx.IsTensorValueExpression(argument.Expression)),
            CollectionExpressionSyntax collection => collection.Elements.OfType<ExpressionElementSyntax>().Any(element => ctx.IsTensorValueExpression(element.Expression)),
            ObjectCreationExpressionSyntax objectCreation => this.ArgumentListContainsTensorEscape(objectCreation.ArgumentList)
                || this.InitializerContainsTensorEscape(objectCreation.Initializer),
            ImplicitObjectCreationExpressionSyntax implicitObjectCreation => this.ArgumentListContainsTensorEscape(implicitObjectCreation.ArgumentList)
                || this.InitializerContainsTensorEscape(implicitObjectCreation.Initializer),
            _ => false,
        };

        private bool ArgumentListContainsTensorEscape(ArgumentListSyntax? argumentList) =>
            argumentList is not null && argumentList.Arguments.Any(argument => ctx.IsTensorValueExpression(argument.Expression));

        private bool InitializerContainsTensorEscape(InitializerExpressionSyntax? initializer)
        {
            if (initializer is null)
                return false;

            foreach (var expression in initializer.Expressions)
            {
                if (expression is AssignmentExpressionSyntax assignment && ctx.IsTensorValueExpression(assignment.Right))
                    return true;

                if (expression is InitializerExpressionSyntax nestedInitializer && this.InitializerContainsTensorEscape(nestedInitializer))
                    return true;

                if (ctx.IsTensorValueExpression(expression))
                    return true;
            }

            return false;
        }

        private static bool IsSupportedValueAssignment(AssignmentExpressionSyntax assignment) => assignment.Kind() is
            SyntaxKind.SimpleAssignmentExpression or
            SyntaxKind.AddAssignmentExpression or
            SyntaxKind.SubtractAssignmentExpression or
            SyntaxKind.MultiplyAssignmentExpression or
            SyntaxKind.DivideAssignmentExpression or
            SyntaxKind.ModuloAssignmentExpression or
            SyntaxKind.AndAssignmentExpression or
            SyntaxKind.ExclusiveOrAssignmentExpression or
            SyntaxKind.OrAssignmentExpression or
            SyntaxKind.LeftShiftAssignmentExpression or
            SyntaxKind.RightShiftAssignmentExpression or
            SyntaxKind.UnsignedRightShiftAssignmentExpression;

        private bool IsSupportedTensorAssignment(AssignmentExpressionSyntax assignment)
        {
            if (!IsSupportedValueAssignment(assignment))
                return false;

            if (!ctx.IsTensorValueExpression(assignment))
                return true;

            if (assignment.Parent is not ExpressionStatementSyntax)
                return false;

            return ctx.Model.GetSymbolInfo(assignment.Left).Symbol is ILocalSymbol local && ctx.IsTensor(local.Type);
        }

        private static bool IsSupportedTensorStatement(StatementSyntax statement) => statement is
            BlockSyntax or
            LocalDeclarationStatementSyntax or
            ExpressionStatementSyntax or
            ReturnStatementSyntax or
            IfStatementSyntax or
            WhileStatementSyntax or
            DoStatementSyntax or
            ForStatementSyntax or
            ForEachStatementSyntax or
            ForEachVariableStatementSyntax or
            SwitchStatementSyntax or
            ThrowStatementSyntax or
            EmptyStatementSyntax;

        private static bool IsLegacyUnsupportedStatement(StatementSyntax statement) => statement is
            TryStatementSyntax or
            UsingStatementSyntax or
            BreakStatementSyntax or
            ContinueStatementSyntax or
            GotoStatementSyntax or
            LocalFunctionStatementSyntax or
            YieldStatementSyntax;

        private BlockSyntax RewriteBlock(BlockSyntax block) =>
            block.WithStatements(SyntaxFactory.List(this.RewriteScopedStatements(block.Statements)));

        private List<StatementSyntax> RewriteScopedStatements(SyntaxList<StatementSyntax> statements)
        {
            this.scopes.Push([]);
            try { return this.RewriteStatementList(statements, 0); }
            finally { this.scopes.Pop(); }
        }

        private List<StatementSyntax> RewriteStatementList(SyntaxList<StatementSyntax> statements, int index)
        {
            if (index >= statements.Count)
                return [];

            if (statements[index] is LocalDeclarationStatementSyntax declaration && this.CanRewriteTensorLocalScope(declaration))
                return this.RewriteTensorLocalScope(declaration, 0, statements, index + 1).ToList();

            var rewritten = this.RewriteStatement(statements[index]).ToList();
            rewritten.AddRange(this.RewriteStatementList(statements, index + 1));
            return rewritten;
        }

        private bool CanRewriteTensorLocalScope(LocalDeclarationStatementSyntax declaration)
        {
            if (declaration.Declaration.Variables.Count == 0)
                return false;

            foreach (var variable in declaration.Declaration.Variables)
            {
                if (ctx.Model.GetDeclaredSymbol(variable) is not ILocalSymbol symbol
                    || !ctx.IsTensor(symbol.Type)
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
            var lowered = this.LowerTensorValue(variable.Initializer!.Value);
            this.scopes.Peek().Add(localName);

            var bodyStatements = this.CreateProtectedAssignmentStatements(localName, lowered, releaseActiveLocalsOnThrow: true).ToList();
            bodyStatements.AddRange(this.RewriteTensorLocalScope(declaration, variableIndex + 1, statements, nextStatementIndex));

            var localDeclaration = TensorDefaultDeclaration(localName);
            if (variableIndex == 0)
                localDeclaration = localDeclaration.WithLeadingTrivia(declaration.GetLeadingTrivia());

            return [
                localDeclaration,
                CreateProtectedCleanupStatement(bodyStatements, [ReleaseStatement(localName)]).WithTrailingTrivia(declaration.GetTrailingTrivia()),
            ];
        }

        private IEnumerable<StatementSyntax> RewriteStatement(StatementSyntax statement) => statement switch
        {
            BlockSyntax block => [this.RewriteBlock(block)],
            LocalDeclarationStatementSyntax local => this.RewriteLocalDeclaration(local),
            ExpressionStatementSyntax expression => this.RewriteExpressionStatement(expression),
            ReturnStatementSyntax @return => this.RewriteReturn(@return),
            IfStatementSyntax @if => this.RewriteIf(@if),
            WhileStatementSyntax @while => this.RewriteHeaderCheckedLoop(@while.Condition, @while, code: "MLXT0006",
                message: "Tensor expressions are not allowed in loop conditions.",
                rebuild: s => @while.WithStatement(s)),
            DoStatementSyntax @do => this.RewriteHeaderCheckedLoop(@do.Condition, @do, code: "MLXT0006",
                message: "Tensor expressions are not allowed in loop conditions.",
                rebuild: s => @do.WithStatement(s)),
            ForStatementSyntax @for => this.RewriteFor(@for),
            ForEachStatementSyntax @foreach => this.RewriteForeachLike(@foreach.Expression, @foreach, s => @foreach.WithStatement(s)),
            ForEachVariableStatementSyntax @foreach => this.RewriteForeachLike(@foreach.Expression, @foreach, s => @foreach.WithStatement(s)),
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
                if (ctx.Model.GetDeclaredSymbol(variable) is not ILocalSymbol symbol)
                {
                    rewritten.Add(statement);
                    continue;
                }

                if (!ctx.IsTensor(symbol.Type))
                {
                    rewritten.AddRange(this.RewriteNonTensorLocal(statement, variable, symbol));
                    continue;
                }

                if (variable.Initializer?.Value is null)
                {
                    diagnostics.Add(RewriteDiagnostic.Create("MLXT0005", variable, ctx.Model, "Tensor locals must be initialized explicitly."));
                    rewritten.Add(statement);
                    continue;
                }

                var lowered = this.LowerTensorValue(variable.Initializer.Value);
                rewritten.AddRange(lowered.PrefixStatements);
                rewritten.Add(TensorLocalDeclaration(variable.Identifier.Text, lowered.Expression)
                    .WithLeadingTrivia(statement.GetLeadingTrivia())
                    .WithTrailingTrivia(statement.GetTrailingTrivia()));
                rewritten.AddRange(lowered.CleanupStatements);
                this.scopes.Peek().Add(variable.Identifier.Text);
            }

            return rewritten;
        }

        private IEnumerable<StatementSyntax> RewriteNonTensorLocal(
            LocalDeclarationStatementSyntax statement,
            VariableDeclaratorSyntax variable,
            ILocalSymbol symbol)
        {
            var declarationType = statement.Declaration.Type.WithTriviaFrom(statement.Declaration.Type);

            LocalDeclarationStatementSyntax MakeDeclaration(VariableDeclaratorSyntax declarator) =>
                ApplyUsing(SyntaxFactory.LocalDeclarationStatement(
                            SyntaxFactory.VariableDeclaration(declarationType).WithVariables(SyntaxFactory.SingletonSeparatedList(declarator)))
                        .WithLeadingTrivia(statement.GetLeadingTrivia()).WithTrailingTrivia(statement.GetTrailingTrivia()),
                    statement.UsingKeyword);

            if (variable.Initializer is null)
                return [MakeDeclaration(SyntaxFactory.VariableDeclarator(variable.Identifier))];

            var lowered = this.LowerNonTensorExpression(variable.Initializer.Value);

            if (lowered.CleanupStatements.Count == 0)
            {
                var result = new List<StatementSyntax>();
                result.AddRange(lowered.PrefixStatements);
                result.Add(MakeDeclaration(SyntaxFactory.VariableDeclarator(variable.Identifier)
                    .WithInitializer(SyntaxFactory.EqualsValueClause(lowered.Expression))));
                return result;
            }

            var rewritten = new List<StatementSyntax>
            {
                ValueDefaultDeclaration(variable.Identifier.Text, symbol.Type).WithLeadingTrivia(statement.GetLeadingTrivia()),
            };
            rewritten.AddRange(this.CreateProtectedAssignmentStatements(variable.Identifier.Text, lowered, releaseActiveLocalsOnThrow: false));
            return rewritten;
        }

        private static LocalDeclarationStatementSyntax ApplyUsing(LocalDeclarationStatementSyntax declaration, SyntaxToken usingKeyword) =>
            usingKeyword.IsKind(SyntaxKind.None) ? declaration : declaration.WithUsingKeyword(usingKeyword);

        private IEnumerable<StatementSyntax> RewriteExpressionStatement(ExpressionStatementSyntax statement)
        {
            if (statement.Expression is AssignmentExpressionSyntax assignment
                && ctx.Model.GetSymbolInfo(assignment.Left).Symbol is ILocalSymbol local
                && ctx.IsTensor(local.Type))
            {
                var lowered = this.LowerTensorValue(assignment.Right);
                var assignedName = this.NextIdentifier("__mlxTensorAssigned");
                var assignedValue = lowered.Expression is ThrowExpressionSyntax
                    ? lowered
                    : lowered with { Expression = this.CreateStoredTensorExpression(lowered) };

                var statements = new List<StatementSyntax>
                {
                    TensorDefaultDeclaration(assignedName).WithLeadingTrivia(statement.GetLeadingTrivia()),
                };
                statements.AddRange(this.CreateProtectedAssignmentStatements(assignedName, assignedValue, releaseActiveLocalsOnThrow: true));
                statements.Add(ReleaseStatement(local.Name));
                statements.Add(SyntaxFactory.ExpressionStatement(
                        SyntaxFactory.AssignmentExpression(
                            SyntaxKind.SimpleAssignmentExpression,
                            SyntaxFactory.IdentifierName(local.Name),
                            TensorCompilerCall("AdoptOwned",
                                TensorCompilerCall("TakeOwned", assignedName, false),
                                byRef: false)))
                    .WithTrailingTrivia(statement.GetTrailingTrivia()));

                return statements;
            }

            var loweredExpression = this.LowerNonTensorExpression(statement.Expression);
            return this.CreateProtectedExpressionStatements(loweredExpression,
                SyntaxFactory.ExpressionStatement(loweredExpression.Expression).WithLeadingTrivia(statement.GetLeadingTrivia())
                    .WithTrailingTrivia(statement.GetTrailingTrivia()));
        }

        private IEnumerable<StatementSyntax> RewriteExpressionBody(ExpressionSyntax expression)
        {
            var lowered = this.LowerNonTensorExpression(expression);
            return this.CreateProtectedExpressionStatements(lowered, SyntaxFactory.ExpressionStatement(lowered.Expression));
        }

        private IEnumerable<StatementSyntax> RewriteReturn(ReturnStatementSyntax statement)
        {
            if (statement.Expression is null)
            {
                var result = this.ReleaseActiveLocals().ToList();
                result.Add(statement);
                return result;
            }

            return this.RewriteReturnValue(statement.Expression, statement.GetLeadingTrivia(), statement.GetTrailingTrivia(), fallback: () => [statement]);
        }

        private IEnumerable<StatementSyntax> RewriteReturnExpression(ExpressionSyntax expression) =>
            this.RewriteReturnValue(expression, default, default, fallback: () => [SyntaxFactory.ReturnStatement(expression)]);

        private IEnumerable<StatementSyntax> RewriteReturnValue(
            ExpressionSyntax expression,
            SyntaxTriviaList leadingTrivia,
            SyntaxTriviaList trailingTrivia,
            Func<IEnumerable<StatementSyntax>> fallback)
        {
            if (ctx.IsTensorValueExpression(expression))
            {
                var lowered = this.LowerExpression(expression);

                if (!lowered.IsTensor || lowered.OriginKind == TensorOriginKind.None || lowered.Identifier is null)
                {
                    diagnostics.Add(RewriteDiagnostic.Create("MLXT0010", expression, ctx.Model, "Tensor expression could not be lowered."));
                    return fallback();
                }

                var returnHandleName = this.NextIdentifier("__mlxTensorReturnHandle");
                var statements = new List<StatementSyntax> { HandleDefaultDeclaration(returnHandleName) };
                statements.AddRange(this.CreateProtectedAssignmentStatements(returnHandleName,
                    lowered with { Expression = this.CreateReturnHandleExpression(lowered) },
                    releaseActiveLocalsOnThrow: false));
                statements.AddRange(this.ReleaseActiveLocals());
                statements.Add(SyntaxFactory.ReturnStatement(TensorCompilerCall("AdoptOwned", returnHandleName, byRef: false))
                    .WithLeadingTrivia(leadingTrivia).WithTrailingTrivia(trailingTrivia));
                return statements;
            }

            var loweredScalar = this.LowerNonTensorExpression(expression);
            var returnValueName = this.NextIdentifier("__mlxReturnValue");
            var rewritten = new List<StatementSyntax>
            {
                ValueDefaultDeclaration(returnValueName, this.currentMethodReturnType ?? ctx.Model.Compilation.ObjectType),
            };
            rewritten.AddRange(this.CreateProtectedAssignmentStatements(returnValueName, loweredScalar, releaseActiveLocalsOnThrow: true));
            rewritten.AddRange(this.ReleaseActiveLocals());
            rewritten.Add(SyntaxFactory.ReturnStatement(SyntaxFactory.IdentifierName(returnValueName))
                .WithLeadingTrivia(leadingTrivia).WithTrailingTrivia(trailingTrivia));
            return rewritten;
        }

        private IEnumerable<StatementSyntax> RewriteIf(IfStatementSyntax statement)
        {
            var condition = this.LowerCondition(statement.Condition);
            var statements = new List<StatementSyntax>(condition.PrefixStatements);
            statements.Add(SyntaxFactory.IfStatement(
                    condition.Expression,
                    this.RewriteEmbedded(statement.Statement),
                    statement.Else is null ? null : SyntaxFactory.ElseClause(this.RewriteEmbedded(statement.Else.Statement)))
                .WithLeadingTrivia(statement.GetLeadingTrivia()).WithTrailingTrivia(statement.GetTrailingTrivia()));
            return statements;
        }

        private IEnumerable<StatementSyntax> RewriteHeaderCheckedLoop<T>(
            ExpressionSyntax condition,
            T statement,
            string code,
            string message,
            Func<StatementSyntax, T> rebuild) where T : StatementSyntax
        {
            if (this.ContainsMaterialTensorExpression(condition))
            {
                diagnostics.Add(RewriteDiagnostic.Create(code, condition, ctx.Model, message));
                return [statement];
            }

            return [(StatementSyntax)rebuild(this.RewriteEmbedded(GetEmbeddedBody(statement)))];
        }

        private static StatementSyntax GetEmbeddedBody(StatementSyntax statement) => statement switch
        {
            WhileStatementSyntax w => w.Statement,
            DoStatementSyntax d => d.Statement,
            ForStatementSyntax f => f.Statement,
            ForEachStatementSyntax fe => fe.Statement,
            ForEachVariableStatementSyntax fev => fev.Statement,
            _ => throw new ArgumentException($"Unsupported loop statement: {statement.Kind()}", nameof(statement)),
        };

        private IEnumerable<StatementSyntax> RewriteFor(ForStatementSyntax statement)
        {
            if (statement.Initializers.Any(this.ContainsMaterialTensorExpression)
                || statement.Incrementors.Any(this.ContainsMaterialTensorExpression)
                || (statement.Condition is not null && this.ContainsMaterialTensorExpression(statement.Condition)))
            {
                diagnostics.Add(RewriteDiagnostic.Create("MLXT0006", statement, ctx.Model, "Tensor expressions are not allowed in for-loop headers."));
                return [statement];
            }

            return [statement.WithStatement(this.RewriteEmbedded(statement.Statement))];
        }

        private IEnumerable<StatementSyntax> RewriteForeachLike(ExpressionSyntax expression, StatementSyntax statement, Func<StatementSyntax, StatementSyntax> rebuild)
        {
            if (this.ContainsMaterialTensorExpression(expression))
            {
                diagnostics.Add(RewriteDiagnostic.Create("MLXT0007", expression, ctx.Model, "Tensor cannot be enumerated or stored in collections."));
                return [statement];
            }

            return [rebuild(this.RewriteEmbedded(GetEmbeddedBody(statement)))];
        }

        private IEnumerable<StatementSyntax> RewriteSwitch(SwitchStatementSyntax statement)
        {
            if (this.ContainsMaterialTensorExpression(statement.Expression))
            {
                diagnostics.Add(RewriteDiagnostic.Create("MLXT0006", statement.Expression, ctx.Model, "Tensor expressions are not allowed in switch headers."));
                return [statement];
            }

            var rewrittenSections = statement.Sections
                .Select(section => section.WithStatements(SyntaxFactory.List(this.RewriteScopedStatements(section.Statements))));
            return [statement.WithSections(SyntaxFactory.List(rewrittenSections))];
        }

        private IEnumerable<StatementSyntax> RewriteThrow(ThrowStatementSyntax statement)
        {
            if (statement.Expression is null)
            {
                var result = this.ReleaseActiveLocals().ToList();
                result.Add(statement);
                return result;
            }

            var lowered = this.LowerNonTensorExpression(statement.Expression);
            var exceptionName = this.NextIdentifier("__mlxException");
            var exceptionType = ctx.Model.GetTypeInfo(statement.Expression).ConvertedType
                ?? ctx.Model.GetTypeInfo(statement.Expression).Type
                ?? ctx.Model.Compilation.GetTypeByMetadataName("System.Exception")
                ?? ctx.Model.Compilation.ObjectType;

            var rewritten = new List<StatementSyntax> { ValueDefaultDeclaration(exceptionName, exceptionType) };
            rewritten.AddRange(this.CreateProtectedAssignmentStatements(exceptionName, lowered, releaseActiveLocalsOnThrow: true));
            rewritten.AddRange(this.ReleaseActiveLocals());
            rewritten.Add(SyntaxFactory.ThrowStatement(SyntaxFactory.IdentifierName(exceptionName))
                .WithLeadingTrivia(statement.GetLeadingTrivia()).WithTrailingTrivia(statement.GetTrailingTrivia()));
            return rewritten;
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
            finally { this.scopes.Pop(); }
        }

        private LoweredExpression LowerCondition(ExpressionSyntax expression)
        {
            var lowered = this.LowerNonTensorExpression(expression);

            if (lowered.CleanupStatements.Count == 0)
                return lowered;

            var conditionName = this.NextIdentifier("__mlxCondition");
            var prefix = new List<StatementSyntax>
            {
                ValueDefaultDeclaration(conditionName, ctx.Model.Compilation.GetSpecialType(SpecialType.System_Boolean)),
            };
            prefix.AddRange(this.CreateProtectedAssignmentStatements(conditionName, lowered, releaseActiveLocalsOnThrow: true));
            return new LoweredExpression(prefix, SyntaxFactory.IdentifierName(conditionName), [], false, TensorOriginKind.None, null);
        }

        private LoweredExpression LowerTensorValue(ExpressionSyntax expression)
        {
            var lowered = this.LowerExpression(expression);

            if (lowered.Expression is ThrowExpressionSyntax)
                return lowered;

            if (!lowered.IsTensor || lowered.OriginKind == TensorOriginKind.None || lowered.Identifier is null)
            {
                diagnostics.Add(RewriteDiagnostic.Create("MLXT0010", expression, ctx.Model, "Tensor expression could not be lowered."));
                return lowered;
            }

            return lowered with { Expression = this.CreateStoredTensorExpression(lowered) };
        }

        private LoweredExpression LowerNonTensorExpression(ExpressionSyntax expression)
        {
            var lowered = this.LowerExpression(expression);

            if (lowered.IsTensor)
                diagnostics.Add(RewriteDiagnostic.Create("MLXT0011", expression, ctx.Model,
                    "Tensor value cannot be used where a scalar/value expression is required."));

            return lowered;
        }

        private LoweredExpression LowerExpression(ExpressionSyntax expression) =>
            ctx.IsTensorValueExpression(expression) ? this.LowerTensorExpression(expression) : this.LowerValueExpression(expression);

        private LoweredExpression LowerTensorExpression(ExpressionSyntax expression)
        {
            if (expression is IdentifierNameSyntax identifier
                && ctx.Model.GetSymbolInfo(identifier).Symbol is ISymbol symbol
                && ctx.IsTensor(TensorContext.GetSymbolType(symbol)))
            {
                var origin = symbol.Kind switch
                {
                    SymbolKind.Parameter => TensorOriginKind.Parameter,
                    SymbolKind.Local => TensorOriginKind.Local,
                    _ => TensorOriginKind.None,
                };
                var name = origin == TensorOriginKind.None ? null : identifier.Identifier.Text;
                return new LoweredExpression([], identifier, [], true, origin, name);
            }

            if (expression is ParenthesizedExpressionSyntax parenthesized)
                return this.LowerTensorExpression(parenthesized.Expression);

            if (expression is ThrowExpressionSyntax throwExpression)
            {
                var loweredThrow = this.LowerNonTensorExpression(throwExpression.Expression);
                return new LoweredExpression(loweredThrow.PrefixStatements, SyntaxFactory.ThrowExpression(loweredThrow.Expression),
                    loweredThrow.CleanupStatements, true, TensorOriginKind.None, null);
            }

            if (expression is ConditionalExpressionSyntax conditional)
                return this.LowerTensorConditional(conditional);

            var children = GetDirectSubExpressions(expression).Select(this.LowerExpression).ToArray();
            var prefix = new List<StatementSyntax>();
            foreach (var child in children)
                prefix.AddRange(child.PrefixStatements);

            var rewrittenExpression = ReplaceDirectSubExpressions(expression, children.Select(child => child.Expression).ToArray());
            var tempName = this.NextIdentifier("__mlxTensorTemp");
            prefix.Add(TensorLocalDeclaration(tempName, rewrittenExpression));

            var cleanup = new List<StatementSyntax> { ReleaseStatement(tempName) };
            cleanup.AddRange(children.SelectMany(child => child.CleanupStatements));

            return new LoweredExpression(prefix, SyntaxFactory.IdentifierName(tempName), cleanup, true, TensorOriginKind.Temp, tempName);
        }

        private LoweredExpression LowerTensorConditional(ConditionalExpressionSyntax conditional)
        {
            var condition = this.LowerCondition(conditional.Condition);
            var tempName = this.NextIdentifier("__mlxTensorTemp");
            var trueValue = this.LowerTensorValue(conditional.WhenTrue);
            var falseValue = this.LowerTensorValue(conditional.WhenFalse);

            var prefix = new List<StatementSyntax>(condition.PrefixStatements) { TensorDefaultDeclaration(tempName) };
            prefix.Add(SyntaxFactory.IfStatement(
                condition.Expression,
                SyntaxFactory.Block(this.CreateProtectedAssignmentStatements(tempName, trueValue, releaseActiveLocalsOnThrow: true)),
                SyntaxFactory.ElseClause(SyntaxFactory.Block(this.CreateProtectedAssignmentStatements(tempName, falseValue, releaseActiveLocalsOnThrow: true)))));

            return new LoweredExpression(prefix, SyntaxFactory.IdentifierName(tempName), [ReleaseStatement(tempName)], true, TensorOriginKind.Temp, tempName);
        }

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
            var tempType = ctx.Model.GetTypeInfo(conditional).ConvertedType ?? ctx.Model.GetTypeInfo(conditional).Type;

            if (tempType is null)
                return new LoweredExpression(condition.PrefixStatements, conditional, [], false, TensorOriginKind.None, null);

            var tempName = this.NextIdentifier("__mlxValueTemp");
            var prefix = new List<StatementSyntax>(condition.PrefixStatements) { ValueDefaultDeclaration(tempName, tempType) };
            prefix.Add(SyntaxFactory.IfStatement(
                condition.Expression,
                SyntaxFactory.Block(this.CreateProtectedAssignmentStatements(tempName, trueValue, releaseActiveLocalsOnThrow: true)),
                SyntaxFactory.ElseClause(SyntaxFactory.Block(this.CreateProtectedAssignmentStatements(tempName, falseValue, releaseActiveLocalsOnThrow: true)))));

            return new LoweredExpression(prefix, SyntaxFactory.IdentifierName(tempName), [], false, TensorOriginKind.None, null);
        }

        private bool ContainsAnyTensorExpression(SyntaxNode node) =>
            node.DescendantNodesAndSelf().OfType<ExpressionSyntax>().Any(ctx.IsTensorValueExpression);

        private bool ContainsMaterialTensorExpression(ExpressionSyntax expression)
        {
            if (ctx.IsTensorValueExpression(expression))
                return true;

            return expression.DescendantNodesAndSelf().OfType<ExpressionSyntax>().Any(this.IsMaterialTensorExpression);
        }

        private bool IsMaterialTensorExpression(ExpressionSyntax expression)
        {
            if (!ctx.IsTensorValueExpression(expression))
                return false;

            return expression.Parent switch
            {
                MemberAccessExpressionSyntax member when member.Expression == expression => this.MemberAccessProducesTensor(member),
                ElementAccessExpressionSyntax element when element.Expression == expression => ctx.IsTensorValueExpression(element),
                _ => true,
            };
        }

        private bool MemberAccessProducesTensor(MemberAccessExpressionSyntax member)
        {
            if (ctx.IsTensorValueExpression(member))
                return true;

            if (!ReferenceEquals(member.SyntaxTree, ctx.Model.SyntaxTree))
                return false;

            try
            {
                return ctx.Model.GetSymbolInfo(member).Symbol switch
                {
                    IPropertySymbol property => ctx.ContainsTensor(property.Type),
                    IFieldSymbol field => ctx.ContainsTensor(field.Type),
                    IMethodSymbol method => ctx.ContainsTensor(method.ReturnType),
                    _ => false,
                };
            }
            catch (ArgumentException)
            {
                return false;
            }
        }

        private IReadOnlyList<StatementSyntax> ReleaseActiveLocals() =>
            this.scopes.SelectMany(scope => scope).Reverse().Select(ReleaseStatement).ToArray();

        private string NextIdentifier(string prefix) => prefix + this.tempId++.ToString(CultureInfo.InvariantCulture);

        private ExpressionSyntax CreateStoredTensorExpression(LoweredExpression lowered) => lowered.OriginKind switch
        {
            TensorOriginKind.Parameter or TensorOriginKind.Local => TensorCompilerCall(
                "AdoptOwned",
                TensorCompilerCall("RetainBorrowed", lowered.Identifier!, false, false),
                byRef: false),
            TensorOriginKind.Temp => TensorCompilerCall("AdoptOwned", TensorCompilerCall("TakeOwned", lowered.Identifier!, false), byRef: false),
            _ => lowered.Expression,
        };

        private ExpressionSyntax CreateReturnHandleExpression(LoweredExpression lowered) => lowered.OriginKind switch
        {
            TensorOriginKind.Parameter => TensorCompilerCall("RetainBorrowed", lowered.Identifier!, false, false),
            TensorOriginKind.Local or TensorOriginKind.Temp => TensorCompilerCall("TakeOwned", lowered.Identifier!, false),
            _ => lowered.Expression,
        };

        private static LocalDeclarationStatementSyntax LocalDeclaration(TypeSyntax type, string name, ExpressionSyntax? initializer)
        {
            var declarator = SyntaxFactory.VariableDeclarator(SyntaxFactory.Identifier(name));
            if (initializer is not null)
                declarator = declarator.WithInitializer(SyntaxFactory.EqualsValueClause(initializer));

            return SyntaxFactory.LocalDeclarationStatement(
                SyntaxFactory.VariableDeclaration(type).WithVariables(SyntaxFactory.SingletonSeparatedList(declarator)));
        }

        private static LocalDeclarationStatementSyntax TensorLocalDeclaration(string name, ExpressionSyntax initializer) =>
            LocalDeclaration(SyntaxFactory.IdentifierName("Tensor"), name, initializer);

        private static LocalDeclarationStatementSyntax TensorDefaultDeclaration(string name) =>
            TensorLocalDeclaration(name, SyntaxFactory.LiteralExpression(SyntaxKind.DefaultLiteralExpression));

        private static LocalDeclarationStatementSyntax HandleDefaultDeclaration(string name) =>
            LocalDeclaration(SyntaxFactory.ParseTypeName("global::Itexoft.Mlx.MlxArrayHandle"), name,
                SyntaxFactory.LiteralExpression(SyntaxKind.DefaultLiteralExpression));

        private static LocalDeclarationStatementSyntax ValueDefaultDeclaration(string name, ITypeSymbol type) =>
            LocalDeclaration(SyntaxFactory.ParseTypeName(type.ToDisplayString(nullableFullyQualifiedFormat)), name,
                SyntaxFactory.PostfixUnaryExpression(SyntaxKind.SuppressNullableWarningExpression,
                    SyntaxFactory.LiteralExpression(SyntaxKind.DefaultLiteralExpression)));

        private static AccessorListSyntax CreateGetterAccessorList(BlockSyntax getterBody) =>
            SyntaxFactory.AccessorList(SyntaxFactory.SingletonList(
                SyntaxFactory.AccessorDeclaration(SyntaxKind.GetAccessorDeclaration).WithBody(getterBody)));

        private IEnumerable<StatementSyntax> CreateProtectedExpressionStatements(LoweredExpression lowered, StatementSyntax body)
        {
            var statements = new List<StatementSyntax>(lowered.PrefixStatements);
            statements.AddRange(WrapWithCleanup([body], lowered.CleanupStatements));
            return statements;
        }

        private IEnumerable<StatementSyntax> CreateProtectedAssignmentStatements(string targetName, LoweredExpression lowered, bool releaseActiveLocalsOnThrow)
        {
            var statements = new List<StatementSyntax>(lowered.PrefixStatements);
            var bodyStatements = new List<StatementSyntax>();

            if (lowered.Expression is ThrowExpressionSyntax throwExpression)
            {
                if (releaseActiveLocalsOnThrow)
                    bodyStatements.AddRange(this.ReleaseActiveLocals());
                bodyStatements.Add(SyntaxFactory.ThrowStatement(throwExpression.Expression));
            }
            else
            {
                bodyStatements.Add(SyntaxFactory.ExpressionStatement(
                    SyntaxFactory.AssignmentExpression(SyntaxKind.SimpleAssignmentExpression,
                        SyntaxFactory.IdentifierName(targetName), lowered.Expression)));
            }

            statements.AddRange(WrapWithCleanup(bodyStatements, lowered.CleanupStatements));
            return statements;
        }

        private static IEnumerable<StatementSyntax> WrapWithCleanup(IReadOnlyList<StatementSyntax> bodyStatements, IReadOnlyList<StatementSyntax> cleanupStatements) =>
            cleanupStatements.Count == 0 ? bodyStatements : [CreateProtectedCleanupStatement(bodyStatements, cleanupStatements)];

        private static TryStatementSyntax CreateProtectedCleanupStatement(IReadOnlyList<StatementSyntax> bodyStatements, IReadOnlyList<StatementSyntax> cleanupStatements) =>
            SyntaxFactory.TryStatement(SyntaxFactory.Block(bodyStatements), SyntaxFactory.List<CatchClauseSyntax>(),
                SyntaxFactory.FinallyClause(SyntaxFactory.Block(cleanupStatements)));

        private static StatementSyntax ReleaseStatement(string name) =>
            SyntaxFactory.ExpressionStatement(TensorCompilerCall("Release", name, false));

        private static InvocationExpressionSyntax TensorCompilerCall(string method, string name, bool returnsTensor = true, bool byRef = true) =>
            TensorCompilerCall(method, SyntaxFactory.IdentifierName(name), returnsTensor, byRef);

        private static InvocationExpressionSyntax TensorCompilerCall(string method, ExpressionSyntax argumentExpression, bool returnsTensor = true, bool byRef = true)
        {
            var member = SyntaxFactory.MemberAccessExpression(SyntaxKind.SimpleMemberAccessExpression,
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
            CheckedExpressionSyntax @checked => [@checked.Expression],
            AssignmentExpressionSyntax assignment => [assignment.Left, assignment.Right],
            ArrayCreationExpressionSyntax arrayCreation => [.. EnumerateArrayCreationExpressions(arrayCreation)],
            ImplicitArrayCreationExpressionSyntax implicitArrayCreation => [.. implicitArrayCreation.Initializer.Expressions],
            TupleExpressionSyntax tuple => [.. tuple.Arguments.Select(argument => argument.Expression)],
            MemberAccessExpressionSyntax member => [member.Expression],
            InvocationExpressionSyntax invocation => [invocation.Expression, .. invocation.ArgumentList.Arguments.Select(argument => argument.Expression)],
            ElementAccessExpressionSyntax element => [element.Expression, .. element.ArgumentList.Arguments.Select(argument => argument.Expression)],
            ElementBindingExpressionSyntax elementBinding => [.. elementBinding.ArgumentList.Arguments.Select(argument => argument.Expression)],
            ConditionalAccessExpressionSyntax conditionalAccess => [conditionalAccess.Expression, conditionalAccess.WhenNotNull],
            CastExpressionSyntax cast => [cast.Expression],
            RangeExpressionSyntax range => [.. EnumerateMaybe(range.LeftOperand), .. EnumerateMaybe(range.RightOperand)],
            IsPatternExpressionSyntax isPattern => [isPattern.Expression],
            ObjectCreationExpressionSyntax objectCreation =>
            [
                .. objectCreation.ArgumentList?.Arguments.Select(argument => argument.Expression) ?? [],
                .. objectCreation.Initializer?.Expressions ?? [],
            ],
            ImplicitObjectCreationExpressionSyntax implicitObjectCreation =>
            [
                .. implicitObjectCreation.ArgumentList?.Arguments.Select(argument => argument.Expression) ?? [],
                .. implicitObjectCreation.Initializer?.Expressions ?? [],
            ],
            CollectionExpressionSyntax collection => [.. collection.Elements.OfType<ExpressionElementSyntax>().Select(element => element.Expression)],
            InitializerExpressionSyntax initializer => [.. initializer.Expressions],
            InterpolatedStringExpressionSyntax interpolatedString => [.. EnumerateInterpolatedExpressions(interpolatedString)],
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
            CheckedExpressionSyntax @checked => @checked.WithExpression(replacements[0]),
            AssignmentExpressionSyntax assignment => assignment.WithLeft(replacements[0]).WithRight(replacements[1]),
            ArrayCreationExpressionSyntax arrayCreation => ReplaceArrayCreation(arrayCreation, replacements),
            ImplicitArrayCreationExpressionSyntax implicitArrayCreation => implicitArrayCreation.WithInitializer(
                implicitArrayCreation.Initializer.WithExpressions(SyntaxFactory.SeparatedList(replacements))),
            TupleExpressionSyntax tuple => tuple.WithArguments(SyntaxFactory.SeparatedList(
                tuple.Arguments.Select((argument, index) => argument.WithExpression(replacements[index])))),
            MemberAccessExpressionSyntax member => member.WithExpression(replacements[0]),
            InvocationExpressionSyntax invocation => invocation.WithExpression(replacements[0])
                .WithArgumentList(invocation.ArgumentList.WithArguments(SyntaxFactory.SeparatedList(
                    invocation.ArgumentList.Arguments.Select((argument, index) => argument.WithExpression(replacements[index + 1]))))),
            ElementAccessExpressionSyntax element => element.WithExpression(replacements[0])
                .WithArgumentList(element.ArgumentList.WithArguments(SyntaxFactory.SeparatedList(
                    element.ArgumentList.Arguments.Select((argument, index) => argument.WithExpression(replacements[index + 1]))))),
            ElementBindingExpressionSyntax elementBinding => elementBinding.WithArgumentList(
                elementBinding.ArgumentList.WithArguments(SyntaxFactory.SeparatedList(
                    elementBinding.ArgumentList.Arguments.Select((argument, index) => argument.WithExpression(replacements[index]))))),
            ConditionalAccessExpressionSyntax conditionalAccess => conditionalAccess.WithExpression(replacements[0]).WithWhenNotNull(replacements[1]),
            CastExpressionSyntax cast => cast.WithExpression(replacements[0]),
            RangeExpressionSyntax range => ReplaceRange(range, replacements),
            IsPatternExpressionSyntax isPattern => isPattern.WithExpression(replacements[0]),
            ObjectCreationExpressionSyntax objectCreation => ReplaceObjectCreationLike(objectCreation, replacements),
            ImplicitObjectCreationExpressionSyntax implicitObjectCreation => ReplaceObjectCreationLike(implicitObjectCreation, replacements),
            CollectionExpressionSyntax collection => ReplaceCollection(collection, replacements),
            InitializerExpressionSyntax initializer => initializer.WithExpressions(SyntaxFactory.SeparatedList(replacements)),
            InterpolatedStringExpressionSyntax interpolatedString => ReplaceInterpolatedString(interpolatedString, replacements),
            ConditionalExpressionSyntax conditional => conditional.WithCondition(replacements[0]).WithWhenTrue(replacements[1]).WithWhenFalse(replacements[2]),
            _ => expression,
        };

        private static ExpressionSyntax ReplaceRange(RangeExpressionSyntax range, ExpressionSyntax[] replacements)
        {
            var index = 0;
            var left = range.LeftOperand is null ? null : replacements[index++];
            var right = range.RightOperand is null ? null : replacements[index];
            return range.WithLeftOperand(left).WithRightOperand(right);
        }

        private static ExpressionSyntax ReplaceCollection(CollectionExpressionSyntax collection, ExpressionSyntax[] replacements)
        {
            var index = 0;
            var elements = collection.Elements.Select(element =>
                element is ExpressionElementSyntax expressionElement ? (CollectionElementSyntax)expressionElement.WithExpression(replacements[index++]) : element);
            return collection.WithElements(SyntaxFactory.SeparatedList(elements));
        }

        private static T ReplaceObjectCreationLike<T>(T node, ExpressionSyntax[] replacements) where T : BaseObjectCreationExpressionSyntax
        {
            var index = 0;
            var rewrittenArguments = node.ArgumentList;
            if (rewrittenArguments is not null)
            {
                rewrittenArguments = rewrittenArguments.WithArguments(SyntaxFactory.SeparatedList(
                    rewrittenArguments.Arguments.Select(argument => argument.WithExpression(replacements[index++]))));
            }

            var rewrittenInitializer = node.Initializer;
            if (rewrittenInitializer is not null)
            {
                rewrittenInitializer = rewrittenInitializer.WithExpressions(SyntaxFactory.SeparatedList(
                    rewrittenInitializer.Expressions.Select(_ => replacements[index++])));
            }

            return (T)node.WithArgumentList(rewrittenArguments!).WithInitializer(rewrittenInitializer);
        }

        private static ExpressionSyntax ReplaceInterpolatedString(InterpolatedStringExpressionSyntax interpolatedString, ExpressionSyntax[] replacements)
        {
            var index = 0;
            var contents = interpolatedString.Contents.Select(content =>
            {
                if (content is not InterpolationSyntax interpolation)
                    return content;

                var rewritten = interpolation.WithExpression(replacements[index++]);
                if (interpolation.AlignmentClause is not null)
                    rewritten = rewritten.WithAlignmentClause(interpolation.AlignmentClause.WithValue(replacements[index++]));
                return (InterpolatedStringContentSyntax)rewritten;
            });
            return interpolatedString.WithContents(SyntaxFactory.List(contents));
        }

        private static IEnumerable<ExpressionSyntax> EnumerateArrayCreationExpressions(ArrayCreationExpressionSyntax arrayCreation)
        {
            foreach (var rank in arrayCreation.Type.RankSpecifiers)
            {
                foreach (var size in rank.Sizes)
                {
                    if (!size.IsKind(SyntaxKind.OmittedArraySizeExpression))
                        yield return size;
                }
            }

            if (arrayCreation.Initializer is not null)
            {
                foreach (var expression in arrayCreation.Initializer.Expressions)
                    yield return expression;
            }
        }

        private static IEnumerable<ExpressionSyntax> EnumerateInterpolatedExpressions(InterpolatedStringExpressionSyntax interpolatedString)
        {
            foreach (var content in interpolatedString.Contents.OfType<InterpolationSyntax>())
            {
                yield return content.Expression;
                if (content.AlignmentClause is not null)
                    yield return content.AlignmentClause.Value;
            }
        }

        private static ExpressionSyntax ReplaceArrayCreation(ArrayCreationExpressionSyntax arrayCreation, ExpressionSyntax[] replacements)
        {
            var index = 0;
            var rewrittenRanks = arrayCreation.Type.RankSpecifiers.Select(rank =>
            {
                var rewrittenSizes = rank.Sizes.Select(size =>
                    size.IsKind(SyntaxKind.OmittedArraySizeExpression) ? size : replacements[index++]);
                return rank.WithSizes(SyntaxFactory.SeparatedList(rewrittenSizes));
            });

            var rewrittenType = arrayCreation.Type.WithRankSpecifiers(SyntaxFactory.List(rewrittenRanks));
            var rewrittenInitializer = arrayCreation.Initializer;
            if (rewrittenInitializer is not null)
            {
                rewrittenInitializer = rewrittenInitializer.WithExpressions(SyntaxFactory.SeparatedList(
                    rewrittenInitializer.Expressions.Select(_ => replacements[index++])));
            }

            return arrayCreation.WithType(rewrittenType).WithInitializer(rewrittenInitializer);
        }
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
