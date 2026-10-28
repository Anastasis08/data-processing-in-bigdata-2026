using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using Lab4.Model;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;

namespace Lab4.Cli;

internal static class ModelTrainer
{
    public const int Seed = 42;
    private const double TestFraction = 0.2;
    private const int FoldCount = 5;
    private const string BaselineName = "Prior (baseline)";

    public static void TrainAndExport(MLContext mlContext, string dataPath, string modelPath, string onnxPath)
    {
        ArgumentNullException.ThrowIfNull(mlContext);

        var applications = LoanCsvReader.Read(dataPath, requireLabel: true);
        var samples = applications.Select(FeatureBuilder.BuildTrainingRow).ToArray();
        var (trainingSamples, testSamples) = StratifiedSplit(samples, TestFraction, Seed);
        var trainingData = mlContext.Data.LoadFromEnumerable(trainingSamples);
        var testData = mlContext.Data.LoadFromEnumerable(testSamples);
        var featureSchemaData = mlContext.Data.LoadFromEnumerable(Array.Empty<LoanFeatures>());

        Console.WriteLine("Part 1: training, evaluation, ONNX export");
        Console.WriteLine(
            $"Data: {Path.GetFileName(dataPath)} ({samples.Length} applications, " +
            $"approved {samples.Count(static sample => sample.Label)}, " +
            $"rejected {samples.Count(static sample => !sample.Label)})");
        Console.WriteLine($"Stratified split: {trainingSamples.Count} training / {testSamples.Count} test samples");
        Console.WriteLine();
        Console.WriteLine($"{FoldCount}-fold cross-validation on the training set (mean ± std) and ONNX support:");
        Console.WriteLine($"{"Trainer",-26} {"AUC",-15} {"Accuracy",-15} {"F1",-15} {"LogLoss",-8} {"ONNX",-4}");

        var results = CreateCandidates(mlContext)
            .Select(candidate => Evaluate(mlContext, trainingData, featureSchemaData, candidate))
            .ToArray();

        foreach (var result in results)
        {
            Console.WriteLine(
                $"{result.Name,-26} {Format(result.AreaUnderRocCurve),-15} {Format(result.Accuracy),-15} " +
                $"{Format(result.F1Score),-15} {result.LogLoss.Mean,-8:F4} {result.OnnxExportable switch { true => "yes", false => "no", null => "-" },-4}");
        }

        var bestOverall = results
            .Where(static result => result.Name != BaselineName)
            .MaxBy(static result => result.AreaUnderRocCurve.Mean)!;
        var best = results
            .Where(static result => result.OnnxExportable == true)
            .MaxBy(static result => result.AreaUnderRocCurve.Mean)
            ?? throw new InvalidOperationException("None of the trainers can be exported to ONNX.");
        var bestCandidate = CreateCandidates(mlContext).Single(candidate => candidate.Name == best.Name);

        Console.WriteLine();
        if (bestOverall.Name != best.Name)
        {
            Console.WriteLine(
                $"Best by AUC is {bestOverall.Name} ({bestOverall.AreaUnderRocCurve.Mean:F4}), " +
                "but it cannot be exported to ONNX.");
        }

        Console.WriteLine($"Selected by mean cross-validated AUC among ONNX-exportable trainers: {best.Name}");

        var model = bestCandidate.Pipeline.Fit(trainingData);
        var testMetrics = mlContext.BinaryClassification.Evaluate(model.Transform(testData));
        var baselineModel = CreateCandidates(mlContext).Single(static candidate => candidate.Name == BaselineName)
            .Pipeline.Fit(trainingData);
        var baselineMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(baselineModel.Transform(testData));

        PrintTestMetrics(testMetrics, baselineMetrics);

        var fullData = mlContext.Data.LoadFromEnumerable(samples);
        var finalModel = bestCandidate.Pipeline.Fit(fullData);
        var modelDirectory = Path.GetDirectoryName(modelPath)
            ?? throw new InvalidOperationException("Unable to determine the model directory.");
        Directory.CreateDirectory(modelDirectory);
        mlContext.Model.Save(finalModel, featureSchemaData.Schema, modelPath);

        OnnxExporter.Export(mlContext, finalModel, featureSchemaData, onnxPath);

        var onnxModel = OnnxLoanModel.Load(mlContext, onnxPath);
        var parity = ParityChecker.Compare(mlContext, finalModel, onnxModel, testSamples);

        var report = new TrainingReport(
            CreatedAtUtc: DateTimeOffset.UtcNow,
            DataSha256: Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(dataPath))),
            Seed: Seed,
            TrainingRows: trainingSamples.Count,
            TestRows: testSamples.Count,
            SerializedModelRows: samples.Length,
            SelectedTrainer: best.Name,
            BestTrainerOverall: bestOverall.Name,
            SelectionMetric: "Mean cross-validated AUC among ONNX-exportable trainers",
            CrossValidation: results,
            Test: TestMetrics.From(testMetrics),
            BaselineTestAccuracy: baselineMetrics.Accuracy,
            OnnxParity: parity);

        var reportPath = Path.ChangeExtension(modelPath, ".metrics.json");
        File.WriteAllText(reportPath, JsonSerializer.Serialize(report, TrainingReportJsonContext.Default.TrainingReport));

        Console.WriteLine($"Final model refit on all {samples.Length} rows.");
        Console.WriteLine($"ML.NET model saved: {modelPath} ({new FileInfo(modelPath).Length / 1024.0:F1} KB)");
        Console.WriteLine($"ONNX model exported: {onnxPath} ({new FileInfo(onnxPath).Length / 1024.0:F1} KB)");
        Console.WriteLine(
            $"ONNX vs ML.NET on {parity.Rows} test rows: max |dP| = {parity.MaxProbabilityDifference:E2}, " +
            $"same decisions {parity.MatchingDecisions}/{parity.Rows}");
        Console.WriteLine($"Metrics saved: {reportPath}");
        Console.WriteLine();
    }

    private static IEnumerable<Candidate> CreateCandidates(MLContext mlContext)
    {
        var trainers = mlContext.BinaryClassification.Trainers;

        yield return new(BaselineName, Preprocessing(mlContext).Append(trainers.Prior()));

        yield return new("SdcaLogisticRegression", Preprocessing(mlContext).Append(
            trainers.SdcaLogisticRegression(new SdcaLogisticRegressionBinaryTrainer.Options
            {
                L2Regularization = 0.1f,
                MaximumNumberOfIterations = 100,
                NumberOfThreads = 1
            })));

        yield return new("LbfgsLogisticRegression", Preprocessing(mlContext).Append(
            trainers.LbfgsLogisticRegression(new LbfgsLogisticRegressionBinaryTrainer.Options
            {
                L1Regularization = 0.01f,
                L2Regularization = 0.1f
            })));

        yield return new("FastTree", Preprocessing(mlContext).Append(
            trainers.FastTree(new FastTreeBinaryTrainer.Options
            {
                NumberOfLeaves = 8,
                NumberOfTrees = 100,
                MinimumExampleCountPerLeaf = 10,
                LearningRate = 0.05,
                NumberOfThreads = 1
            })));

        yield return new("FastForest", Preprocessing(mlContext)
            .Append(trainers.FastForest(new FastForestBinaryTrainer.Options
            {
                NumberOfLeaves = 16,
                NumberOfTrees = 200,
                MinimumExampleCountPerLeaf = 5,
                NumberOfThreads = 1
            }))
            .Append(mlContext.BinaryClassification.Calibrators.Platt()));

        yield return new("LightGbm", Preprocessing(mlContext).Append(
            trainers.LightGbm(new LightGbmBinaryTrainer.Options
            {
                NumberOfLeaves = 8,
                NumberOfIterations = 100,
                MinimumExampleCountPerLeaf = 10,
                LearningRate = 0.05,
                Seed = Seed,
                NumberOfThreads = 1
            })));

        yield return new("Gam", Preprocessing(mlContext).Append(
            trainers.Gam(new GamBinaryTrainer.Options
            {
                NumberOfIterations = 500,
                LearningRate = 0.01,
                MinimumExampleCountPerLeaf = 10,
                NumberOfThreads = 1
            })));
    }

    private static EstimatorChain<NormalizingTransformer> Preprocessing(MLContext mlContext)
    {
        var oneHotColumns = FeatureBuilder.CategoricalColumns
            .Select(static column => new InputOutputColumnPair(column))
            .ToArray();
        var imputedColumns = FeatureBuilder.ImputedNumericColumns
            .Select(static column => new InputOutputColumnPair(column))
            .ToArray();

        return mlContext.Transforms.Categorical.OneHotEncoding(oneHotColumns)
            .Append(mlContext.Transforms.ReplaceMissingValues(
                imputedColumns,
                MissingValueReplacingEstimator.ReplacementMode.Mean))
            .Append(mlContext.Transforms.Concatenate("Features", FeatureBuilder.InputColumns))
            .Append(mlContext.Transforms.NormalizeMeanVariance("Features"))
            .AppendCacheCheckpoint(mlContext);
    }

    private static CandidateResult Evaluate(
        MLContext mlContext,
        IDataView trainingData,
        IDataView featureSchemaData,
        Candidate candidate)
    {
        var folds = candidate.Name == BaselineName
            ? mlContext.BinaryClassification.CrossValidateNonCalibrated(trainingData, candidate.Pipeline, FoldCount)
                .Select(static fold => (Metrics: (BinaryClassificationMetrics)fold.Metrics, LogLoss: double.NaN))
                .ToArray()
            : mlContext.BinaryClassification.CrossValidate(trainingData, candidate.Pipeline, FoldCount)
                .Select(static fold => (Metrics: (BinaryClassificationMetrics)fold.Metrics, fold.Metrics.LogLoss))
                .ToArray();

        return new CandidateResult(
            candidate.Name,
            MetricSummary.From(folds.Select(static fold => fold.Metrics.AreaUnderRocCurve)),
            MetricSummary.From(folds.Select(static fold => fold.Metrics.Accuracy)),
            MetricSummary.From(folds.Select(static fold => fold.Metrics.F1Score)),
            MetricSummary.From(folds.Select(static fold => fold.LogLoss)),
            candidate.Name == BaselineName
                ? null
                : IsOnnxExportable(mlContext, candidate.Pipeline.Fit(trainingData), featureSchemaData));
    }

    private static bool IsOnnxExportable(MLContext mlContext, ITransformer model, IDataView featureSchemaData)
    {
        try
        {
            using var stream = new MemoryStream();
            mlContext.Model.ConvertToOnnx(model, featureSchemaData, stream, OnnxLoanModel.ExportedColumns);
            return stream.Length > 0;
        }
        catch (InvalidOperationException)
        {
            return false;
        }
    }

    private static (List<LoanTrainingRow> Training, List<LoanTrainingRow> Test) StratifiedSplit(
        IReadOnlyList<LoanTrainingRow> samples,
        double testFraction,
        int seed)
    {
        var random = new Random(seed);
        var training = new List<LoanTrainingRow>();
        var test = new List<LoanTrainingRow>();

        foreach (var group in samples.GroupBy(static sample => sample.Label))
        {
            var shuffled = group.ToArray();
            random.Shuffle(shuffled);
            var testCount = (int)Math.Round(shuffled.Length * testFraction);
            test.AddRange(shuffled.Take(testCount));
            training.AddRange(shuffled.Skip(testCount));
        }

        var trainingArray = training.ToArray();
        random.Shuffle(trainingArray);
        return ([.. trainingArray], test);
    }

    private static void PrintTestMetrics(
        CalibratedBinaryClassificationMetrics metrics,
        BinaryClassificationMetrics baseline)
    {
        Console.WriteLine();
        Console.WriteLine("Hold-out test set metrics:");
        Console.WriteLine($"Accuracy:            {metrics.Accuracy:P2} (baseline \"always approve\": {baseline.Accuracy:P2})");
        Console.WriteLine($"AUC ROC:             {metrics.AreaUnderRocCurve:F4}");
        Console.WriteLine($"AUC PR:              {metrics.AreaUnderPrecisionRecallCurve:F4}");
        Console.WriteLine($"F1 (approved):       {metrics.F1Score:F4}");
        Console.WriteLine($"Precision approved:  {metrics.PositivePrecision:P2}, recall: {metrics.PositiveRecall:P2}");
        Console.WriteLine($"Precision rejected:  {metrics.NegativePrecision:P2}, recall: {metrics.NegativeRecall:P2}");
        Console.WriteLine($"LogLoss:             {metrics.LogLoss:F4}, reduction: {metrics.LogLossReduction:F4}");
        Console.WriteLine("Confusion matrix:");
        Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
    }

    private static string Format(MetricSummary summary) => $"{summary.Mean:F4}±{summary.StandardDeviation:F3}";

    private sealed record Candidate(string Name, IEstimator<ITransformer> Pipeline);
}

internal sealed record MetricSummary(double Mean, double StandardDeviation)
{
    public static MetricSummary From(IEnumerable<double> values)
    {
        var array = values.ToArray();
        var mean = array.Average();
        return new MetricSummary(mean, Math.Sqrt(array.Average(value => Math.Pow(value - mean, 2))));
    }
}

internal sealed record CandidateResult(
    string Name,
    MetricSummary AreaUnderRocCurve,
    MetricSummary Accuracy,
    MetricSummary F1Score,
    MetricSummary LogLoss,
    bool? OnnxExportable);

internal sealed record TestMetrics(
    double Accuracy,
    double AreaUnderRocCurve,
    double AreaUnderPrecisionRecallCurve,
    double F1Score,
    double PositivePrecision,
    double PositiveRecall,
    double NegativePrecision,
    double NegativeRecall,
    double LogLoss,
    double LogLossReduction,
    double[][] ConfusionMatrix)
{
    public static TestMetrics From(CalibratedBinaryClassificationMetrics metrics) => new(
        metrics.Accuracy,
        metrics.AreaUnderRocCurve,
        metrics.AreaUnderPrecisionRecallCurve,
        metrics.F1Score,
        metrics.PositivePrecision,
        metrics.PositiveRecall,
        metrics.NegativePrecision,
        metrics.NegativeRecall,
        metrics.LogLoss,
        metrics.LogLossReduction,
        [.. metrics.ConfusionMatrix.Counts.Select(static row => row.ToArray())]);
}

internal sealed record TrainingReport(
    DateTimeOffset CreatedAtUtc,
    string DataSha256,
    int Seed,
    int TrainingRows,
    int TestRows,
    int SerializedModelRows,
    string SelectedTrainer,
    string BestTrainerOverall,
    string SelectionMetric,
    IReadOnlyList<CandidateResult> CrossValidation,
    TestMetrics Test,
    double BaselineTestAccuracy,
    ParityResult OnnxParity);

[JsonSourceGenerationOptions(WriteIndented = true, NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals)]
[JsonSerializable(typeof(TrainingReport))]
internal sealed partial class TrainingReportJsonContext : JsonSerializerContext;
