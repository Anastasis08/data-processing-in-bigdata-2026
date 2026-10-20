using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

namespace Lab2;

internal static class ModelTrainer
{
    public const int Seed = 42;
    private const double TestFraction = 0.2;
    private const int FoldCount = 5;
    private const string BaselineName = "Prior (baseline)";

    public static void TrainAndSave(MLContext mlContext, string dataPath, string modelPath)
    {
        ArgumentNullException.ThrowIfNull(mlContext);

        var applications = LoanCsvReader.Read(dataPath, requireLabel: true);
        var samples = applications.Select(FeatureBuilder.Build).ToArray();
        var (trainingSamples, testSamples) = StratifiedSplit(samples, TestFraction, Seed);
        var trainingData = mlContext.Data.LoadFromEnumerable(trainingSamples);
        var testData = mlContext.Data.LoadFromEnumerable(testSamples);

        Console.WriteLine("Part 1: training, evaluation, and serialization");
        Console.WriteLine(
            $"Data: {Path.GetFileName(dataPath)} ({samples.Length} applications, " +
            $"approved {samples.Count(static sample => sample.Label)}, " +
            $"rejected {samples.Count(static sample => !sample.Label)})");
        Console.WriteLine(
            $"Stratified split: {trainingSamples.Count} training / {testSamples.Count} test samples");
        Console.WriteLine();
        Console.WriteLine($"{FoldCount}-fold cross-validation on the training set (mean ± std):");
        Console.WriteLine(
            $"{"Trainer",-26} {"AUC",-15} {"Accuracy",-15} {"F1",-15} {"LogLoss",-8}");

        var candidates = CreateCandidates(mlContext)
            .Select(candidate => CrossValidate(mlContext, trainingData, candidate))
            .ToArray();

        foreach (var result in candidates)
        {
            Console.WriteLine(
                $"{result.Name,-26} {Format(result.AreaUnderRocCurve),-15} " +
                $"{Format(result.Accuracy),-15} {Format(result.F1Score),-15} {result.LogLoss.Mean,-8:F4}");
        }

        var best = candidates
            .Where(static result => result.Name != BaselineName)
            .MaxBy(static result => result.AreaUnderRocCurve.Mean)!;
        var bestCandidate = CreateCandidates(mlContext).Single(candidate => candidate.Name == best.Name);

        Console.WriteLine();
        Console.WriteLine($"Selected by mean cross-validated AUC: {best.Name}");

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
        mlContext.Model.Save(finalModel, fullData.Schema, modelPath);

        var report = new TrainingReport(
            CreatedAtUtc: DateTimeOffset.UtcNow,
            DataSha256: Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(dataPath))),
            Seed: Seed,
            TrainingRows: trainingSamples.Count,
            TestRows: testSamples.Count,
            SerializedModelRows: samples.Length,
            SelectedTrainer: best.Name,
            SelectionMetric: "Mean cross-validated AUC",
            CrossValidation: candidates,
            Test: TestMetrics.From(testMetrics),
            BaselineTestAccuracy: baselineMetrics.Accuracy);

        var reportPath = Path.ChangeExtension(modelPath, ".metrics.json");
        File.WriteAllText(
            reportPath,
            JsonSerializer.Serialize(report, TrainingReportJsonContext.Default.TrainingReport));

        Console.WriteLine($"Final model refit on all {samples.Length} rows and saved: {modelPath}");
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

    private static EstimatorChain<Microsoft.ML.Transforms.NormalizingTransformer> Preprocessing(MLContext mlContext)
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
                Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean))
            .Append(mlContext.Transforms.Concatenate(
                "Features",
                [.. FeatureBuilder.CategoricalColumns, .. FeatureBuilder.NumericColumns]))
            .Append(mlContext.Transforms.NormalizeMeanVariance("Features"))
            .AppendCacheCheckpoint(mlContext);
    }

    private static CrossValidationResult CrossValidate(MLContext mlContext, IDataView data, Candidate candidate)
    {
        var folds = candidate.Name == BaselineName
            ? mlContext.BinaryClassification.CrossValidateNonCalibrated(data, candidate.Pipeline, FoldCount)
                .Select(static fold => (Metrics: (BinaryClassificationMetrics)fold.Metrics, LogLoss: double.NaN))
                .ToArray()
            : mlContext.BinaryClassification.CrossValidate(data, candidate.Pipeline, FoldCount)
                .Select(static fold => (Metrics: (BinaryClassificationMetrics)fold.Metrics, fold.Metrics.LogLoss))
                .ToArray();

        return new CrossValidationResult(
            candidate.Name,
            MetricSummary.From(folds.Select(static fold => fold.Metrics.AreaUnderRocCurve)),
            MetricSummary.From(folds.Select(static fold => fold.Metrics.Accuracy)),
            MetricSummary.From(folds.Select(static fold => fold.Metrics.F1Score)),
            MetricSummary.From(folds.Select(static fold => fold.LogLoss)));
    }

    private static (List<LoanModelInput> Training, List<LoanModelInput> Test) StratifiedSplit(
        IReadOnlyList<LoanModelInput> samples,
        double testFraction,
        int seed)
    {
        var random = new Random(seed);
        var training = new List<LoanModelInput>();
        var test = new List<LoanModelInput>();

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

internal sealed record CrossValidationResult(
    string Name,
    MetricSummary AreaUnderRocCurve,
    MetricSummary Accuracy,
    MetricSummary F1Score,
    MetricSummary LogLoss);

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
    string SelectionMetric,
    IReadOnlyList<CrossValidationResult> CrossValidation,
    TestMetrics Test,
    double BaselineTestAccuracy);

[JsonSourceGenerationOptions(WriteIndented = true, NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals)]
[JsonSerializable(typeof(TrainingReport))]
internal sealed partial class TrainingReportJsonContext : JsonSerializerContext;
