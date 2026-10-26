using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Lab3;

internal static class ModelTrainer
{
    public const int Seed = 42;
    private const double TrainingFraction = 0.6;
    private const double ValidationFraction = 0.2;
    private const string BaselineName = "Seasonal residual (baseline)";

    private static readonly (int Window, int Rank)[] SearchSpace =
    [
        (4, 2), (4, 4), (8, 4), (8, 8), (16, 8), (16, 16), (32, 16)
    ];

    public static void TrainAll(MLContext mlContext, IReadOnlyList<string> dataPaths, string modelDirectory)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(dataPaths);

        Console.WriteLine("Part 1: training, evaluation, and serialization");
        Console.WriteLine(
            $"Chronological split {TrainingFraction:P0} / {ValidationFraction:P0} / " +
            $"{1 - TrainingFraction - ValidationFraction:P0}; RandomizedPcaTrainer is fitted on normal training points only.");
        Console.WriteLine(
            $"Metrics are delay-adjusted: an anomaly segment counts as detected if it is flagged within " +
            $"{AnomalyMetrics.DetectionDelay} points of its start.");

        Directory.CreateDirectory(modelDirectory);
        var reports = dataPaths
            .Select(path => TrainSeries(mlContext, path, modelDirectory))
            .ToArray();

        Console.WriteLine();
        Console.WriteLine("Summary on the test parts:");
        Console.WriteLine($"{"KPI",-32} {"Model",-18} {"F1",6} {"Precision",10} {"Recall",7} {"AUC",6} {"Baseline F1",12}");
        foreach (var report in reports)
        {
            Console.WriteLine(
                $"{report.Kpi,-32} {$"w={report.Window}, k={report.Rank}",-18} {report.Test.DelayAdjusted.F1Score,6:F3} " +
                $"{report.Test.DelayAdjusted.Precision,10:F3} {report.Test.DelayAdjusted.Recall,7:F3} " +
                $"{report.Test.AreaUnderRocCurve,6:F3} {report.BaselineTest.F1Score,12:F3}");
        }

        var summary = new TrainingSummary(
            CreatedAtUtc: DateTimeOffset.UtcNow,
            Seed: Seed,
            TrainingFraction: TrainingFraction,
            ValidationFraction: ValidationFraction,
            DetectionDelay: AnomalyMetrics.DetectionDelay,
            Kpis: reports);
        var summaryPath = Path.Combine(modelDirectory, "training-report.json");
        File.WriteAllText(summaryPath, JsonSerializer.Serialize(summary, TrainingReportJsonContext.Default.TrainingSummary));

        Console.WriteLine();
        Console.WriteLine($"Metrics saved: {summaryPath}");
        Console.WriteLine();
    }

    private static KpiReport TrainSeries(MLContext mlContext, string dataPath, string modelDirectory)
    {
        var series = KpiCsvReader.Read(dataPath);
        var features = SeasonalFeatures.Create(series);
        var labels = series.Points.Select(static point => point.IsAnomaly).ToArray();
        var count = series.Points.Count;
        var validationStart = (int)(count * TrainingFraction);
        var testStart = (int)(count * (TrainingFraction + ValidationFraction));

        if (!series.HasLabels)
        {
            throw new InvalidDataException($"Training requires a labelled KPI file: {dataPath}");
        }

        Console.WriteLine();
        Console.WriteLine(
            $"KPI {series.Name}: {count} points every {series.StepSeconds} s " +
            $"({series.Points[0].Time:yyyy-MM-dd} .. {series.Points[^1].Time:yyyy-MM-dd}), " +
            $"anomalies {series.AnomalyCount} ({(double)series.AnomalyCount / count:P2}) in " +
            $"{AnomalyMetrics.Segments([.. labels.Select(static label => label == true)]).Count} segments, " +
            $"gaps filled {series.ImputedCount}");
        Console.WriteLine($"{"Candidate",-18} {"Val F1",7} {"Precision",10} {"Recall",7} {"Threshold",10}");

        var candidates = SearchSpace
            .Select(space => FitCandidate(mlContext, features, labels, space.Window, space.Rank, validationStart, testStart))
            .ToArray();

        foreach (var candidate in candidates)
        {
            Console.WriteLine(
                $"{$"w={candidate.Window}, k={candidate.Rank}",-18} {candidate.Validation.F1Score,7:F3} " +
                $"{candidate.Validation.Precision,10:F3} {candidate.Validation.Recall,7:F3} {candidate.Threshold,10:F4}");
        }

        var best = candidates.MaxBy(static candidate => candidate.Validation.F1Score)!;
        var model = WithThreshold(mlContext, best.Model, best.Threshold);
        var testPredictions = Score(mlContext, model, features, best.Window, testStart, count);
        var testLabels = labels[testStart..count];
        var predicted = testPredictions.Select(static prediction => prediction.PredictedLabel).ToArray();
        var testMetrics = mlContext.AnomalyDetection.Evaluate(
            mlContext.Data.LoadFromEnumerable(
                testPredictions.Select((prediction, index) => new EvaluationRow
                {
                    Label = testLabels[index] == true ? 1f : 0f,
                    Score = prediction.Score,
                    PredictedLabel = prediction.PredictedLabel
                })));

        var baseline = EvaluateBaseline(features, labels, validationStart, testStart);
        var test = new TestMetrics(
            PointWise: AnomalyMetrics.PointWise(testLabels, predicted),
            DelayAdjusted: AnomalyMetrics.DelayAdjusted(testLabels, predicted),
            AreaUnderRocCurve: testMetrics.AreaUnderRocCurve,
            DetectionRateAtFalsePositiveCount: testMetrics.DetectionRateAtFalsePositiveCount);

        Console.WriteLine($"Selected by validation F1: w={best.Window}, k={best.Rank}, threshold {best.Threshold:F4}");
        Console.WriteLine(
            $"Test: F1 {test.DelayAdjusted.F1Score:F3} (precision {test.DelayAdjusted.Precision:F3}, " +
            $"recall {test.DelayAdjusted.Recall:F3}), point-wise F1 {test.PointWise.F1Score:F3}, " +
            $"AUC {test.AreaUnderRocCurve:F3}; {BaselineName} F1 {baseline.Test.F1Score:F3}");
        Console.WriteLine(
            $"Test confusion (delay-adjusted): TP {test.DelayAdjusted.TruePositives}, " +
            $"FP {test.DelayAdjusted.FalsePositives}, FN {test.DelayAdjusted.FalseNegatives}");

        var modelPath = Path.Combine(modelDirectory, $"{series.Name}.zip");
        var schema = mlContext.Data.LoadFromEnumerable(
            Array.Empty<KpiModelInput>(),
            KpiModelInput.Schema(SeasonalFeatures.FeatureCount(best.Window))).Schema;
        mlContext.Model.Save(model, schema, modelPath);
        Console.WriteLine($"Model saved: {modelPath}");

        return new KpiReport(
            Kpi: series.Name,
            DataSha256: Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(dataPath))),
            Points: count,
            StepSeconds: series.StepSeconds,
            ImputedPoints: series.ImputedCount,
            Anomalies: series.AnomalyCount,
            TrainingNormalPoints: best.TrainingRows,
            ValidationRange: [validationStart, testStart - 1],
            TestRange: [testStart, count - 1],
            Window: best.Window,
            Rank: best.Rank,
            Threshold: best.Threshold,
            SelectionMetric: "Delay-adjusted F1 on the validation part",
            Candidates: [.. candidates.Select(static candidate => new CandidateReport(
                candidate.Window, candidate.Rank, candidate.Threshold, candidate.Validation))],
            Test: test,
            BaselineThreshold: baseline.Threshold,
            BaselineValidation: baseline.Validation,
            BaselineTest: baseline.Test,
            ModelFile: Path.GetFileName(modelPath));
    }

    private static Candidate FitCandidate(
        MLContext mlContext,
        SeasonalFeatures features,
        bool?[] labels,
        int window,
        int rank,
        int validationStart,
        int testStart)
    {
        var featureCount = SeasonalFeatures.FeatureCount(window);
        var trainingRows = Enumerable.Range(features.FirstScorableIndex(window), validationStart - features.FirstScorableIndex(window))
            .Where(index => labels[index] == false && !features.Series.Points[index].IsImputed)
            .Select(index => features.Input(index, window))
            .ToArray();

        var trainer = mlContext.AnomalyDetection.Trainers.RandomizedPca(new RandomizedPcaTrainer.Options
        {
            Rank = rank,
            Oversampling = 20,
            EnsureZeroMean = true,
            Seed = Seed
        });
        var pipeline = mlContext.Transforms.NormalizeMeanVariance(nameof(KpiModelInput.Features)).Append(trainer);
        var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(trainingRows, KpiModelInput.Schema(featureCount)));

        var validation = Score(mlContext, model, features, window, validationStart, testStart);
        var (threshold, metrics) = AnomalyMetrics.BestThreshold(
            labels[validationStart..testStart],
            [.. validation.Select(static prediction => prediction.Score)]);

        return new Candidate(window, rank, trainingRows.Length, model, threshold, metrics);
    }

    private static TransformerChain<ITransformer> WithThreshold(
        MLContext mlContext,
        TransformerChain<AnomalyPredictionTransformer<PcaModelParameters>> model,
        float threshold) =>
        new([.. model.Take(model.Count() - 1), mlContext.AnomalyDetection.ChangeModelThreshold(model.LastTransformer, threshold)]);

    private static KpiPrediction[] Score(
        MLContext mlContext,
        ITransformer model,
        SeasonalFeatures features,
        int window,
        int start,
        int end)
    {
        var rows = Enumerable.Range(start, end - start).Select(index => features.Input(index, window));
        var data = mlContext.Data.LoadFromEnumerable(rows, KpiModelInput.Schema(SeasonalFeatures.FeatureCount(window)));
        return [.. mlContext.Data.CreateEnumerable<KpiPrediction>(model.Transform(data), reuseRowObject: false)];
    }

    private static (float Threshold, DetectionMetrics Validation, DetectionMetrics Test) EvaluateBaseline(
        SeasonalFeatures features,
        bool?[] labels,
        int validationStart,
        int testStart)
    {
        var trainingResiduals = Enumerable.Range(features.HistoryLength, validationStart - features.HistoryLength)
            .Where(index => labels[index] == false)
            .Select(features.DailyResidual)
            .ToArray();
        var median = Median(trainingResiduals);
        var scale = Median([.. trainingResiduals.Select(residual => MathF.Abs(residual - median))]) + 1e-6f;

        float[] ScoreRange(int start, int end) =>
            [.. Enumerable.Range(start, end - start).Select(index => MathF.Abs(features.DailyResidual(index) - median) / scale)];

        var (threshold, validation) = AnomalyMetrics.BestThreshold(
            labels[validationStart..testStart],
            ScoreRange(validationStart, testStart));
        var test = AnomalyMetrics.DelayAdjusted(
            labels[testStart..],
            [.. ScoreRange(testStart, labels.Length).Select(score => score >= threshold)]);

        return (threshold, validation, test);
    }

    private static float Median(float[] values)
    {
        var sorted = values.Order().ToArray();
        return sorted.Length == 0 ? 0f : sorted[sorted.Length / 2];
    }

    private sealed record Candidate(
        int Window,
        int Rank,
        int TrainingRows,
        TransformerChain<AnomalyPredictionTransformer<PcaModelParameters>> Model,
        float Threshold,
        DetectionMetrics Validation);

    private sealed class EvaluationRow
    {
        public float Label { get; init; }

        public float Score { get; init; }

        public bool PredictedLabel { get; init; }
    }
}

internal sealed record CandidateReport(int Window, int Rank, float Threshold, DetectionMetrics Validation);

internal sealed record TestMetrics(
    DetectionMetrics PointWise,
    DetectionMetrics DelayAdjusted,
    double AreaUnderRocCurve,
    double DetectionRateAtFalsePositiveCount);

internal sealed record KpiReport(
    string Kpi,
    string DataSha256,
    int Points,
    int StepSeconds,
    int ImputedPoints,
    int Anomalies,
    int TrainingNormalPoints,
    int[] ValidationRange,
    int[] TestRange,
    int Window,
    int Rank,
    float Threshold,
    string SelectionMetric,
    IReadOnlyList<CandidateReport> Candidates,
    TestMetrics Test,
    float BaselineThreshold,
    DetectionMetrics BaselineValidation,
    DetectionMetrics BaselineTest,
    string ModelFile);

internal sealed record TrainingSummary(
    DateTimeOffset CreatedAtUtc,
    int Seed,
    double TrainingFraction,
    double ValidationFraction,
    int DetectionDelay,
    IReadOnlyList<KpiReport> Kpis);

[JsonSourceGenerationOptions(WriteIndented = true, NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals)]
[JsonSerializable(typeof(TrainingSummary))]
internal sealed partial class TrainingReportJsonContext : JsonSerializerContext;
