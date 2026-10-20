using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML;

namespace Lab1;

internal static class ModelTrainer
{
    private const double TrainingFraction = 0.8;

    private static readonly string[] FeatureColumns =
    [
        nameof(TeslaModelInput.Lag1),
        nameof(TeslaModelInput.Lag2),
        nameof(TeslaModelInput.Lag3),
        nameof(TeslaModelInput.Lag5),
        nameof(TeslaModelInput.RollingMean5),
        nameof(TeslaModelInput.RollingStdDev5),
        nameof(TeslaModelInput.PreviousDailyReturn),
        nameof(TeslaModelInput.PreviousRangeRatio),
        nameof(TeslaModelInput.PreviousVolumeLog10),
        nameof(TeslaModelInput.DayIndex),
        nameof(TeslaModelInput.MonthSin),
        nameof(TeslaModelInput.MonthCos)
    ];

    public static void TrainAndSave(MLContext mlContext, string dataPath, string modelPath)
    {
        ArgumentNullException.ThrowIfNull(mlContext);

        var quotes = TeslaCsvReader.Read(dataPath);
        var samples = FeatureBuilder.BuildTrainingSamples(quotes);
        var trainingCount = (int)Math.Floor(samples.Count * TrainingFraction);

        if (trainingCount == 0 || trainingCount == samples.Count)
        {
            throw new InvalidOperationException("The dataset is too small for a chronological split.");
        }

        var trainingSamples = samples.Take(trainingCount).Select(static sample => sample.Input);
        var testSamples = samples.Skip(trainingCount).Select(static sample => sample.Input);
        var trainingData = mlContext.Data.LoadFromEnumerable(trainingSamples);
        var testData = mlContext.Data.LoadFromEnumerable(testSamples);

        var pipeline = mlContext.Transforms
            .Concatenate("Features", FeatureColumns)
            .Append(mlContext.Transforms.NormalizeMeanVariance("Features"))
            .AppendCacheCheckpoint(mlContext)
            .Append(mlContext.Regression.Trainers.Ols(
                labelColumnName: "Label",
                featureColumnName: "Features"));

        Console.WriteLine("Part 1: training, evaluation, and serialization");
        Console.WriteLine($"Data: {Path.GetFileName(dataPath)} ({quotes.Count} trading days)");
        Console.WriteLine(
            $"Chronological split: {trainingCount} training / {samples.Count - trainingCount} test samples");

        var model = pipeline.Fit(trainingData);
        var predictions = model.Transform(testData);
        var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label");

        var testInputs = samples.Skip(trainingCount).Select(static sample => sample.Input).ToArray();
        var baselineMae = testInputs.Average(input => Math.Abs(input.Label - input.Lag1));
        var baselineRmse = Math.Sqrt(testInputs.Average(input => Math.Pow(input.Label - input.Lag1, 2)));

        Console.WriteLine($"MAE:  {metrics.MeanAbsoluteError:F4}");
        Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:F4}");
        Console.WriteLine($"R²:   {metrics.RSquared:F4}");
        Console.WriteLine($"Baseline model (previous day's price) - MAE: {baselineMae:F4}, RMSE: {baselineRmse:F4}");

      
        var fullData = mlContext.Data.LoadFromEnumerable(samples.Select(static sample => sample.Input));
        var finalModel = pipeline.Fit(fullData);
        var modelDirectory = Path.GetDirectoryName(modelPath)
            ?? throw new InvalidOperationException("Unable to determine the model directory.");
        Directory.CreateDirectory(modelDirectory);
        mlContext.Model.Save(finalModel, fullData.Schema, modelPath);

        var report = new TrainingReport(
            CreatedAtUtc: DateTimeOffset.UtcNow,
            DataSha256: Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(dataPath))),
            FirstDate: samples[0].Date,
            TrainingEndDate: samples[trainingCount - 1].Date,
            TestStartDate: samples[trainingCount].Date,
            LastDate: samples[^1].Date,
            TrainingRows: trainingCount,
            TestRows: samples.Count - trainingCount,
            SerializedModelRows: samples.Count,
            MeanAbsoluteError: metrics.MeanAbsoluteError,
            RootMeanSquaredError: metrics.RootMeanSquaredError,
            RSquared: metrics.RSquared,
            BaselineMeanAbsoluteError: baselineMae,
            BaselineRootMeanSquaredError: baselineRmse);

        var reportPath = Path.ChangeExtension(modelPath, ".metrics.json");
        File.WriteAllText(
            reportPath,
            JsonSerializer.Serialize(report, TrainingReportJsonContext.Default.TrainingReport));

        Console.WriteLine($"Model saved: {modelPath}");
        Console.WriteLine($"Metrics saved: {reportPath}");
        Console.WriteLine();
    }
}

internal sealed record TrainingReport(
    DateTimeOffset CreatedAtUtc,
    string DataSha256,
    DateOnly FirstDate,
    DateOnly TrainingEndDate,
    DateOnly TestStartDate,
    DateOnly LastDate,
    int TrainingRows,
    int TestRows,
    int SerializedModelRows,
    double MeanAbsoluteError,
    double RootMeanSquaredError,
    double RSquared,
    double BaselineMeanAbsoluteError,
    double BaselineRootMeanSquaredError);

[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(TrainingReport))]
internal sealed partial class TrainingReportJsonContext : JsonSerializerContext;
