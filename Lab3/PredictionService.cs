using System.Globalization;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Lab3;

internal static class PredictionService
{
    private const int MaxListedSegments = 10;

    public static void DetectAll(MLContext mlContext, IReadOnlyList<string> inputPaths, string modelDirectory, int days)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(inputPaths);

        Console.WriteLine("Part 2: loading the models and detecting anomalies");
        Console.WriteLine($"Scoring the last {days} days of every KPI point by point, as a monitoring service would.");

        foreach (var inputPath in inputPaths)
        {
            var modelPath = Path.Combine(modelDirectory, $"{Path.GetFileNameWithoutExtension(inputPath)}.zip");
            Detect(mlContext, inputPath, modelPath, days);
        }
    }

    public static void Detect(MLContext mlContext, string inputPath, string modelPath, int days)
    {
        ArgumentNullException.ThrowIfNull(mlContext);

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException(
                "The serialized model was not found. Run the train command first.",
                modelPath);
        }

        var model = mlContext.Model.Load(modelPath, out var inputSchema);
        var featureColumn = inputSchema[nameof(KpiModelInput.Features)].Type as VectorDataViewType
            ?? throw new InvalidDataException("The model input column 'Features' is not a vector.");
        var window = SeasonalFeatures.WindowFromFeatureCount(featureColumn.Size);
        var schema = KpiModelInput.Schema(featureColumn.Size);
        using var engine = mlContext.Model.CreatePredictionEngine<KpiModelInput, KpiPrediction>(
            model,
            inputSchemaDefinition: schema);

        var series = KpiCsvReader.Read(inputPath);
        var features = SeasonalFeatures.Create(series);
        var start = Math.Max(features.FirstScorableIndex(window), series.Points.Count - days * series.PointsPerDay);
        var points = series.Points.Skip(start).ToArray();
        var predictions = points
            .Select((_, offset) => engine.Predict(new KpiModelInput { Features = features.Vector(start + offset, window) }))
            .ToArray();
        var flags = predictions.Select(static prediction => prediction.PredictedLabel).ToArray();

        Console.WriteLine();
        Console.WriteLine($"KPI {series.Name}: model {Path.GetFileName(modelPath)} (window {window} points)");
        Console.WriteLine(
            $"Scored {points.Length} points {points[0].Time:yyyy-MM-dd HH:mm} .. {points[^1].Time:yyyy-MM-dd HH:mm} UTC, " +
            $"flagged {flags.Count(static flag => flag)}");

        var detected = AnomalyMetrics.Segments(flags);
        Console.WriteLine($"Detected anomaly segments: {detected.Count}");
        foreach (var segment in detected.Take(MaxListedSegments))
        {
            var peak = Enumerable.Range(segment.Start, segment.Length).MaxBy(index => predictions[index].Score);
            var actual = points[segment.Start..(segment.End + 1)].Any(static point => point.IsAnomaly == true)
                ? "labelled"
                : points[segment.Start].IsAnomaly is null ? "-" : "false alarm";
            Console.WriteLine(
                $"  {points[segment.Start].Time:MM-dd HH:mm} .. {points[segment.End].Time:MM-dd HH:mm} " +
                $"{segment.Length,4} pts  max score {predictions[peak].Score,8:F4}  value {points[peak].Value,9:F4}  {actual}");
        }

        if (detected.Count > MaxListedSegments)
        {
            Console.WriteLine($"  ... and {detected.Count - MaxListedSegments} more (see the CSV output)");
        }

        if (series.HasLabels)
        {
            var labels = points.Select(static point => point.IsAnomaly).ToArray();
            var metrics = AnomalyMetrics.DelayAdjusted(labels, flags);
            Console.WriteLine(
                $"Against labels: {AnomalyMetrics.Segments([.. labels.Select(static label => label == true)]).Count} " +
                $"labelled segments, delay-adjusted F1 {metrics.F1Score:F3} " +
                $"(precision {metrics.Precision:F3}, recall {metrics.Recall:F3})");
        }

        var outputPath = WriteCsv(modelPath, series.Name, points, predictions);
        Console.WriteLine($"Scores saved: {outputPath}");
    }

    private static string WriteCsv(string modelPath, string name, KpiPoint[] points, KpiPrediction[] predictions)
    {
        var directory = Path.Combine(Path.GetDirectoryName(modelPath)!, "predictions");
        Directory.CreateDirectory(directory);
        var outputPath = Path.Combine(directory, $"{name}.csv");

        var builder = new StringBuilder("timestamp,time_utc,value,label,imputed,score,is_anomaly\n");
        for (var index = 0; index < points.Length; index++)
        {
            var point = points[index];
            builder.Append(CultureInfo.InvariantCulture,
                $"{point.Timestamp},{point.Time:yyyy-MM-ddTHH:mm:ssZ},{point.Value},{Label(point.IsAnomaly)}," +
                $"{(point.IsImputed ? 1 : 0)},{predictions[index].Score},{(predictions[index].PredictedLabel ? 1 : 0)}\n");
        }

        File.WriteAllText(outputPath, builder.ToString());
        return outputPath;
    }

    private static string Label(bool? label) => label switch
    {
        true => "1",
        false => "0",
        null => string.Empty
    };
}
