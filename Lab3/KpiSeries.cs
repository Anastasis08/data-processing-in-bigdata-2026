using System.Globalization;

namespace Lab3;

internal sealed record KpiPoint(long Timestamp, float Value, bool? IsAnomaly, bool IsImputed)
{
    public DateTimeOffset Time => DateTimeOffset.FromUnixTimeSeconds(Timestamp);
}

internal sealed record KpiSeries(string Name, int StepSeconds, IReadOnlyList<KpiPoint> Points)
{
    private const int SecondsPerDay = 86_400;

    public int PointsPerDay => SecondsPerDay / StepSeconds;

    public int ImputedCount => Points.Count(static point => point.IsImputed);

    public int AnomalyCount => Points.Count(static point => point.IsAnomaly == true);

    public bool HasLabels => Points.Any(static point => point.IsAnomaly is not null);
}

internal static class KpiCsvReader
{
    private const string TimestampColumn = "timestamp";
    private const string ValueColumn = "value";
    private const string LabelColumn = "label";

    public static KpiSeries Read(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("The KPI file was not found.", path);
        }

        using var reader = new StreamReader(path);
        var header = reader.ReadLine() ?? throw new InvalidDataException($"The file is empty: {path}");
        var columns = header.Split(',')
            .Select(static (name, index) => (Name: name.Trim().ToLowerInvariant(), Index: index))
            .ToDictionary(static column => column.Name, static column => column.Index);

        if (!columns.TryGetValue(TimestampColumn, out var timestampIndex) || !columns.TryGetValue(ValueColumn, out var valueIndex))
        {
            throw new InvalidDataException($"Columns '{TimestampColumn}' and '{ValueColumn}' are required: {path}");
        }

        var hasLabel = columns.TryGetValue(LabelColumn, out var labelIndex);
        var observations = new SortedDictionary<long, KpiPoint>();
        var lineNumber = 1;

        while (reader.ReadLine() is { } line)
        {
            lineNumber++;
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var fields = line.Split(',');
            var labelMissing = hasLabel && fields.Length == columns.Count - 1 && labelIndex == columns.Count - 1;
            if (fields.Length != columns.Count && !labelMissing)
            {
                throw new InvalidDataException(
                    $"Line {lineNumber}: expected {columns.Count} values, found {fields.Length}.");
            }

            try
            {
                var timestamp = long.Parse(fields[timestampIndex], NumberStyles.Integer, CultureInfo.InvariantCulture);
                var value = float.Parse(fields[valueIndex], NumberStyles.Float, CultureInfo.InvariantCulture);
                var label = hasLabel && !labelMissing ? ParseLabel(fields[labelIndex].Trim()) : null;

                if (!float.IsFinite(value))
                {
                    throw new FormatException($"Value '{fields[valueIndex]}' is not a finite number.");
                }

                observations[timestamp] = new KpiPoint(timestamp, value, label, IsImputed: false);
            }
            catch (FormatException exception)
            {
                throw new InvalidDataException($"Line {lineNumber}: {exception.Message}", exception);
            }
        }

        if (observations.Count < 2)
        {
            throw new InvalidDataException($"At least two observations are required: {path}");
        }

        var ordered = observations.Values.ToArray();
        var step = InferStep(ordered);
        return new KpiSeries(Path.GetFileNameWithoutExtension(path), step, Regularize(ordered, step));
    }

    private static int InferStep(KpiPoint[] points)
    {
        var step = points
            .Zip(points.Skip(1), static (previous, next) => next.Timestamp - previous.Timestamp)
            .GroupBy(static difference => difference)
            .MaxBy(static group => group.Count())!
            .Key;

        if (step <= 0 || 86_400 % step != 0)
        {
            throw new InvalidDataException($"The sampling interval of {step} s does not divide a day evenly.");
        }

        return (int)step;
    }

    private static List<KpiPoint> Regularize(KpiPoint[] points, int step)
    {
        var regular = new List<KpiPoint>(points.Length) { points[0] };

        foreach (var point in points.Skip(1))
        {
            var previous = regular[^1];
            var gap = point.Timestamp - previous.Timestamp;

            if (gap % step != 0)
            {
                throw new InvalidDataException(
                    $"Timestamp {point.Timestamp} is not aligned to the {step} s sampling grid.");
            }

            var missing = gap / step - 1;
            for (var offset = 1; offset <= missing; offset++)
            {
                var fraction = (float)offset / (missing + 1);
                regular.Add(new KpiPoint(
                    previous.Timestamp + offset * step,
                    previous.Value + (point.Value - previous.Value) * fraction,
                    IsAnomaly: null,
                    IsImputed: true));
            }

            regular.Add(point);
        }

        return regular;
    }

    private static bool? ParseLabel(string value) => value switch
    {
        "" => null,
        "0" or "0.0" => false,
        "1" or "1.0" => true,
        _ => throw new FormatException($"Invalid label '{value}', expected 0 or 1.")
    };
}
