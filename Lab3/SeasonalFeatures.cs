using Microsoft.ML.Data;

namespace Lab3;

internal sealed class SeasonalFeatures
{
    public const int SeasonalDays = 7;
    public const int ChannelCount = 3;

    private readonly float[] values;
    private readonly float[] dailyResiduals;
    private readonly float[] differences;

    private SeasonalFeatures(KpiSeries series)
    {
        Series = series;
        values = [.. series.Points.Select(static point => point.Value)];
        dailyResiduals = new float[values.Length];
        differences = new float[values.Length];
        HistoryLength = SeasonalDays * series.PointsPerDay;

        var period = series.PointsPerDay;
        Span<float> sameSlot = stackalloc float[SeasonalDays];
        for (var index = HistoryLength; index < values.Length; index++)
        {
            for (var day = 1; day <= SeasonalDays; day++)
            {
                sameSlot[day - 1] = values[index - day * period];
            }

            sameSlot.Sort();
            dailyResiduals[index] = values[index] - sameSlot[SeasonalDays / 2];
            differences[index] = values[index] - values[index - 1];
        }
    }

    public KpiSeries Series { get; }

    public int HistoryLength { get; }

    public static SeasonalFeatures Create(KpiSeries series) => new(series);

    public static int FeatureCount(int window) => ChannelCount * window;

    public static int WindowFromFeatureCount(int featureCount) =>
        featureCount > 0 && featureCount % ChannelCount == 0
            ? featureCount / ChannelCount
            : throw new InvalidDataException($"The model expects {featureCount} features, which is not a valid window.");

    public int FirstScorableIndex(int window) => HistoryLength + window - 1;

    public float DailyResidual(int index) => dailyResiduals[index];

    public float[] Vector(int index, int window)
    {
        if (index < FirstScorableIndex(window) || index >= values.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(index), "Not enough history to build the feature window.");
        }

        var start = index - window + 1;
        float[] vector = new float[FeatureCount(window)];
        values.AsSpan(start, window).CopyTo(vector);
        dailyResiduals.AsSpan(start, window).CopyTo(vector.AsSpan(window));
        differences.AsSpan(start, window).CopyTo(vector.AsSpan(2 * window));
        return vector;
    }

    public KpiModelInput Input(int index, int window) => new()
    {
        Features = Vector(index, window),
        Label = Series.Points[index].IsAnomaly == true
    };
}

public sealed class KpiModelInput
{
    public float[] Features { get; init; } = [];

    public bool Label { get; init; }

    public static SchemaDefinition Schema(int featureCount)
    {
        var schema = SchemaDefinition.Create(typeof(KpiModelInput));
        schema[nameof(Features)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, featureCount);
        return schema;
    }
}

public sealed class KpiPrediction
{
    public bool PredictedLabel { get; init; }

    public float Score { get; init; }
}
