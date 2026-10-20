using Microsoft.ML.Data;

namespace Lab1;

public sealed class TeslaModelInput
{
    public float Lag1 { get; init; }
    public float Lag2 { get; init; }
    public float Lag3 { get; init; }
    public float Lag5 { get; init; }
    public float RollingMean5 { get; init; }
    public float RollingStdDev5 { get; init; }
    public float PreviousDailyReturn { get; init; }
    public float PreviousRangeRatio { get; init; }
    public float PreviousVolumeLog10 { get; init; }
    public float DayIndex { get; init; }
    public float MonthSin { get; init; }
    public float MonthCos { get; init; }

    [ColumnName("Label")]
    public float Label { get; init; }
}

public sealed class TeslaPricePrediction
{
    [ColumnName("Score")]
    public float Price { get; init; }
}

internal sealed record DatedSample(DateOnly Date, TeslaModelInput Input);
