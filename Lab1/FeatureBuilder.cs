namespace Lab1;

internal static class FeatureBuilder
{
    private const int Lookback = 5;
    public const int MinimumQuoteCount = Lookback + 2;

    public static IReadOnlyList<DatedSample> BuildTrainingSamples(IReadOnlyList<TeslaQuote> quotes)
    {
        ArgumentNullException.ThrowIfNull(quotes);

        var samples = new List<DatedSample>(quotes.Count - Lookback);
        for (var targetIndex = Lookback; targetIndex < quotes.Count; targetIndex++)
        {
            samples.Add(new DatedSample(
                quotes[targetIndex].Date,
                BuildInput(quotes, targetIndex, quotes[targetIndex].AdjustedClose)));
        }

        return samples;
    }

    public static TeslaModelInput BuildNextDayInput(IReadOnlyList<TeslaQuote> quotes)
    {
        ArgumentNullException.ThrowIfNull(quotes);
        return BuildInput(quotes, quotes.Count, label: 0f);
    }

    public static DateOnly GetNextTradingDay(DateOnly lastDate)
    {
        var candidate = lastDate.AddDays(1);
        while (candidate.DayOfWeek is DayOfWeek.Saturday or DayOfWeek.Sunday)
        {
            candidate = candidate.AddDays(1);
        }

        return candidate;
    }

    private static TeslaModelInput BuildInput(
        IReadOnlyList<TeslaQuote> quotes,
        int targetIndex,
        float label)
    {
        if (targetIndex < Lookback || targetIndex > quotes.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(targetIndex));
        }

        var previous = quotes[targetIndex - 1];
        var lag2 = quotes[targetIndex - 2].AdjustedClose;
        var rollingPrices = Enumerable.Range(targetIndex - Lookback, Lookback)
            .Select(index => quotes[index].AdjustedClose)
            .ToArray();
        var mean = rollingPrices.Average();
        var variance = rollingPrices.Average(price => MathF.Pow(price - mean, 2));
        var targetDate = targetIndex < quotes.Count
            ? quotes[targetIndex].Date
            : GetNextTradingDay(previous.Date);
        var monthAngle = 2f * MathF.PI * targetDate.Month / 12f;

        return new TeslaModelInput
        {
            Lag1 = previous.AdjustedClose,
            Lag2 = lag2,
            Lag3 = quotes[targetIndex - 3].AdjustedClose,
            Lag5 = quotes[targetIndex - 5].AdjustedClose,
            RollingMean5 = mean,
            RollingStdDev5 = MathF.Sqrt(variance),
            PreviousDailyReturn = SafeRatio(previous.AdjustedClose, lag2) - 1f,
            PreviousRangeRatio = SafeRatio(previous.High - previous.Low, previous.Close),
            PreviousVolumeLog10 = MathF.Log10(previous.Volume + 1f),
            DayIndex = targetDate.DayNumber - quotes[0].Date.DayNumber,
            MonthSin = MathF.Sin(monthAngle),
            MonthCos = MathF.Cos(monthAngle),
            Label = label
        };
    }

    private static float SafeRatio(float numerator, float denominator) =>
        MathF.Abs(denominator) < float.Epsilon ? 0f : numerator / denominator;
}
