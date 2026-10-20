using System.Globalization;

namespace Lab1;

internal sealed record TeslaQuote(
    DateOnly Date,
    float Open,
    float High,
    float Low,
    float Close,
    float AdjustedClose,
    long Volume);

internal static class TeslaCsvReader
{
    private const int ExpectedColumnCount = 7;

    public static IReadOnlyList<TeslaQuote> Read(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("The CSV file containing Tesla data was not found.", path);
        }

        var quotes = new List<TeslaQuote>();
        var lineNumber = 0;

        foreach (var line in File.ReadLines(path))
        {
            lineNumber++;
            if (lineNumber == 1)
            {
                ValidateHeader(line);
                continue;
            }

            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var columns = line.Split(',');
            if (columns.Length != ExpectedColumnCount)
            {
                throw new InvalidDataException($"Line {lineNumber}: expected {ExpectedColumnCount} columns.");
            }

            try
            {
                quotes.Add(new TeslaQuote(
                    DateOnly.ParseExact(columns[0], "yyyy-MM-dd", CultureInfo.InvariantCulture),
                    ParseSingle(columns[1]),
                    ParseSingle(columns[2]),
                    ParseSingle(columns[3]),
                    ParseSingle(columns[4]),
                    ParseSingle(columns[5]),
                    long.Parse(columns[6], NumberStyles.Integer, CultureInfo.InvariantCulture)));
            }
            catch (FormatException exception)
            {
                throw new InvalidDataException($"Line {lineNumber} contains an invalid value.", exception);
            }
        }

        if (quotes.Count < FeatureBuilder.MinimumQuoteCount)
        {
            throw new InvalidDataException(
                $"At least {FeatureBuilder.MinimumQuoteCount} records are required to build the features.");
        }

        return quotes.OrderBy(static quote => quote.Date).ToArray();
    }

    private static float ParseSingle(string value) =>
        float.Parse(value, NumberStyles.Float, CultureInfo.InvariantCulture);

    private static void ValidateHeader(string header)
    {
        const string expected = "Date,Open,High,Low,Close,Adj Close,Volume";
        if (!string.Equals(header.TrimStart('\uFEFF'), expected, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"Invalid CSV header. Expected: {expected}");
        }
    }
}
