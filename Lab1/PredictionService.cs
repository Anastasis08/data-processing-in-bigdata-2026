using Microsoft.ML;

namespace Lab1;

internal static class PredictionService
{
    public static void PredictNextTradingDay(MLContext mlContext, string dataPath, string modelPath)
    {
        ArgumentNullException.ThrowIfNull(mlContext);

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException(
                "The serialized model was not found. Run the train command first.",
                modelPath);
        }

        var quotes = TeslaCsvReader.Read(dataPath);
        var input = FeatureBuilder.BuildNextDayInput(quotes);
        var model = mlContext.Model.Load(modelPath, out _);
        var inputView = mlContext.Data.LoadFromEnumerable([input]);
        var transformed = model.Transform(inputView);
        var prediction = mlContext.Data
            .CreateEnumerable<TeslaPricePrediction>(transformed, reuseRowObject: false)
            .Single();

        var lastQuote = quotes[^1];
        var predictionDate = FeatureBuilder.GetNextTradingDay(lastQuote.Date);
        var change = prediction.Price - lastQuote.AdjustedClose;
        var changePercent = 100f * change / lastQuote.AdjustedClose;

        Console.WriteLine("Part 2: loading the model and prediction");
        Console.WriteLine($"Last known date: {lastQuote.Date:yyyy-MM-dd}");
        Console.WriteLine($"Last adjusted closing price: ${lastQuote.AdjustedClose:F2}");
        Console.WriteLine($"Prediction for {predictionDate:yyyy-MM-dd}: ${prediction.Price:F2}");
        Console.WriteLine($"Expected change: {change:+0.00;-0.00;0.00} ({changePercent:+0.00;-0.00;0.00}%)");
    }
}
