using Lab1;

try
{
    var options = CliOptions.Parse(args);

    if (options.ShowHelp)
    {
        CliOptions.PrintHelp();
        return 0;
    }

    var mlContext = new Microsoft.ML.MLContext(seed: 42);

    if (options.Command is Command.Train or Command.All)
    {
        ModelTrainer.TrainAndSave(mlContext, options.DataPath, options.ModelPath);
    }

    if (options.Command is Command.Predict or Command.All)
    {
        PredictionService.PredictNextTradingDay(mlContext, options.DataPath, options.ModelPath);
    }

    return 0;
}
catch (Exception exception) when (exception is not OutOfMemoryException)
{
    Console.Error.WriteLine($"Error: {exception.Message}");
    return 1;
}
