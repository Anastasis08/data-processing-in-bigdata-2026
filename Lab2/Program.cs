using Lab2;
using Microsoft.ML;

try
{
    var options = CliOptions.Parse(args);

    if (options.ShowHelp)
    {
        CliOptions.PrintHelp();
        return 0;
    }

    if (options.Command is Command.Train or Command.All)
    {
        var trainingContext = new MLContext(seed: ModelTrainer.Seed);
        ModelTrainer.TrainAndSave(trainingContext, options.DataPath, options.ModelPath);
    }

    if (options.Command is Command.Predict or Command.All)
    {
        var predictionContext = new MLContext(seed: ModelTrainer.Seed);
        PredictionService.PredictApplications(predictionContext, options.InputPath, options.ModelPath);
    }

    return 0;
}
catch (Exception exception) when (exception is not OutOfMemoryException)
{
    Console.Error.WriteLine($"Error: {exception.Message}");
    return 1;
}
