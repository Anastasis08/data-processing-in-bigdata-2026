using Lab4.Cli;
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
        ModelTrainer.TrainAndExport(trainingContext, options.DataPath, options.ModelPath, options.OnnxPath);
    }

    if (options.Command is Command.Predict or Command.All)
    {
        var predictionContext = new MLContext(seed: ModelTrainer.Seed);
        PredictionService.PredictApplications(predictionContext, options.InputPath, options.OnnxPath, options.ModelPath);
    }

    return 0;
}
catch (Exception exception) when (exception is not OutOfMemoryException)
{
    Console.Error.WriteLine($"Error: {exception.Message}");
    return 1;
}