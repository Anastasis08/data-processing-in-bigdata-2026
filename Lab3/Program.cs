using Lab3;
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
        ModelTrainer.TrainAll(trainingContext, options.DataPaths, options.ModelDirectory);
    }

    if (options.Command is Command.Predict or Command.All)
    {
        var predictionContext = new MLContext(seed: ModelTrainer.Seed);
        if (options.InputPath is { } inputPath)
        {
            var modelPath = options.ModelPath
                ?? Path.Combine(options.ModelDirectory, $"{Path.GetFileNameWithoutExtension(inputPath)}.zip");
            PredictionService.Detect(predictionContext, inputPath, modelPath, options.Days);
        }
        else
        {
            PredictionService.DetectAll(predictionContext, options.DataPaths, options.ModelDirectory, options.Days);
        }
    }

    return 0;
}
catch (Exception exception) when (exception is not OutOfMemoryException)
{
    Console.Error.WriteLine($"Error: {exception.Message}");
    return 1;
}
