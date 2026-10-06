namespace Lab1;

internal enum Command
{
    All,
    Train,
    Predict
}

internal sealed record CliOptions(
    Command Command,
    string DataPath,
    string ModelPath,
    bool ShowHelp)
{
    public static CliOptions Parse(string[] args)
    {
        ArgumentNullException.ThrowIfNull(args);

        var projectDirectory = FindProjectDirectory();
        var command = Command.All;
        var dataPath = Path.Combine(projectDirectory, "Data", "TSLA.csv");
        var modelPath = Path.Combine(projectDirectory, "Artifacts", "tesla-price-model.zip");
        var showHelp = false;

        var index = 0;
        if (args.Length > 0 && !args[0].StartsWith('-'))
        {
            command = args[0].ToLowerInvariant() switch
            {
                "all" => Command.All,
                "train" => Command.Train,
                "predict" => Command.Predict,
                _ => throw new ArgumentException($"Unknown command: {args[0]}")
            };
            index = 1;
        }

        while (index < args.Length)
        {
            switch (args[index])
            {
                case "--data":
                    dataPath = ReadOptionValue(args, ref index, "--data");
                    break;
                case "--model":
                    modelPath = ReadOptionValue(args, ref index, "--model");
                    break;
                case "--help" or "-h":
                    showHelp = true;
                    index++;
                    break;
                default:
                    throw new ArgumentException($"Unknown option: {args[index]}");
            }
        }

        return new CliOptions(
            command,
            Path.GetFullPath(dataPath),
            Path.GetFullPath(modelPath),
            showHelp);
    }

    public static void PrintHelp()
    {
        Console.WriteLine(
            """
            Laboratory work 1, variant 1 - Tesla stock price prediction.

            Usage:
              dotnet run                         Train, evaluate, and predict
              dotnet run -- train                Train, evaluate, and save the model
              dotnet run -- predict              Load the model and predict
              dotnet run -- <command> --data <path> --model <path>

            Commands: all, train, predict
            """);
    }

    private static string ReadOptionValue(string[] args, ref int index, string optionName)
    {
        if (index + 1 >= args.Length)
        {
            throw new ArgumentException($"A value must be specified after {optionName}.");
        }

        index += 2;
        return args[index - 1];
    }

    private static string FindProjectDirectory()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);

        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "Lab1.csproj")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return AppContext.BaseDirectory;
    }
}
