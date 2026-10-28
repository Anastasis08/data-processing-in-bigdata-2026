namespace Lab3;

internal enum Command
{
    All,
    Train,
    Predict
}

internal sealed record CliOptions(
    Command Command,
    IReadOnlyList<string> DataPaths,
    string? InputPath,
    string ModelDirectory,
    string? ModelPath,
    int Days,
    bool ShowHelp)
{
    private const string ProjectFileName = "Lab3.csproj";
    private const int DefaultDays = 12;

    public static CliOptions Parse(string[] args)
    {
        ArgumentNullException.ThrowIfNull(args);

        var projectDirectory = FindProjectDirectory();
        var command = Command.All;
        var dataPath = Path.Combine(projectDirectory, "Data");
        string? inputPath = null;
        var modelDirectory = Path.Combine(projectDirectory, "Artifacts");
        string? modelPath = null;
        var days = DefaultDays;
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
                case "--input":
                    inputPath = Path.GetFullPath(ReadOptionValue(args, ref index, "--input"));
                    break;
                case "--models":
                    modelDirectory = ReadOptionValue(args, ref index, "--models");
                    break;
                case "--model":
                    modelPath = Path.GetFullPath(ReadOptionValue(args, ref index, "--model"));
                    break;
                case "--days":
                    var value = ReadOptionValue(args, ref index, "--days");
                    days = int.TryParse(value, out var parsed) && parsed > 0
                        ? parsed
                        : throw new ArgumentException($"--days must be a positive integer, got '{value}'.");
                    break;
                case "--help" or "-h":
                    showHelp = true;
                    index++;
                    break;
                default:
                    throw new ArgumentException($"Unknown option: {args[index]}");
            }
        }

        if (modelPath is not null && inputPath is null)
        {
            throw new ArgumentException("--model requires --input to name the KPI file to score.");
        }

        return new CliOptions(
            command,
            ResolveDataFiles(Path.GetFullPath(dataPath)),
            inputPath,
            Path.GetFullPath(modelDirectory),
            modelPath,
            days,
            showHelp);
    }

    public static void PrintHelp()
    {
        Console.WriteLine(
            """
            Laboratory work 3, variant 1 - anomaly detection in seasonal KPIs (RandomizedPcaTrainer).

            Usage:
              dotnet run                         Train, evaluate, save, then detect anomalies
              dotnet run -- train                Train, evaluate, and save one model per KPI
              dotnet run -- predict              Load the models and score the latest days of every KPI
              dotnet run -- predict --input <csv> [--model <zip>] [--days <n>]

            Commands: all, train, predict
              --data    KPI csv file or directory with csv files (default: Data/)
              --models  Directory for models and reports (default: Artifacts/)
              --input   Single KPI file to score; the label column is optional
              --model   Model for --input (default: <models>/<input name>.zip)
              --days    Number of most recent days to score (default: 12)
            """);
    }

    private static string[] ResolveDataFiles(string path)
    {
        if (File.Exists(path))
        {
            return [path];
        }

        if (!Directory.Exists(path))
        {
            throw new FileNotFoundException("The KPI data path was not found.", path);
        }

        var files = Directory.GetFiles(path, "*.csv").Order(StringComparer.OrdinalIgnoreCase).ToArray();
        return files.Length > 0 ? files : throw new FileNotFoundException("No csv files were found.", path);
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
            if (File.Exists(Path.Combine(directory.FullName, ProjectFileName)))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return Directory.GetCurrentDirectory();
    }
}
