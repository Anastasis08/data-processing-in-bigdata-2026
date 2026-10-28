namespace Lab4.Cli;

internal enum Command
{
    All,
    Train,
    Predict
}

internal sealed record CliOptions(
    Command Command,
    string DataPath,
    string InputPath,
    string ModelPath,
    bool ShowHelp)
{
    private const string ProjectFileName = "Lab4.slnx";

    public string OnnxPath => Path.ChangeExtension(ModelPath, ".onnx");

    public static CliOptions Parse(string[] args)
    {
        ArgumentNullException.ThrowIfNull(args);

        var projectDirectory = FindProjectDirectory();
        var command = Command.All;
        var dataPath = Path.Combine(projectDirectory, "Data", "loan_data_set.csv");
        var inputPath = Path.Combine(projectDirectory, "Data", "new_applicants.csv");
        var modelPath = Path.Combine(projectDirectory, "Artifacts", "loan-approval-model.zip");
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
                    inputPath = ReadOptionValue(args, ref index, "--input");
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
            Path.GetFullPath(inputPath),
            Path.GetFullPath(modelPath),
            showHelp);
    }

    public static void PrintHelp()
    {
        Console.WriteLine(
            """
            Laboratory work 4 - ONNX export of the loan approval model (task of laboratory work 2).

            Usage:
              dotnet run --project src/Lab4.Cli              Train, export, then predict
              dotnet run --project src/Lab4.Cli -- train     Train, evaluate, save the .zip and export the .onnx model
              dotnet run --project src/Lab4.Cli -- predict   Load the ONNX model (ApplyOnnxModel) and score new applicants
              dotnet run --project src/Lab4.Cli -- <command> --data <path> --input <path> --model <path>

            Commands: all, train, predict
              --data   Labelled training dataset (default: Data/loan_data_set.csv)
              --input  Applicants to score, Loan_Status is optional (default: Data/new_applicants.csv)
              --model  ML.NET model path, the .onnx file is stored next to it (default: Artifacts/loan-approval-model.zip)
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
            if (File.Exists(Path.Combine(directory.FullName, ProjectFileName)))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return Directory.GetCurrentDirectory();
    }
}
