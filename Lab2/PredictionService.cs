using System.Globalization;
using Microsoft.ML;

namespace Lab2;

internal static class PredictionService
{
    public static void PredictApplications(MLContext mlContext, string inputPath, string modelPath)
    {
        ArgumentNullException.ThrowIfNull(mlContext);

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException(
                "The serialized model was not found. Run the train command first.",
                modelPath);
        }

        var applications = LoanCsvReader.Read(inputPath, requireLabel: false);
        var model = mlContext.Model.Load(modelPath, out _);
        using var engine = mlContext.Model.CreatePredictionEngine<LoanModelInput, LoanPrediction>(model);

        Console.WriteLine("Part 2: loading the model and prediction");
        Console.WriteLine($"Model: {Path.GetFileName(modelPath)}");
        Console.WriteLine($"Applicants: {Path.GetFileName(inputPath)} ({applications.Count} records)");
        Console.WriteLine();
        Console.WriteLine(
            $"{"Loan ID",-10} {"Income",8} {"Co-inc.",8} {"Amount",7} {"Term",5} {"Credit",7} {"Area",-10} " +
            $"{"P(approve)",10}  {"Decision",-9} {"Actual",-6}");

        var correct = 0;
        var labelled = 0;
        foreach (var application in applications)
        {
            var prediction = engine.Predict(FeatureBuilder.Build(application));
            var actual = application.IsApproved switch
            {
                true => "Y",
                false => "N",
                null => "-"
            };

            if (application.IsApproved is { } isApproved)
            {
                labelled++;
                correct += isApproved == prediction.IsApproved ? 1 : 0;
            }

            Console.WriteLine(
                $"{application.LoanId,-10} {application.ApplicantIncome,8:F0} {application.CoapplicantIncome,8:F0} " +
                $"{FormatOptional(application.LoanAmount),7} {FormatOptional(application.LoanAmountTerm),5} " +
                $"{application.CreditHistory,7} {application.PropertyArea,-10} {prediction.Probability,10:P1}  " +
                $"{(prediction.IsApproved ? "APPROVE" : "REJECT"),-9} {actual,-6}");
        }

        if (labelled > 0)
        {
            Console.WriteLine();
            Console.WriteLine($"Accuracy on labelled input rows: {correct}/{labelled} ({(double)correct / labelled:P1})");
        }
    }

    private static string FormatOptional(float value) => float.IsNaN(value) ? "n/a" : value.ToString("F0", CultureInfo.InvariantCulture);
}
