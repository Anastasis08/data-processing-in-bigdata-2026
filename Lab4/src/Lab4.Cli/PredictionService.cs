using System.Globalization;
using Lab4.Model;
using Microsoft.ML;

namespace Lab4.Cli;

internal static class PredictionService
{
    public static void PredictApplications(MLContext mlContext, string inputPath, string onnxPath, string modelPath)
    {
        ArgumentNullException.ThrowIfNull(mlContext);

        var applications = LoanCsvReader.Read(inputPath, requireLabel: false);
        var onnxModel = OnnxLoanModel.Load(mlContext, onnxPath);
        using var onnxEngine = mlContext.Model.CreatePredictionEngine<LoanFeatures, OnnxLoanPrediction>(onnxModel);

        using var mlNetEngine = File.Exists(modelPath)
            ? mlContext.Model.CreatePredictionEngine<LoanFeatures, LoanPrediction>(mlContext.Model.Load(modelPath, out _))
            : null;

        Console.WriteLine("Part 2: loading the ONNX model and prediction");
        Console.WriteLine($"Model: {Path.GetFileName(onnxPath)} (ApplyOnnxModel, fitted on an empty data view)");
        Console.WriteLine($"Applicants: {Path.GetFileName(inputPath)} ({applications.Count} records)");
        Console.WriteLine();
        Console.WriteLine(
            $"{"Loan ID",-10} {"Income",8} {"Co-inc.",8} {"Amount",7} {"Term",5} {"Credit",7} {"Area",-10} " +
            $"{"P(ONNX)",8} {"P(ML.NET)",9}  {"Decision",-9} {"Actual",-6}");

        var correct = 0;
        var labelled = 0;
        var maxDifference = 0d;
        foreach (var application in applications)
        {
            var features = FeatureBuilder.Build(application);
            var prediction = onnxEngine.Predict(features);
            var reference = mlNetEngine?.Predict(features);
            var actual = application.IsApproved switch
            {
                true => "Y",
                false => "N",
                null => "-"
            };

            if (reference is not null)
            {
                maxDifference = Math.Max(maxDifference, Math.Abs(reference.Probability - prediction.ApprovalProbability));
            }

            if (application.IsApproved is { } isApproved)
            {
                labelled++;
                correct += isApproved == prediction.IsApproved ? 1 : 0;
            }

            Console.WriteLine(
                $"{application.LoanId,-10} {application.ApplicantIncome,8:F0} {application.CoapplicantIncome,8:F0} " +
                $"{FormatOptional(application.LoanAmount),7} {FormatOptional(application.LoanAmountTerm),5} " +
                $"{application.CreditHistory,7} {application.PropertyArea,-10} {prediction.ApprovalProbability,8:P1} " +
                $"{(reference is null ? "n/a" : reference.Probability.ToString("P1", CultureInfo.InvariantCulture)),9}  " +
                $"{(prediction.IsApproved ? "APPROVE" : "REJECT"),-9} {actual,-6}");
        }

        Console.WriteLine();
        if (mlNetEngine is not null)
        {
            Console.WriteLine($"Max |P(ONNX) - P(ML.NET)| = {maxDifference:E2}");
        }

        if (labelled > 0)
        {
            Console.WriteLine($"Accuracy on labelled input rows: {correct}/{labelled} ({(double)correct / labelled:P1})");
        }
    }

    private static string FormatOptional(float value) =>
        float.IsNaN(value) ? "n/a" : value.ToString("F0", CultureInfo.InvariantCulture);
}