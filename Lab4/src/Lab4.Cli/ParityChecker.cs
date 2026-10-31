using Lab4.Model;
using Microsoft.ML;

namespace Lab4.Cli;

internal sealed record ParityResult(int Rows, double MaxProbabilityDifference, int MatchingDecisions);

internal static class ParityChecker
{
    public static ParityResult Compare(
        MLContext mlContext,
        ITransformer mlNetModel,
        ITransformer onnxModel,
        IReadOnlyCollection<LoanFeatures> rows)
    {
        ArgumentNullException.ThrowIfNull(mlContext);
        ArgumentNullException.ThrowIfNull(rows);

        using var mlNetEngine = mlContext.Model.CreatePredictionEngine<LoanFeatures, LoanPrediction>(mlNetModel);
        using var onnxEngine = mlContext.Model.CreatePredictionEngine<LoanFeatures, OnnxLoanPrediction>(onnxModel);

        var maxDifference = 0d;
        var matching = 0;
        foreach (var row in rows)
        {
            var expected = mlNetEngine.Predict(row);
            var actual = onnxEngine.Predict(row);
            maxDifference = Math.Max(maxDifference, Math.Abs(expected.Probability - actual.ApprovalProbability));
            matching += expected.IsApproved == actual.IsApproved ? 1 : 0;
        }

        return new ParityResult(rows.Count, maxDifference, matching);
    }
}