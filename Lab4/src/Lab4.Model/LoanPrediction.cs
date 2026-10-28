using Microsoft.ML.Data;

namespace Lab4.Model;

public sealed class LoanPrediction
{
    [ColumnName("PredictedLabel")]
    public bool IsApproved { get; init; }

    public float Probability { get; init; }

    public float Score { get; init; }
}

public sealed class OnnxLoanPrediction
{
    [ColumnName(OnnxLoanModel.PredictedLabelColumn)]
    public bool[] PredictedLabel { get; init; } = [];

    [ColumnName(OnnxLoanModel.ProbabilityColumn)]
    public float[] Probability { get; init; } = [];

    public bool IsApproved => PredictedLabel is [var value, ..] && value;

    public float ApprovalProbability => Probability is [var value, ..] ? value : float.NaN;
}