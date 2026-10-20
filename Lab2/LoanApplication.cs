using Microsoft.ML.Data;

namespace Lab2;

internal sealed record LoanApplication(
    string LoanId,
    string Gender,
    string Married,
    string Dependents,
    string Education,
    string SelfEmployed,
    float ApplicantIncome,
    float CoapplicantIncome,
    float LoanAmount,
    float LoanAmountTerm,
    string CreditHistory,
    string PropertyArea,
    bool? IsApproved);

public sealed class LoanModelInput
{
    public string Gender { get; init; } = string.Empty;
    public string Married { get; init; } = string.Empty;
    public string Dependents { get; init; } = string.Empty;
    public string Education { get; init; } = string.Empty;
    public string SelfEmployed { get; init; } = string.Empty;
    public string CreditHistory { get; init; } = string.Empty;
    public string PropertyArea { get; init; } = string.Empty;
    public float LogApplicantIncome { get; init; }
    public float LogCoapplicantIncome { get; init; }
    public float LogTotalIncome { get; init; }
    public float LogLoanAmount { get; init; }
    public float LoanAmountTerm { get; init; }
    public float PaymentToIncomeRatio { get; init; }
    public float LoanAmountMissing { get; init; }
    public float LoanAmountTermMissing { get; init; }

    [ColumnName("Label")]
    public bool Label { get; init; }
}

public sealed class LoanPrediction
{
    [ColumnName("PredictedLabel")]
    public bool IsApproved { get; init; }

    public float Probability { get; init; }

    public float Score { get; init; }
}
