using Microsoft.ML.Data;

namespace Lab4.Model;

public class LoanFeatures
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
}

public sealed class LoanTrainingRow : LoanFeatures
{
    [ColumnName("Label")]
    public bool Label { get; init; }
}