namespace Lab4.Model;

public static class FeatureBuilder
{
    public static readonly string[] CategoricalColumns =
    [
        nameof(LoanFeatures.Gender),
        nameof(LoanFeatures.Married),
        nameof(LoanFeatures.Dependents),
        nameof(LoanFeatures.Education),
        nameof(LoanFeatures.SelfEmployed),
        nameof(LoanFeatures.CreditHistory),
        nameof(LoanFeatures.PropertyArea)
    ];

    public static readonly string[] ImputedNumericColumns =
    [
        nameof(LoanFeatures.LogLoanAmount),
        nameof(LoanFeatures.LoanAmountTerm),
        nameof(LoanFeatures.PaymentToIncomeRatio)
    ];

    public static readonly string[] NumericColumns =
    [
        nameof(LoanFeatures.LogApplicantIncome),
        nameof(LoanFeatures.LogCoapplicantIncome),
        nameof(LoanFeatures.LogTotalIncome),
        .. ImputedNumericColumns,
        nameof(LoanFeatures.LoanAmountMissing),
        nameof(LoanFeatures.LoanAmountTermMissing)
    ];

    public static readonly string[] InputColumns = [.. CategoricalColumns, .. NumericColumns];

    public static LoanFeatures Build(LoanApplication application) => BuildTrainingRow(application);

    public static LoanTrainingRow BuildTrainingRow(LoanApplication application)
    {
        ArgumentNullException.ThrowIfNull(application);

        var totalIncome = application.ApplicantIncome + application.CoapplicantIncome;
        var monthlyPayment = application.LoanAmount * 1000f / application.LoanAmountTerm;

        return new LoanTrainingRow
        {
            Gender = application.Gender,
            Married = application.Married,
            Dependents = application.Dependents,
            Education = application.Education,
            SelfEmployed = application.SelfEmployed,
            CreditHistory = application.CreditHistory,
            PropertyArea = application.PropertyArea,
            LogApplicantIncome = MathF.Log(1f + application.ApplicantIncome),
            LogCoapplicantIncome = MathF.Log(1f + application.CoapplicantIncome),
            LogTotalIncome = MathF.Log(1f + totalIncome),
            LogLoanAmount = MathF.Log(1f + application.LoanAmount),
            LoanAmountTerm = application.LoanAmountTerm,
            PaymentToIncomeRatio = totalIncome > 0f ? monthlyPayment / totalIncome : float.NaN,
            LoanAmountMissing = float.IsNaN(application.LoanAmount) ? 1f : 0f,
            LoanAmountTermMissing = float.IsNaN(application.LoanAmountTerm) ? 1f : 0f,
            Label = application.IsApproved ?? false
        };
    }
}