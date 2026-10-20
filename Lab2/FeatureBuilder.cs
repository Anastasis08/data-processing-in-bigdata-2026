namespace Lab2;

internal static class FeatureBuilder
{
    public static readonly string[] CategoricalColumns =
    [
        nameof(LoanModelInput.Gender),
        nameof(LoanModelInput.Married),
        nameof(LoanModelInput.Dependents),
        nameof(LoanModelInput.Education),
        nameof(LoanModelInput.SelfEmployed),
        nameof(LoanModelInput.CreditHistory),
        nameof(LoanModelInput.PropertyArea)
    ];

    public static readonly string[] ImputedNumericColumns =
    [
        nameof(LoanModelInput.LogLoanAmount),
        nameof(LoanModelInput.LoanAmountTerm),
        nameof(LoanModelInput.PaymentToIncomeRatio)
    ];

    public static readonly string[] NumericColumns =
    [
        nameof(LoanModelInput.LogApplicantIncome),
        nameof(LoanModelInput.LogCoapplicantIncome),
        nameof(LoanModelInput.LogTotalIncome),
        .. ImputedNumericColumns,
        nameof(LoanModelInput.LoanAmountMissing),
        nameof(LoanModelInput.LoanAmountTermMissing)
    ];

    public static LoanModelInput Build(LoanApplication application)
    {
        ArgumentNullException.ThrowIfNull(application);

        var totalIncome = application.ApplicantIncome + application.CoapplicantIncome;
        var monthlyPayment = application.LoanAmount * 1000f / application.LoanAmountTerm;

        return new LoanModelInput
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
