namespace Lab4.Model;

public sealed record LoanApplication(
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