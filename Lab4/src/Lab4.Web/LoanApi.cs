using System.ComponentModel.DataAnnotations;
using Lab4.Model;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.Extensions.ML;

namespace Lab4.Web;

public sealed record LoanApplicationRequest(
    [property: AllowedValues("Male", "Female", null)] string? Gender,
    [property: AllowedValues("Yes", "No", null)] string? Married,
    [property: AllowedValues("0", "1", "2", "3+", null)] string? Dependents,
    [property: AllowedValues("Graduate", "Not Graduate", null)] string? Education,
    [property: AllowedValues("Yes", "No", null)] string? SelfEmployed,
    [property: Range(0, 1_000_000)] float ApplicantIncome,
    [property: Range(0, 1_000_000)] float CoapplicantIncome,
    [property: Range(1, 10_000)] float? LoanAmount,
    [property: Range(1, 600)] float? LoanAmountTerm,
    [property: AllowedValues("0", "1", null)] string? CreditHistory,
    [property: AllowedValues("Urban", "Semiurban", "Rural", null)] string? PropertyArea)
{
    public LoanApplication ToApplication() => new(
        LoanId: "API",
        Gender: Gender ?? LoanCsvReader.MissingCategory,
        Married: Married ?? LoanCsvReader.MissingCategory,
        Dependents: Dependents ?? LoanCsvReader.MissingCategory,
        Education: Education ?? LoanCsvReader.MissingCategory,
        SelfEmployed: SelfEmployed ?? LoanCsvReader.MissingCategory,
        ApplicantIncome: ApplicantIncome,
        CoapplicantIncome: CoapplicantIncome,
        LoanAmount: LoanAmount ?? float.NaN,
        LoanAmountTerm: LoanAmountTerm ?? float.NaN,
        CreditHistory: CreditHistory ?? LoanCsvReader.MissingCategory,
        PropertyArea: PropertyArea ?? LoanCsvReader.MissingCategory,
        IsApproved: null);
}

public sealed record LoanDecisionResponse(bool IsApproved, string Decision, float ApprovalProbability);

public static class LoanApi
{
    public static IEndpointRouteBuilder MapLoanApi(this IEndpointRouteBuilder endpoints)
    {
        var group = endpoints.MapGroup("/api/loans").WithTags("Loans");

        group.MapPost("/predict", Predict)
            .WithName("PredictLoanApproval")
            .WithSummary("Scores a loan application with the ONNX model");

        return endpoints;
    }

    private static Ok<LoanDecisionResponse> Predict(
        LoanApplicationRequest request,
        PredictionEnginePool<LoanFeatures, OnnxLoanPrediction> predictionEnginePool)
    {
        var features = FeatureBuilder.Build(request.ToApplication());
        var prediction = predictionEnginePool.Predict(features);

        return TypedResults.Ok(new LoanDecisionResponse(
            prediction.IsApproved,
            prediction.IsApproved ? "APPROVE" : "REJECT",
            prediction.ApprovalProbability));
    }
}