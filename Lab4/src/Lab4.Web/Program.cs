using Lab4.Model;
using Lab4.Web;
using Microsoft.Extensions.ML;

var builder = WebApplication.CreateBuilder(args);

var onnxPath = Path.GetFullPath(
    builder.Configuration["Model:OnnxPath"] ?? throw new InvalidOperationException("Model:OnnxPath is not configured."),
    builder.Environment.ContentRootPath);

builder.Services.AddPredictionEnginePool<LoanFeatures, OnnxLoanPrediction>()
    .FromOnnxFile(onnxPath, watchForChanges: true);
builder.Services.AddValidation();
builder.Services.AddProblemDetails();
builder.Services.AddOpenApi();
builder.Services.AddHealthChecks();

var app = builder.Build();

app.UseExceptionHandler();
app.UseStatusCodePages();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseDefaultFiles();
app.UseStaticFiles();
app.MapHealthChecks("/health");
app.MapLoanApi();

await app.RunAsync();