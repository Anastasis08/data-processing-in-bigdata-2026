using Microsoft.ML;

namespace Lab4.Model;

public static class OnnxLoanModel
{
    public const string PredictedLabelColumn = "PredictedLabel";
    public const string ProbabilityColumn = "Probability";

    public static readonly string[] ExportedColumns = [PredictedLabelColumn, ProbabilityColumn];

    private static readonly string[] OnnxGraphOutputs = [$"{PredictedLabelColumn}.output", $"{ProbabilityColumn}.output"];

    public static ITransformer Load(MLContext mlContext, string onnxPath)
    {
        ArgumentNullException.ThrowIfNull(mlContext);

        if (!File.Exists(onnxPath))
        {
            throw new FileNotFoundException("The ONNX model was not found. Run the train command first.", onnxPath);
        }

        var pipeline = mlContext.Transforms.ApplyOnnxModel(
            outputColumnNames: OnnxGraphOutputs,
            inputColumnNames: FeatureBuilder.InputColumns,
            modelFile: onnxPath);

        var emptyData = mlContext.Data.LoadFromEnumerable(Array.Empty<LoanFeatures>());
        return pipeline.Fit(emptyData);
    }
}