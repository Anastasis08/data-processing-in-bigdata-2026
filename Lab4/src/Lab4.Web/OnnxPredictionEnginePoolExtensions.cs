using Microsoft.Extensions.ML;
using Microsoft.Extensions.Options;

namespace Lab4.Web;

public static class OnnxPredictionEnginePoolExtensions
{
    public static PredictionEnginePoolBuilder<TData, TPrediction> FromOnnxFile<TData, TPrediction>(
        this PredictionEnginePoolBuilder<TData, TPrediction> builder,
        string filePath,
        bool watchForChanges = true)
        where TData : class
        where TPrediction : class, new()
    {
        ArgumentNullException.ThrowIfNull(builder);
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);

        builder.Services.AddSingleton<OnnxModelLoader>();
        builder.Services
            .AddOptions<PredictionEnginePoolOptions<TData, TPrediction>>(Options.DefaultName)
            .Configure<OnnxModelLoader>((options, loader) =>
            {
                loader.Start(filePath, watchForChanges);
                options.ModelLoader = loader;
            })
            .ValidateOnStart();

        return builder;
    }
}