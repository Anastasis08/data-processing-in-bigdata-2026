using Lab4.Model;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.ML;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Primitives;
using Microsoft.ML;

namespace Lab4.Web;

public sealed partial class OnnxModelLoader : ModelLoader, IDisposable
{
    private readonly MLContext _mlContext;
    private readonly ILogger<OnnxModelLoader> _logger;
    private readonly Lock _gate = new();
    private string _filePath = string.Empty;
    private ITransformer? _model;
    private CancellationTokenSource _reloadSource = new();
    private PhysicalFileProvider? _fileProvider;
    private IDisposable? _fileWatch;

    public OnnxModelLoader(IOptions<MLOptions> mlOptions, ILogger<OnnxModelLoader> logger)
    {
        ArgumentNullException.ThrowIfNull(mlOptions);

        _mlContext = mlOptions.Value.MLContext;
        _logger = logger;
    }

    public void Start(string filePath, bool watchForChanges)
    {
        _filePath = Path.GetFullPath(filePath);
        _model = OnnxLoanModel.Load(_mlContext, _filePath);
        LogModelLoaded(_filePath);

        if (watchForChanges)
        {
            _fileProvider = new PhysicalFileProvider(Path.GetDirectoryName(_filePath)!);
            var fileName = Path.GetFileName(_filePath);
            _fileWatch = ChangeToken.OnChange(() => _fileProvider.Watch(fileName), Reload);
        }
    }

    public override ITransformer GetModel() =>
        _model ?? throw new InvalidOperationException("The ONNX model loader has not been started.");

    public override IChangeToken GetReloadToken()
    {
        lock (_gate)
        {
            return new CancellationChangeToken(_reloadSource.Token);
        }
    }

    public void Dispose()
    {
        _fileWatch?.Dispose();
        _fileProvider?.Dispose();
        _reloadSource.Dispose();
    }

    private void Reload()
    {
        try
        {
            var model = OnnxLoanModel.Load(_mlContext, _filePath);
            CancellationTokenSource previous;
            lock (_gate)
            {
                _model = model;
                previous = _reloadSource;
                _reloadSource = new CancellationTokenSource();
            }

            LogModelLoaded(_filePath);
            previous.Cancel();
            previous.Dispose();
        }
        catch (Exception exception) when (exception is IOException or InvalidOperationException or FileNotFoundException)
        {
            LogReloadFailed(exception, _filePath);
        }
    }

    [LoggerMessage(Level = LogLevel.Information, Message = "ONNX model loaded from {Path}")]
    private partial void LogModelLoaded(string path);

    [LoggerMessage(Level = LogLevel.Warning, Message = "ONNX model reload from {Path} failed, the previous model stays active")]
    private partial void LogReloadFailed(Exception exception, string path);
}