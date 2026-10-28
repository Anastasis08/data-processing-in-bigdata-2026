using Google.Protobuf;
using Lab4.Model;
using Microsoft.ML;

namespace Lab4.Cli;

internal static class OnnxExporter
{
    private const int ModelGraphField = 7;
    private const int GraphValueInfoField = 13;

    public static void Export(MLContext mlContext, ITransformer model, IDataView featureSchemaData, string onnxPath)
    {
        ArgumentNullException.ThrowIfNull(mlContext);

        using var buffer = new MemoryStream();
        mlContext.Model.ConvertToOnnx(model, featureSchemaData, buffer, OnnxLoanModel.ExportedColumns);

        var cleanedModel = RewriteMessage(
            buffer.ToArray(),
            static (field, value) => field == ModelGraphField
                ? ByteString.CopyFrom(RewriteMessage(value.ToByteArray(), static (graphField, graphValue) =>
                    graphField == GraphValueInfoField ? null : graphValue))
                : value);

        File.WriteAllBytes(onnxPath, cleanedModel);
    }

    private static byte[] RewriteMessage(byte[] message, Func<int, ByteString, ByteString?> transform)
    {
        var input = new CodedInputStream(message);
        using var buffer = new MemoryStream();
        var output = new CodedOutputStream(buffer);

        uint tag;
        while ((tag = input.ReadTag()) != 0)
        {
            var wireType = WireFormat.GetTagWireType(tag);
            if (wireType == WireFormat.WireType.LengthDelimited)
            {
                if (transform(WireFormat.GetTagFieldNumber(tag), input.ReadBytes()) is { } value)
                {
                    output.WriteTag(tag);
                    output.WriteBytes(value);
                }

                continue;
            }

            output.WriteTag(tag);
            switch (wireType)
            {
                case WireFormat.WireType.Varint:
                    output.WriteUInt64(input.ReadUInt64());
                    break;
                case WireFormat.WireType.Fixed64:
                    output.WriteFixed64(input.ReadFixed64());
                    break;
                case WireFormat.WireType.Fixed32:
                    output.WriteFixed32(input.ReadFixed32());
                    break;
                default:
                    throw new InvalidDataException($"Unsupported protobuf wire type {wireType}.");
            }
        }

        output.Flush();
        return buffer.ToArray();
    }
}