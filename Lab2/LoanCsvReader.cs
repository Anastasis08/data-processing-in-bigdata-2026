using System.Globalization;

namespace Lab2;

internal static class LoanCsvReader
{
    public const string MissingCategory = "Unknown";

    private static readonly string[] RequiredColumns =
    [
        "Loan_ID", "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
        "Credit_History", "Property_Area"
    ];

    private const string LabelColumn = "Loan_Status";

    public static IReadOnlyList<LoanApplication> Read(string path, bool requireLabel)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("The CSV file was not found.", path);
        }

        using var reader = new StreamReader(path);
        var header = reader.ReadLine()
            ?? throw new InvalidDataException($"The file is empty: {path}");
        var columns = header.TrimStart('\uFEFF').Split(',')
            .Select(static (name, position) => (Name: name.Trim(), Position: position))
            .ToDictionary(static column => column.Name, static column => column.Position, StringComparer.Ordinal);

        var missingColumns = RequiredColumns.Where(name => !columns.ContainsKey(name)).ToArray();
        if (missingColumns.Length > 0)
        {
            throw new InvalidDataException($"Missing CSV columns: {string.Join(", ", missingColumns)}");
        }

        var hasLabel = columns.ContainsKey(LabelColumn);
        if (requireLabel && !hasLabel)
        {
            throw new InvalidDataException($"The training dataset must contain the {LabelColumn} column.");
        }

        var applications = new List<LoanApplication>();
        var lineNumber = 1;
        while (reader.ReadLine() is { } line)
        {
            lineNumber++;
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var fields = line.Split(',');
            if (fields.Length != columns.Count)
            {
                throw new InvalidDataException(
                    $"Line {lineNumber}: expected {columns.Count} values, found {fields.Length}.");
            }

            string Field(string name) => fields[columns[name]].Trim();

            try
            {
                applications.Add(new LoanApplication(
                    LoanId: Field("Loan_ID"),
                    Gender: Category(Field("Gender")),
                    Married: Category(Field("Married")),
                    Dependents: Category(Field("Dependents")),
                    Education: Category(Field("Education")),
                    SelfEmployed: Category(Field("Self_Employed")),
                    ApplicantIncome: RequiredNumber(Field("ApplicantIncome")),
                    CoapplicantIncome: RequiredNumber(Field("CoapplicantIncome")),
                    LoanAmount: OptionalNumber(Field("LoanAmount")),
                    LoanAmountTerm: OptionalNumber(Field("Loan_Amount_Term")),
                    CreditHistory: Category(Field("Credit_History")),
                    PropertyArea: Category(Field("Property_Area")),
                    IsApproved: hasLabel ? ParseLabel(Field(LabelColumn), requireLabel) : null));
            }
            catch (FormatException exception)
            {
                throw new InvalidDataException($"Line {lineNumber}: {exception.Message}", exception);
            }
        }

        if (applications.Count == 0)
        {
            throw new InvalidDataException($"The file contains no records: {path}");
        }

        return applications;
    }

    private static string Category(string value) =>
        string.IsNullOrEmpty(value) ? MissingCategory : value;

    private static float RequiredNumber(string value) =>
        float.Parse(value, NumberStyles.Float, CultureInfo.InvariantCulture);

    private static float OptionalNumber(string value) =>
        string.IsNullOrEmpty(value) ? float.NaN : RequiredNumber(value);

    private static bool? ParseLabel(string value, bool required) => value switch
    {
        "Y" => true,
        "N" => false,
        "" when !required => null,
        _ => throw new FormatException($"Invalid Loan_Status value '{value}', expected Y or N.")
    };
}
