namespace Lab3;

internal sealed record DetectionMetrics(
    int TruePositives,
    int FalsePositives,
    int FalseNegatives,
    double Precision,
    double Recall,
    double F1Score)
{
    public static DetectionMetrics From(int truePositives, int falsePositives, int falseNegatives)
    {
        var precision = truePositives + falsePositives == 0 ? 0 : (double)truePositives / (truePositives + falsePositives);
        var recall = truePositives + falseNegatives == 0 ? 0 : (double)truePositives / (truePositives + falseNegatives);
        var f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
        return new DetectionMetrics(truePositives, falsePositives, falseNegatives, precision, recall, f1);
    }
}

internal sealed record AnomalySegment(int Start, int End)
{
    public int Length => End - Start + 1;
}

internal static class AnomalyMetrics
{
    public const int DetectionDelay = 7;

    public static DetectionMetrics PointWise(IReadOnlyList<bool?> labels, IReadOnlyList<bool> predictions)
    {
        var (truePositives, falsePositives, falseNegatives) = (0, 0, 0);
        for (var index = 0; index < labels.Count; index++)
        {
            switch (labels[index], predictions[index])
            {
                case (true, true):
                    truePositives++;
                    break;
                case (false, true):
                    falsePositives++;
                    break;
                case (true, false):
                    falseNegatives++;
                    break;
            }
        }

        return DetectionMetrics.From(truePositives, falsePositives, falseNegatives);
    }

    public static DetectionMetrics DelayAdjusted(IReadOnlyList<bool?> labels, IReadOnlyList<bool> predictions) =>
        PointWise(labels, AdjustPredictions(labels, predictions));

    public static bool[] AdjustPredictions(IReadOnlyList<bool?> labels, IReadOnlyList<bool> predictions)
    {
        var adjusted = predictions.ToArray();
        foreach (var segment in Segments(labels.Select(static label => label == true).ToArray()))
        {
            var detectionEnd = Math.Min(segment.End, segment.Start + DetectionDelay);
            var detected = false;
            for (var index = segment.Start; index <= detectionEnd && !detected; index++)
            {
                detected = predictions[index];
            }

            for (var index = segment.Start; index <= segment.End; index++)
            {
                adjusted[index] = detected;
            }
        }

        return adjusted;
    }

    public static List<AnomalySegment> Segments(IReadOnlyList<bool> flags)
    {
        var segments = new List<AnomalySegment>();
        var start = -1;

        for (var index = 0; index <= flags.Count; index++)
        {
            var flagged = index < flags.Count && flags[index];
            if (flagged && start < 0)
            {
                start = index;
            }
            else if (!flagged && start >= 0)
            {
                segments.Add(new AnomalySegment(start, index - 1));
                start = -1;
            }
        }

        return segments;
    }

    public static (float Threshold, DetectionMetrics Metrics) BestThreshold(
        IReadOnlyList<bool?> labels,
        IReadOnlyList<float> scores)
    {
        var sorted = scores.Where(float.IsFinite).Order().ToArray();
        var candidates = Enumerable.Range(0, 400)
            .Select(step => sorted[(int)((sorted.Length - 1) * (0.9 + 0.1 * step / 400.0))])
            .Concat(scores.Where((_, index) => labels[index] == true))
            .Distinct()
            .ToArray();

        var best = (Threshold: sorted[^1], Metrics: DetectionMetrics.From(0, 0, 0));
        foreach (var threshold in candidates)
        {
            var metrics = DelayAdjusted(labels, [.. scores.Select(score => score >= threshold)]);
            if (metrics.F1Score > best.Metrics.F1Score)
            {
                best = (threshold, metrics);
            }
        }

        return best;
    }
}
