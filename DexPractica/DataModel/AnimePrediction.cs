using Microsoft.ML.Data;

namespace DexPractica.AnimeData;

public class AnimePrediction
{
    [ColumnName("Score")]
    public float PredictedRating;
}