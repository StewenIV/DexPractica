using Microsoft.ML.Data;

namespace DexPractica.AnimeData;

public class AnimeData
{
    [LoadColumn(1)] public string Name { get; set; }
    [LoadColumn(2)] public string Genre { get; set; }
    [LoadColumn(3)] public string Type { get; set; }
    [LoadColumn(4)] public float Episodes { get; set; }
    [LoadColumn(5)] public float Rating { get; set; }
    [LoadColumn(6)] public float Members { get; set; }
}
