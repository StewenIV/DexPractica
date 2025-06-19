using Microsoft.ML.Data;

namespace DexPractica.AnimeData;

public class AnimeData
{
    [LoadColumn(0)] public float AnimeId;
    [LoadColumn(1)] public string Name;
    [LoadColumn(2)] public string Genre;
    [LoadColumn(3)] public string Type;
    [LoadColumn(4)] public float Episodes;
    [LoadColumn(5)] public float Rating;
    [LoadColumn(6)] public float Members;
}