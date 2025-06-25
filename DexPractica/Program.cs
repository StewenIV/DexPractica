using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using DexPractica.AnimeData;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

class Program
{
    static void Main(string[] args)
    {
        var context = new MLContext();
        var path = Path.Combine(Directory.GetParent(Environment.CurrentDirectory)
            .Parent
            .Parent
            .FullName, "Data", "anime.csv");

        var data = context.Data.LoadFromTextFile<AnimeData>(path
            , hasHeader: true,
            separatorChar: ',',
            allowQuoting: true);


        var filteredData = context.Data.FilterRowsByMissingValues(data, "Episodes", "Rating", "Members");
        filteredData = context.Data.FilterRowsByColumn(filteredData, "Episodes", lowerBound: 1);
        filteredData = context.Data.FilterRowsByColumn(filteredData, "Rating", lowerBound: 3, upperBound: 9.5);
        filteredData =
            context.Data.FilterRowsByColumn(filteredData, "Members", lowerBound: 240);
        filteredData =
            context.Data.FilterByCustomPredicate<AnimeData>(filteredData,
                anime => string.IsNullOrWhiteSpace(anime.Genre));
        
        var split = context.Data.TrainTestSplit(filteredData, testFraction: 0.2, seed: 0);
        
        var conveyor = context.Transforms.Text.FeaturizeText("GenreFeaturized", "Genre")
            .Append(context.Transforms.Text.FeaturizeText("NameFeaturized", "Name"))
            .Append(context.Transforms.Categorical.OneHotHashEncoding("TypeEncoded", "Type"))
            .Append(context.Transforms.NormalizeLogMeanVariance("MembersNormalized", "Members"))
            .Append(context.Transforms.NormalizeLogMeanVariance("EpisodesNormalized", "Episodes"))
            .Append(context.Transforms.Concatenate("FeaturesRaw", "GenreFeaturized", "TypeEncoded",
                "EpisodesNormalized", "MembersNormalized", "NameFeaturized"))
            .Append(context.Transforms.NormalizeMinMax("Features", "FeaturesRaw"))
            .Append(context.Regression.Trainers.LightGbm(labelColumnName: "Rating", featureColumnName: "Features",
                numberOfLeaves: 64, minimumExampleCountPerLeaf: 30, learningRate: 0.1f, numberOfIterations: 150));

        var model = conveyor.Fit(split.TrainSet);

        var predictions = model.Transform(split.TestSet);
        var metrics = context.Regression.Evaluate(predictions, labelColumnName: "Rating");

        Console.WriteLine($"R^2: {metrics.RSquared:0.##}");
        Console.WriteLine($"MAE: {metrics.MeanAbsoluteError:0.##}");
        Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:0.##}");


        var engine = context.Model.CreatePredictionEngine<AnimeData, AnimePrediction>(model);
        var fullData =
            context.Data.LoadFromTextFile<AnimeData>(path, hasHeader: true, separatorChar: ',', allowQuoting: true);
        var animeList = context.Data.CreateEnumerable<AnimeData>(fullData, reuseRowObject: false)
            .Where(x => !string.IsNullOrEmpty(x.Name) && !string.IsNullOrEmpty(x.Genre) &&
                        !string.IsNullOrEmpty(x.Type))
            .Where(x => x.Rating > 0 && x.Members > 10000 && x.Episodes > 0)
            .OrderBy(_ => Guid.NewGuid())
            .Take(10)
            .ToList();
        animeList.AddRange(new[]
        {
            new AnimeData
            {
                Name = "Chrono Skater",
                Genre = "Action, Sci-Fi, Time Travel",
                Type = "TV",
                Episodes = 12,
                Members = 35000
            },
            new AnimeData
            {
                Name = "Mystic Bakery",
                Genre = "Slice of Life, Comedy, Fantasy",
                Type = "TV",
                Episodes = 24,
                Members = 47000
            }
        });


        Console.WriteLine(
            "│ ID │ Name                           │ Genre                          │ Ep.  │    Mem.  │ Real  │ Pred. │"
                .Replace(':', '-'));
        Console.WriteLine(
            "├────┼────────────────────────────────┼────────────────────────────────┼──────┼──────────┼───────┼───────┤");

        var i = 1;
        foreach (var anime in animeList)
        {
            var prediction = engine.Predict(anime);
            var name = (anime.Name ?? "").PadRight(30).Substring(0, 30);
            var genre = (anime.Genre ?? "").PadRight(30).Substring(0, 30);
            Console.WriteLine(
                $"│ {i,2} │ {name} │ {genre} │ {anime.Episodes,4} │ {anime.Members,8} │ {anime.Rating,5:0.00} │ {prediction.PredictedRating,5:0.00} │");
            i++;
        }
    }
}