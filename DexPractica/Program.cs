using DexPractica.AnimeData;
using Microsoft.ML;

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
            ,hasHeader:true, separatorChar:',');
        
        var split = context.Data.TrainTestSplit(data, testFraction: 0.2);
        
        var pipeline = context.Transforms.Categorical.OneHotEncoding("Genre")
            .Append(context.Transforms.Categorical.OneHotEncoding("Type"))
            .Append(context.Transforms.Concatenate("FeaturesRaw", "Genre", "Type", "Episodes", "Members"))
            .Append(context.Transforms.NormalizeMinMax("Features", "FeaturesRaw"))
            .Append(context.Regression.Trainers.Sdca(labelColumnName: "Rating", featureColumnName: "Features"));

        var model = pipeline.Fit(split.TrainSet);
        
        var predictions = model.Transform(split.TestSet);
        var metrics = context.Regression.Evaluate(predictions, labelColumnName: "Rating");

        Console.WriteLine($"R^2: {metrics.RSquared:0.##}");
        Console.WriteLine($"MAE: {metrics.MeanAbsoluteError:0.##}");
        Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:0.##}");

        // 6. Прогноз для новых данных
        var engine = context.Model.CreatePredictionEngine<AnimeData, AnimePrediction>(model);

        var example = new AnimeData
        {
            Name = "MyAnime",
            Genre = "Action, Comedy",
            Type = "TV",
            Episodes = 12,
            Members = 50000
        };

        var prediction = engine.Predict(example);
        Console.WriteLine($"Предсказанный рейтинг: {prediction.PredictedRating:0.00}");
    }
}