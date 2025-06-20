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
            ,hasHeader:true, 
            separatorChar:',',
            allowQuoting:true);

        var filteredData = context.Data.FilterRowsByMissingValues(data, "Episodes", "Rating", "Members");
        filteredData = context.Data.FilterRowsByColumn(filteredData,"Episodes");
        filteredData = context.Data.FilterRowsByColumn(filteredData, "Rating", lowerBound: 1.0f, upperBound: 10.0f);
        filteredData = context.Data.FilterRowsByColumn(filteredData,"Members", lowerBound: 190);//170,190
        
        var count = 0;
        foreach (var columnInfo in filteredData.Preview(74000).RowView)
        {
            count++;
            Console.WriteLine($"{count}: Column: {columnInfo.Values[0].Key}, Type: {columnInfo.Values[0].Value}, " +
                              $"2: {columnInfo.Values[2].Value} {columnInfo.Values[2].Key}, " +
                              $"3: {columnInfo.Values[3].Value} {columnInfo.Values[3].Key}," +
                              $"4: {columnInfo.Values[4].Value} {columnInfo.Values[4].Key}");
        }

        Console.WriteLine($"{count}");
        var split = context.Data.TrainTestSplit(filteredData, testFraction: 0.2, seed: 0);
        
        /*var options = new FastForestRegressionTrainer.Options 
        { 
            LabelColumnName = "Rating", 
            FeatureColumnName = "Features", 
            FeatureFraction = 0.8,  
            FeatureFirstUsePenalty = 0.1, 
            NumberOfTrees = 50
        }; */

        var conveyor = context.Transforms.Text.FeaturizeText("GenreFeaturized", "Genre")
            .Append(context.Transforms.Categorical.OneHotEncoding("TypeEncoded", "Type"))
            .Append(context.Transforms.Categorical.OneHotEncoding("NameEncoded", "Name"))
            .Append(context.Transforms.Concatenate("FeaturesRaw", "GenreFeaturized", "TypeEncoded", "NameEncoded",
                "Episodes", "Members"))
            .Append(context.Transforms.NormalizeMinMax("Features", "FeaturesRaw"))
            /*.Append(context.Regression.Trainers.FastForest(options));*/
            /*.Append(context.Regression.Trainers.FastTree(labelColumnName: "Rating", featureColumnName: "Features"));*/
            //fastTree 0.68 lightGbm 0.68 FastTreeTweedie 0.67
            .Append(context.Regression.Trainers.LightGbm(labelColumnName: "Rating", featureColumnName: "Features",
                numberOfLeaves: 30, minimumExampleCountPerLeaf: 30, learningRate: 0.30f, numberOfIterations: 50));
        
        
        
        var model = conveyor.Fit(split.TrainSet);
        
        var predictions = model.Transform(split.TestSet);
        var metrics = context.Regression.Evaluate(predictions, labelColumnName: "Rating");

        Console.WriteLine($"R^2: {metrics.RSquared:0.##}");
        Console.WriteLine($"MAE: {metrics.MeanAbsoluteError:0.##}");
        Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:0.##}");

        // 6. Прогноз для новых данных
        var engine = context.Model.CreatePredictionEngine<AnimeData, AnimePrediction>(model);

        var example = new AnimeData
        {
            Genre = "Aku no Onna Kanbu",
            Type = "OVA",
            Episodes = 2,
            Members = 4229,
        }; //11745,Aku no Onna Kanbu,Hentai,OVA,2,6.78,4229

        var prediction = engine.Predict(example);
        Console.WriteLine($"Предсказанный рейтинг: {prediction.PredictedRating:0.00}");
    }
}