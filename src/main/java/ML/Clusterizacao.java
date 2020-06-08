package ML;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Clusterizacao {

    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .master("local[*]")
                .appName("Problemas de classificacao")
                .getOrCreate();

        Logger rootLogger = Logger.getRootLogger();
        rootLogger.setLevel(Level.ERROR);

        kmeans_gym(spark);
    }

    private static void kmeans_gym(SparkSession spark) {

        Dataset<Row> csvData = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("hdfs://localhost:9000/Aula_Andre/GymCompetition.csv");

        csvData.printSchema();
        csvData.show();

        StringIndexer genderIndexer = new StringIndexer();
        genderIndexer.setInputCol("Gender");
        genderIndexer.setOutputCol("GenderIndex");
        csvData = genderIndexer.fit(csvData).transform(csvData);

        OneHotEncoderEstimator genderEncoder = new OneHotEncoderEstimator();
        genderEncoder.setInputCols(new String[] {"GenderIndex"});
        genderEncoder.setOutputCols(new String[] {"GenderVector"});
        csvData = genderEncoder.fit(csvData).transform(csvData);
        csvData.show();

        VectorAssembler vectorAssembler = new VectorAssembler();
        vectorAssembler.setInputCols(new String[] {"GenderVector", "Age","Height","Weight", "NoOfReps"});
        vectorAssembler.setOutputCol("features");

        Dataset<Row> csvDataWithFeatures = vectorAssembler.transform(csvData);
        csvDataWithFeatures.show();

        Dataset<Row> modelInputData = csvDataWithFeatures.select("features");
        modelInputData.show();

        KMeans kMeans = new KMeans();

        for (int k=2; k <= 8; k++) {

            kMeans.setK(k);

            System.out.println("\nNo of clusters " + k);

            KMeansModel model = kMeans.fit(modelInputData);
            Dataset<Row> predictions = model.transform(modelInputData);
            predictions.show();

            predictions.groupBy("prediction").count().show();

            ClusteringEvaluator evaluator = new ClusteringEvaluator();
            double silhouette = evaluator.evaluate(predictions);
            System.out.println("Slihouette with squared euclidean distance is " + silhouette);

            // Shows the result.
            Vector[] centers = model.clusterCenters();
            System.out.println("Cluster Centers: ");
            for (Vector center: centers) {
                System.out.println(center);
            }
        }
    }

}