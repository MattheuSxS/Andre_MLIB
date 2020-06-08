package Aulas;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.col;

public class TesteRegressao {

    public static SparkSession spark;
    public static JavaSparkContext sc;

    public static void main(String[] args) throws AnalysisException {

        spark = SparkSession.builder().appName("Gym_Competitors").master("local[*]").getOrCreate();

        sc = new JavaSparkContext(spark.sparkContext());
        sc.setLogLevel("WARN");

        regressao_linear_Carros(spark);
    }

    private static void regressao_linear_Carros(SparkSession spark){

        Dataset<Row> FileBase = spark.read()
        .option("header", true)
        .option("inferSchema", true)
        .csv("hdfs://localhost:9000/Aula_Andre/Clean-USA-Housing.csv");

        FileBase.show();
        FileBase.printSchema();

        Dataset<Row> dfTratada = FileBase.select(col("Price").as("label")
                , col("Avg Area Income")
                , col("Avg Area House Age")
                , col("Avg Area Number of Rooms")
                , col("Avg Area Number of Bedrooms")
                , col("Area Population"));

        // Montando o vetor de colunas com as features
        VectorAssembler assembler = (new VectorAssembler()
                .setInputCols(new String[] {"Avg Area Income"
                        , "Avg Area House Age"
                        , "Avg Area Number of Rooms"
                        , "Avg Area Number of Bedrooms"
                        , "Area Population"})
                .setOutputCol("features"));

        // Transformando o dataframe para o formato adequado para modelos de regressão:
        // dataframe com 2 colunas
        // uma chamada label com o valor rotulado
        // outra chamada features que contem um vetor das features
        Dataset<Row> dfTrain = assembler.transform(dfTratada).select("label", "features");

        dfTrain.show();

        // Escolhendo qual Machine Learning usar.
        LinearRegression lr = new LinearRegression();

        // Fit the model.
        LinearRegressionModel lrModel = lr.fit(dfTrain);

        // Print the coefficients and intercept for linear regression.
        System.out.println("Coefficients: "
                + lrModel.coefficients() + " Intercept: " + lrModel.intercept());

        // Summarize the model over the training set and print out some metrics.
        LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
        System.out.println("numIterations: " + trainingSummary.totalIterations());
        System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));
        trainingSummary.residuals().show();
        System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
        System.out.println("MSE: " + trainingSummary.meanSquaredError());
        System.out.println("r2: " + trainingSummary.r2());

        Vector x_new = Vectors.dense(79545.4, 5.7, 7.01, 4.1, 23087.0);
        System.out.println("\nPrediction: " + lrModel.predict(x_new));
    }
}
