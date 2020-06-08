package Aulas;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


public class GymCompetitors {

    public static SparkSession spark;
    public static JavaSparkContext sc;

    public static void main(String[] args) throws AnalysisException {

        spark = SparkSession.builder().appName("Gym_Competitors").master("local[*]").getOrCreate();

        sc = new JavaSparkContext(spark.sparkContext());
        sc.setLogLevel("WARN");

        Dataset<Row> FileBase = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("hdfs://localhost:9000/Aula_Andre/GymCompetition.csv");

        FileBase.printSchema();
        FileBase.show();

        VectorAssembler vectorAssembler = new VectorAssembler();
        vectorAssembler.setInputCols(new String[] {"Age","Height","Weight"});
        vectorAssembler.setOutputCol("features");

        Dataset<Row> dfFeatures = vectorAssembler.transform(FileBase);
        dfFeatures.show();

        Dataset<Row> modelInputData = dfFeatures.select("NoOfReps", "features")
                .withColumnRenamed("NoOfReps", "label");

        modelInputData.show();

        LinearRegression linearRegression = new LinearRegression();

        LinearRegressionModel model = linearRegression.fit(modelInputData);
        System.out.println("The model has intercept " + model.intercept() + " and coefficients " + model.coefficients());

        model.transform(modelInputData).show();
    }
}
