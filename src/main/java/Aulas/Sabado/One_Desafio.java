// REGRESSÃO

package Aulas.Sabado;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import static org.apache.spark.sql.functions.col;

public class One_Desafio {

    public static SparkSession spark;
    public static JavaSparkContext sc;

    public static void main(String[] args) throws AnalysisException {

        spark = SparkSession.builder().appName("50_Startups").master("local[*]").getOrCreate();

        sc = new JavaSparkContext(spark.sparkContext());
        sc.setLogLevel("WARN");

        Dataset<Row> FileBase = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("hdfs://localhost:9000/Aula_Andre/50_Startups.csv");

//        FileBase.show();
//
        Dataset<Row> df = (FileBase
        .select(col("R&D Spend")
                ,col("Administration")
                ,col("Marketing Spend")
                ,col("State")
                ,col("Profit").as("label")));

        df.show();

        // PARA O DADO DE SAIDA
        StringIndexer genderIndexer = new StringIndexer()
                .setInputCol("State")
                .setOutputCol("StateIndex");

        StringIndexerModel stateModel = genderIndexer.fit(df);
        Dataset<Row> FileTratado = stateModel.transform(df);
        FileTratado.show();

        // PARA O DADO DE ENTRADA
        OneHotEncoderEstimator genderEncoder = new OneHotEncoderEstimator()
                .setInputCols(new String[]{"StateIndex"})
                .setOutputCols(new String[] {"StateVec"});

        OneHotEncoderModel encoderModel = genderEncoder.fit(FileTratado);
        FileTratado = encoderModel.transform(FileTratado);

        FileTratado.show();

        VectorAssembler vectorAssembler = new VectorAssembler();
        vectorAssembler.setInputCols(new String[] {"R&D Spend","Administration","Marketing Spend", "StateIndex"});
        vectorAssembler.setOutputCol("features");

        Dataset<Row> dfFeatures = vectorAssembler.transform(FileTratado);
        dfFeatures.show();

        Dataset<Row> InputData = dfFeatures.select("label", "features");

        InputData.show();

        LinearRegression linearRegression = new LinearRegression();
        LinearRegressionModel model = linearRegression.fit(InputData);
        System.out.println("The model has intercept " + model.intercept() + " and coefficients " + model.coefficients());

        model.transform(InputData).show();

    }
}
