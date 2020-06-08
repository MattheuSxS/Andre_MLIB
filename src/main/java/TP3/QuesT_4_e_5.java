package TP3;

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

public class QuesT_4_e_5 {
    public static SparkSession spark;
    public static JavaSparkContext sc;

    public static void main(String[] args) throws AnalysisException {

        spark = SparkSession.builder().appName("Perguntas 4 e 5").master("local[*]").getOrCreate();
        sc = new JavaSparkContext(spark.sparkContext());
        sc.setLogLevel("WARN");

        Dataset<Row> FileBase = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("hdfs://localhost:9000/Aula_Andre/auto-miles-per-gallon.csv");

        // Olhando o Schema e o conteúdo do DataSet.
        FileBase.printSchema();
        FileBase.show();

        // Tirando os NAs
        Dataset<Row> dfClean = FileBase.na().drop();

        // Obs: Não foram levado em conta os dados "Missing Values" como [0].

        // Pegando a coluna 'label' e tirando a coluna "NAME", dado que, a mesma não é importante para o modelo.
        Dataset<Row> dfTratada = dfClean.select(col("MPG").as("label")
                ,col("CYLINDERS")
                ,col("DISPLACEMENT")
                ,col("HORSEPOWER")
                ,col("WEIGHT")
                ,col("ACCELERATION")
                ,col("MODELYEAR"));

        // Montando o vetor de colunas com as features
        VectorAssembler assembler = (new VectorAssembler()
                .setInputCols(new String[]
                        {"CYLINDERS"
                        ,"DISPLACEMENT"
                        ,"HORSEPOWER"
                        ,"WEIGHT"
                        ,"ACCELERATION"
                        ,"MODELYEAR"})
                .setOutputCol("features"));


        // Transformando o dataframe para o modelos de regressão deixando somente a label e features:
        Dataset<Row> dfTrain = assembler.transform(dfTratada).select("label", "features");

        // Olhando o resultado de todas as transformações.
        dfTrain.show();

        // Escolhendo qual Machine Learning usar.
        LinearRegression lr = new LinearRegression();

        // Fit the model.
        LinearRegressionModel lrModel = lr.fit(dfTrain);

        // Result das 20 primeiras linhas.
        System.out.println("Interceptação: " + lrModel.intercept() + "\nCoeficientes: " + lrModel.coefficients());
        lrModel.transform(dfTrain).show();

        // Olhando o resíduo das 20 primeiras linhas.
        LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
        trainingSummary.residuals().show();

        // Olhando o resultado por outras métricas.
        System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
        System.out.println("MSE: " + trainingSummary.meanSquaredError());
        System.out.println("r2: " + trainingSummary.r2());

        // Prever o consumo (MPG) de um Honda Civic

        // | Cilindros: 4 | Displacement: 91 | HP: 67 | Peso: 1965kg | Aceleração: 15 | Ano: 82 | //

        Vector search = Vectors.dense(4, 91.0, 67, 1965, 15.0, 82);
        System.out.println("\nPredição: " + lrModel.predict(search));
    }
}