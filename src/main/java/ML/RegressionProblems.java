package ML;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class RegressionProblems {

    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .master("local[*]")
                .appName("Problemas de classificacao")
                .getOrCreate();

        Logger rootLogger = Logger.getRootLogger();
        rootLogger.setLevel(Level.ERROR);

        regressao_linear_startups(spark);
    }

    private static void regressao_linear_startups(SparkSession spark) {

        // Vamos criar um Dataframe df_bruto a partir dos dados em nosso arquivo CSV.
        Dataset<Row> dado_bruto = spark
                .read()
                .option("header",true)
                .option("inferSchema",true)
                .format("csv")
                .load("hdfs://localhost:9000/Aula_Andre/50_Startups.csv");

        // Mostra o conteudo no Dataframe
        dado_bruto.show();

        // Imprime o schema da tabela em formato de arvore
        dado_bruto.printSchema();

        //Spark nao sabe importar dados categoricos.
        //Precisamos codifica-los e transforma-los, antes de usa-los.
        //Lidando com dados categoricos.
        StringIndexer stateIndexer = new StringIndexer()
                .setInputCol("State")
                .setOutputCol("stateIndex");

        StringIndexerModel stateModel = stateIndexer.fit(dado_bruto);
        Dataset<Row> df_tratado = stateModel.transform(dado_bruto);
        df_tratado.show();

        OneHotEncoderEstimator encoder = new OneHotEncoderEstimator()
                .setInputCols(new String[] {"stateIndex"})
                .setOutputCols(new String[] {"stateVec"});

        OneHotEncoderModel encoderModel = encoder.fit(df_tratado);
        df_tratado = encoderModel.transform(df_tratado);
        df_tratado.show();

        // Tratando o dataframe original para renomear a coluna de rótulo para label
        df_tratado = df_tratado.withColumnRenamed("Profit", "label");

        // Montando o vetor de colunas com as features VectorAssembler assembler =
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[] {"R&D Spend" ,
                        "Administration" ,
                        "Marketing Spend" ,
                        "stateVec"})
                .setOutputCol("features");

        // Transformando o dataframe para o formato adequado para modelos de regressão:
        // dataframe com 2 colunas
        // uma chamada label com o valor rotulado
        // outra chamada features que contem um vetor das features
        Dataset<Row> df_preparado = assembler.transform(df_tratado).select("label","features");
        Dataset<Row> bases[] = df_preparado.randomSplit(new double[] {0.7, 0.3});
        Dataset<Row> training = bases[0];
        Dataset<Row> test = bases[1];

        LinearRegression lr = new LinearRegression();

        // Fit the model.
        LinearRegressionModel lrModel = lr.fit(training);

        Dataset<Row> predictions = lrModel.transform(test);

        // Print the coefficients and intercept for linear regression.
        System.out.println("Coefficients: " + lrModel.coefficients() + " Intercept: "
                + lrModel.intercept());

        // Summarize the model over the training set and print out some metrics.
        LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
        System.out.println("numIterations: " + trainingSummary.totalIterations());
        System.out.println("objectiveHistory: " +
                Vectors.dense(trainingSummary.objectiveHistory()));
        trainingSummary.residuals().show(); System.out.println("RMSE: " +
                trainingSummary.rootMeanSquaredError()); System.out.println("MSE: " +
                trainingSummary.meanSquaredError()); System.out.println("r2: " +
                trainingSummary.r2());


    }

}