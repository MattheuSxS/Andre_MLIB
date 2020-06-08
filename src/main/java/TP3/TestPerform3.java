package TP3;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


public class TestPerform3 {

    public static void main(String[] args) {

        // Instanciando um contexto Spark
        SparkSession spark = SparkSession
                .builder()
                .master("local[*]")
                .appName("Predição de Consumo")
                .getOrCreate();

        Logger rootLogger = Logger.getRootLogger();
        rootLogger.setLevel(Level.ERROR);

        regressao_linear_Carros(spark);


    }


    private static void regressao_linear_Carros(SparkSession spark) {

        // Vamos criar um Dataframe df a partir de dados em nosso arquivo de dados.

        Dataset<Row> df = spark.read()
                .format("com.databricks.spark.csv")
                .option("inferschema", true)
                .option("header", true)
                .load("hdfs://localhost:9000/Aula_Andre/auto-miles-per-gallon.csv");

        // Mostra o todo conteudo no Dataframe (Acho que por default só aparecem as primeiras 20 linhas)
        df.show();

        // Imprime o schema da tabela no formto de arvore
        df.printSchema();

        df.createOrReplaceTempView("TBLCARROS");

        Dataset<Row> df_tratado = spark.sql("SELECT MPG as label, CYLINDERS, DISPLACEMENT, HORSEPOWER, WEIGHT, "
                + "ACCELERATION, MODELYEAR FROM TBLCARROS");

        df_tratado.show();


        // Essa parte o professor falou que monta um vetor de colunas.
        VectorAssembler assembler = (new VectorAssembler()
                .setInputCols(new String[] {"CYLINDERS"
                        , "DISPLACEMENT"
                        , "HORSEPOWER"
                        , "WEIGHT"
                        , "ACCELERATION"
                        , "MODELYEAR"})
                .setOutputCol("features"));

        // Do que lembro essa parte transforma o dataset pra um formato que os modelo consegue ler
        Dataset<Row> training = assembler.transform(df_tratado).select("label", "features");
        training.show();


        //Copiei declaração do código do professor para  eu entender (MUDAR DEPOIS)
        LinearRegression lr = new LinearRegression();

        // Separando o modelo
        LinearRegressionModel lrModel = lr.fit(training);

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

        Vector x_new = Vectors.dense(4, 91.0, 67, 1965, 15.0, 82);
        System.out.println("\nPrediction: " + lrModel.predict(x_new));

    }

}