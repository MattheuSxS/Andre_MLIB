package ML;

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

public class TesteRegressao {

    public static void main(String[] args) {

        // Instanciando um contexto Spark
        SparkSession spark = SparkSession
                .builder()
                .master("local[*]")
                .appName("Regressao Linear")
                .getOrCreate();

        Logger rootLogger = Logger.getRootLogger();
        rootLogger.setLevel(Level.ERROR);

        //regressao_linear_Carros(spark);
        regressao_linear_Casas(spark);

    }

    private static void regressao_linear_Casas(SparkSession spark) {

        // Vamos criar um Dataframe df_bruto a partir dos dados em nosso arquivo CSV.
        Dataset<Row> df_bruto = spark.read()
                .format("com.databricks.spark.csv")
                .option("inferschema", true)
                .option("header", true)
                .load("hdfs://localhost:9000/Aula_Andre/Clean-USA-Housing.csv");

        // Mostra o conteudo no Dataframe
        df_bruto.show();
        df_bruto.describe().show();

        for (String col: df_bruto.columns()) {
            System.out.println("A correlacao entre Price e " + col + " eh igual a: " + df_bruto.stat().corr("Price", col));
        }

        System.out.println("A correlacao entre Avg Area Number of Rooms e Avg Area Number of Bedrooms eh igual a: " + df_bruto.stat().corr("Avg Area Number of Rooms", "Avg Area Number of Bedrooms"));


        // Imprime o schema da tabela em formato de arvore
        df_bruto.printSchema();

        // Tratando o dataframe original para renomear a coluna de rótulo para label
		 /* Dataset<Row> df_tratado = df_bruto.select(col("Price").as("label")
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
				  //, "Avg Area Number of Bedrooms"
				  , "Area Population"})
				  .setOutputCol("features"));

		  // Transformando o dataframe para o formato adequado para modelos de regressão:
		  // dataframe com 2 colunas
		  // uma chamada label com o valor rotulado
		  // outra chamada features que contem um vetor das features

		  Dataset<Row> bases[] = assembler.transform(df_tratado).select("label", "features").randomSplit(new double[] {0.7, 0.3});
		  Dataset<Row> training = bases[0];
		  Dataset<Row> test = bases[1];

		  training.show();

		  LinearRegression lr = new LinearRegression();

		  // Fit the model.
		  LinearRegressionModel lrModel = lr.fit(training);

		  lrModel.transform(test).show();

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

		  //Vector x_new = Vectors.dense(79545.4, 5.7, 7.01, 4.1, 23087.0);
		  //System.out.println("\nPrediction: " + lrModel.predict(x_new));*/


    }

    private static void regressao_linear_Carros(SparkSession spark) {

        // Vamos criar um Dataframe df a partir de dados em nosso arquivo de dados.

        Dataset<Row> df = spark.read()
                .format("com.databricks.spark.csv")
                .option("inferschema", true)
                .option("header", true)
                .load("hdfs://localhost:9000/Aula_Andre/auto-miles-per-gallon.csv");

        // Mostra o conteudo no Dataframe
        df.show();

        // Imprime o schema da tabela em formto de arvore
        df.printSchema();

        df.createOrReplaceTempView("carros");

        Dataset<Row> df_tratado = spark.sql("select mpg as label, cylinders, displacement, horsepower, weight, modelyear from carros");

        // Montando o vetor de colunas com as features
        VectorAssembler assembler = (new VectorAssembler()
                .setInputCols(new String[] {"cylinders"
                        , "displacement"
                        , "horsepower"
                        , "weight"
                        , "modelyear"})
                .setOutputCol("features"));

        // Transformando o dataframe para o formato adequado para modelos de regressão:
        // dataframe com 2 colunas
        // uma chamada label com o valor rotulado
        // outra chamada features que contem um vetor das features
        Dataset<Row> training = assembler.transform(df_tratado).select("label", "features");
        training.show();

        LinearRegression lr = new LinearRegression();

        // Fit the model.
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

        Vector x_new = Vectors.dense(4, 91, 67, 1965, 82);
        System.out.println("\nPrediction: " + lrModel.predict(x_new));

    }

}
