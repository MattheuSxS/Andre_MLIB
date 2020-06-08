package ML;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.col;

public class Classificador_PCA {

    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .master("local[*]")
                .appName("Problemas de classificacao")
                .getOrCreate();

        Logger rootLogger = Logger.getRootLogger();
        rootLogger.setLevel(Level.ERROR);

        Dataset<Row> df_bruto = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .option("delimiter", ";")
                .csv("hdfs://localhost:9000/Aula_Andre/bank-full.csv");

        df_bruto.show();
        df_bruto.describe().show();


		/*Dataset<Row> df_correlacoes = df_bruto.drop("job", "marital", "education", "default",
				"housing", "loan", "contact", "month", "day_of_week", "pdays", "poutcome", "y");

		for (String col1: df_correlacoes.columns()) {
			  for (String col2: df_correlacoes.columns()) {
				  System.out.println("A correlacao entre " + col1 + " e " + col2 + " eh igual a: " + df_bruto.stat().corr(col1, col2));
			  }
		  }

		*/

        System.out.println("Dataframe bruto com " + df_bruto.count() + " linhas");
        df_bruto.groupBy(col("y")).count().show();

        df_bruto.show();

        StringIndexer jobIndexer = new StringIndexer()
                .setInputCol("job")
                .setOutputCol("jobIndex");
        StringIndexer maritalIndexer = new StringIndexer()
                .setInputCol("marital")
                .setOutputCol("maritalIndex");
        StringIndexer educationIndexer = new StringIndexer()
                .setInputCol("education")
                .setOutputCol("educationIndex");
        StringIndexer defaultIndexer = new StringIndexer()
                .setInputCol("default")
                .setOutputCol("defaultIndex");
        StringIndexer housingIndexer = new StringIndexer()
                .setInputCol("housing")
                .setOutputCol("housingIndex");
        StringIndexer loanIndexer = new StringIndexer()
                .setInputCol("loan")
                .setOutputCol("loanIndex");
        StringIndexer contactIndexer = new StringIndexer()
                .setInputCol("contact")
                .setOutputCol("contactIndex");
        StringIndexer poutcomeIndexer = new StringIndexer()
                .setInputCol("poutcome")
                .setOutputCol("poutcomeIndex");
        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("y")
                .setOutputCol("labelIndex");

        OneHotEncoderEstimator encoder = new OneHotEncoderEstimator()
                .setInputCols(new String[] {"jobIndex"
                        , "maritalIndex"
                        , "educationIndex"
                        , "defaultIndex"
                        , "housingIndex"
                        , "loanIndex"
                        , "contactIndex"
                        , "poutcomeIndex"})
                .setOutputCols(new String[] {"jobVec"
                        , "maritalVec"
                        , "educationVec"
                        , "defaultVec"
                        , "housingVec"
                        , "loanVec"
                        , "contactVec"
                        , "poutcomeVec"});

        VectorAssembler assembler = (new VectorAssembler()
                .setInputCols(new String[]{"age","jobVec"
                        ,"maritalVec","educationVec"
                        ,"defaultVec","housingVec","loanVec"
                        ,"contactVec","duration"
                        ,"campaign","pdays"
                        ,"previous","poutcomeVec"
                        ,"nr_employed","cons_price_idx"
                        ,"cons_conf_idx","emp_var_rate","euribor3m"}))
                .setOutputCol("features");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{jobIndexer
                        ,maritalIndexer
                        ,educationIndexer
                        ,defaultIndexer
                        ,housingIndexer
                        ,loanIndexer
                        ,contactIndexer
                        ,poutcomeIndexer
                        ,labelIndexer
                        ,encoder
                        ,assembler});

        //Fit the pipeline to training documents.
        PipelineModel model = pipeline.fit(df_bruto);

        //Get Results on Test Set
        Dataset<Row> df_tratado = model.transform(df_bruto);
        System.out.println("Dataframe transformado:");
        df_tratado.show();

        Dataset<Row> df_model = df_tratado.select(col("labelIndex"), col("features"));
        System.out.println("Dataframe featurezado:");
        df_model.show();

        ////////////////////////////
        /// Split the Data ////////
        //////////////////////////

        StandardScaler scaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setWithMean(true)
                .setWithStd(true);

        // Compute summary statistics by fitting the StandardScaler.
        // Basically create a new object called scalerModel by using scaler.fit()
        // on the output of the VectorAssembler
        StandardScalerModel scalerModel = scaler.fit(df_model);

        // Normalize each feature to have unit standard deviation.
        // Use transform() off of this scalerModel object to create your scaledData
        Dataset<Row> scaledData = scalerModel.transform(df_model);
        System.out.println("Dataframe normalizado:");
        scaledData.show();

        // Now its time to use PCA to reduce the features to some principal components
        // Create a new PCA() object that will take in the scaledFeatures
        // and output the pcs features, use 4 principal components
        // Then fit this to the scaledData

        PCA pca = (new PCA()
                .setInputCol("scaledFeatures")
                .setOutputCol("pcaFeatures")
                .setK(4));

        PCAModel pca_model = pca.fit(scaledData);
        Dataset<Row> df_pca = pca_model.transform(scaledData);
        System.out.println("Dataframe com PCA:");
        df_pca.show();

        Dataset<Row>[] split = df_pca.randomSplit(new double[] {0.7, 0.3});
        Dataset<Row> train = split[0];
        Dataset<Row> test  = split[1];

        LogisticRegression lr = new LogisticRegression()
                .setLabelCol("labelIndex")
                .setFeaturesCol("pcaFeatures");

        LogisticRegressionModel lr_model = lr.fit(train);
        Dataset<Row> predictions = lr_model.transform(test);
        System.out.println("Base de teste tem " + predictions.count() + " linhas");
        predictions.groupBy(col("labelIndex")).count().show();

        //////////////////////////////////
        ////MODEL EVALUATION /////////////
        //////////////////////////////////

        //View results
        System.out.println("Logistic Regression Result sample :");
        predictions.select("labelIndex", "prediction", "features").show(5);

        //View confusion matrix
        System.out.println("Logistic Regression Confusion Matrix :");
        predictions.groupBy(col("labelIndex"), col("prediction")).count().show();

        //Accuracy computation
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("labelIndex")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Logistic Regression Accuracy = " + Math.round( accuracy * 100) + "%" );

    }

}
