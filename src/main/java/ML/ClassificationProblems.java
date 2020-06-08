package ML;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionSummary;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class ClassificationProblems {

    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .master("local[*]")
                .appName("Problemas de classificacao")
                .getOrCreate();

        Logger rootLogger = Logger.getRootLogger();
        rootLogger.setLevel(Level.ERROR);

        //regressao_logistica_flores(spark);
        arvore_decisao_flores(spark);
    }


    private static void arvore_decisao_flores(SparkSession spark) {

        Dataset<Row> dado_bruto = spark
                .read()
                .option("header",true)
                .option("inferSchema",true)
                .format("csv")
                .load("hdfs://localhost:9000/Aula_Andre/Iris.csv");

        dado_bruto.printSchema();
        dado_bruto.show();

        //Spark nao sabe importar dados categoricos.
        //Precisamos codifica-los e transforma-los, antes de usa-los.
        //Lidando com dados categoricos.
        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("Species")
                .setOutputCol("label");

        StringIndexerModel labelModel = labelIndexer.fit(dado_bruto);
        Dataset<Row> df_tratado = labelModel.transform(dado_bruto);

        // Assemble everything together to be ("label","features") format
        VectorAssembler assembler = (new VectorAssembler()
                .setInputCols(new String[]{"SepalLengthCm"
                        , "SepalWidthCm"
                        , "PetalLengthCm"
                        ,"PetalWidthCm"}))
                .setOutputCol("features");


        df_tratado = assembler.transform(df_tratado).select("label", "features");

        ////////////////////////////
        /// Split the Data ////////
        //////////////////////////
        Dataset<Row>[] split = df_tratado.randomSplit(new double[] {0.7, 0.3});
        Dataset<Row> train = split[0];
        Dataset<Row> test  = split[1];

        DecisionTreeClassifier dt = new DecisionTreeClassifier();
        dt.setMaxDepth(3);

        DecisionTreeClassificationModel model = dt.fit(train);

        //Get Results on Test Set
        Dataset<Row> predictions = model.transform(test);
        predictions.show();

        //////////////////////////////////
        ////MODEL EVALUATION /////////////
        //////////////////////////////////
        System.out.println(model.toDebugString());

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
        evaluator.setMetricName("accuracy");
        System.out.println("The accuracy of the model is " + evaluator.evaluate(predictions));

        for (Row r : predictions.select("label", "prediction", "probability").collectAsList()) {
            System.out.println("Tipo de flor:" + r.get(0)
                    + ", Prediction:" + r.get(1) + ") --> prob=" + r.get(2));
        }

    }

    private static void regressao_logistica_flores(SparkSession spark) {

        Dataset<Row> dado_bruto = spark
                .read()
                .option("header",true)
                .option("inferSchema",true)
                .format("csv")
                .load("hdfs://localhost:9000/Aula_Andre/Iris.csv");

        dado_bruto.printSchema();
        dado_bruto.show();

        //Spark nao sabe importar dados categoricos.
        //Precisamos codifica-los e transforma-los, antes de usa-los.
        //Lidando com dados categoricos.
        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("Species")
                .setOutputCol("label");

        StringIndexerModel labelModel = labelIndexer.fit(dado_bruto);
        Dataset<Row> df_tratado = labelModel.transform(dado_bruto);

        // Assemble everything together to be ("label","features") format
        VectorAssembler assembler = (new VectorAssembler()
                .setInputCols(new String[]{"SepalLengthCm"
                        , "SepalWidthCm"
                        , "PetalLengthCm"
                        ,"PetalWidthCm"}))
                .setOutputCol("features");


        df_tratado = assembler.transform(df_tratado).select("label", "features");

        ////////////////////////////
        /// Split the Data ////////
        //////////////////////////
        Dataset<Row>[] split = df_tratado.randomSplit(new double[] {0.7, 0.3});
        Dataset<Row> train = split[0];
        Dataset<Row> test  = split[1];

        /////////////////////////////
        //Set Up the Pipeline ///////
        /////////////////////////////
        LogisticRegression lr = new LogisticRegression();

        //Fit the pipeline to training documents.
        LogisticRegressionModel model = lr.fit(train);

        //Get Results on Test Set
        Dataset<Row> predictions = model.transform(test);

        predictions.show();

        //////////////////////////////////
        ////MODEL EVALUATION /////////////
        //////////////////////////////////
        LogisticRegressionSummary summary = model.evaluate(test);
        System.out.println("A precisao do modelo na base de treino eh " + summary.accuracy());

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
        evaluator.setMetricName("accuracy");
        System.out.println("The accuracy of the model is " + evaluator.evaluate(predictions));

        for (Row r : predictions.select("label", "prediction", "probability").collectAsList()) {
            System.out.println("Tipo de flor:" + r.get(0)
                    + ", Prediction:" + r.get(1) + ") --> prob=" + r.get(2));
        }
    }

}