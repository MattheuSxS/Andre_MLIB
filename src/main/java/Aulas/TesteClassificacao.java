package Aulas;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.col;

public class TesteClassificacao {

    public static SparkSession spark;
    public static JavaSparkContext sc;

    public static void main(String[] args) throws AnalysisException {

        spark = SparkSession.builder().appName("Gym_Competitors").master("local[*]").getOrCreate();

        sc = new JavaSparkContext(spark.sparkContext());
        sc.setLogLevel("WARN");

        Dataset<Row> FileBase = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("hdfs://localhost:9000/Aula_Andre/titanic.csv");

//        FileBase.printSchema();
//        FileBase.show();

        // Vamos pegar apenas as colunas que interessam.
        // Todas: PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

        Dataset<Row> df = (FileBase
                .select(col("Survived").as("label")
                        ,col("Pclass")
                        ,col("Name")
                        ,col("Sex")
                        ,col("Age")
                        ,col("SibSp")
                        ,col("Parch")
                        ,col("Fare")
                        ,col("Embarked")));

        df.printSchema();
        df.show();

        Dataset<Row> dfClean = df.na().drop();

        //Spark nao sabe importar dados categoricos.
        //Precisamos codifica-los e transforma-los, antes de usa-los. Lidando com dados categoricos.
        StringIndexer genderIndexer = new StringIndexer()
                .setInputCol("Sex")
                .setOutputCol("SexIndex");
        StringIndexer embarkIndexer = new StringIndexer()
                .setInputCol("Embarked")
                .setOutputCol("EmbarkIndex");

        OneHotEncoderEstimator encoder = new OneHotEncoderEstimator()
                .setInputCols(new String[] {"SexIndex", "EmbarkIndex"})
                .setOutputCols(new String[] {"SexVec", "EmbarkVec"});

        // Assemble everything together to be ("label","features") format
        VectorAssembler assembler = (new VectorAssembler()
                .setInputCols(new String[]{"Pclass"
                        , "SexVec"
                        , "Age"
                        ,"SibSp"
                        ,"Parch"
                        ,"Fare"
                        ,"EmbarkVec"}))
                .setOutputCol("features");

        // Split the data.
        Dataset<Row> [] Split = dfClean.randomSplit(new double[] {0.7, 0.3});
        Dataset<Row> dfTrain = Split[0];
        Dataset<Row> dfTest = Split[1];

        // Escolhendo qual Machine Learning usar.
        LogisticRegression lr = new LogisticRegression();

        // Set Up the Pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]
                        { genderIndexer ,embarkIndexer ,encoder ,assembler ,lr });

        //Fit the pipeline to training documents.
        PipelineModel result = pipeline.fit(dfTrain);

        //Get Results on Test Set
        Dataset<Row> predictions = result.transform(dfTest);

        predictions.show();
        predictions.printSchema();

        //////////////////////////////////
        ////MODEL EVALUATION /////////////
        //////////////////////////////////
        //true positive
        double tp = 0.0;
        //true negative
        double tn = 0.0;
        //False positive
        double fp = 0.0;
        // False Negative
        double fn = 0.0;
        for (Row r : predictions.select("label", "prediction", "probability").collectAsList()) {
            System.out.println("Survived:" + r.get(0)
                    + ", Prediction:" + r.get(1) + ") --> prob=" + r.get(2));

            if(r.get(0).equals(1) && r.get(1).equals(1.0)) tp = tp + 1.0;
            else if(r.get(0).equals(0) && r.get(1).equals(0.0)) tn = tn + 1.0;
            else if(r.get(0).equals(0) && r.get(1).equals(1.0)) fp = fp + 1.0;
            else if(r.get(0).equals(1) && r.get(1).equals(0.0)) fn = fn + 1.0;
            else System.out.println("Isso nao devia acontecer!!!");
        }

        double total = (tp+tn+fp+fn); // Deve ser igual a test.count();
        System.out.println("True Positives : " + tp + " out of " + total + " (" +tp/total+ ");");
        System.out.println("True Negatives : " + tn + " out of " + total + " (" +tn/total+ ");");
        System.out.println("False Positives: " + fp + " out of " + total + " (" +fp/total+ ");");
        System.out.println("False Negatives: " + tn + " out of " + total + " (" +fn/total+ ");");

        System.out.println("Accuracy:" + (tp+tn)/(total) + " in test set of size " + dfTest.count() + ";");

    }
}
