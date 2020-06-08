// CLASSIFICAÇÃO

package Aulas.Sabado;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;

public class Two_Desafio {

    public static SparkSession spark;
    public static JavaSparkContext sc;

    public static void main(String[] args) throws AnalysisException {

        spark = SparkSession.builder().appName("50_Startups").master("local[*]").getOrCreate();

        sc = new JavaSparkContext(spark.sparkContext());
        sc.setLogLevel("WARN");

        Dataset<Row> FileBase = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("hdfs://localhost:9000/Aula_Andre/Iris.csv");

        Dataset<Row> df = (FileBase
                .select(col("SepalLengthCm")
                        ,col("SepalWidthCm")
                        ,col("PetalLengthCm")
                        ,col("PetalWidthCm")
                        ,col("Species").as("label")));

        df.show();


        // PARA O DADO DE SAIDA
        StringIndexer ireIndexer = new StringIndexer()
                .setInputCol("Species")
                .setOutputCol("SpeciesIndex");

        OneHotEncoderEstimator encoder = new OneHotEncoderEstimator()
                .setInputCols(new String[] {"SpeciesIndex"})
                .setOutputCols(new String[] {"SpeciesVec"});

        VectorAssembler assembler = (new VectorAssembler()
                .setInputCols(new String[]{"SepalLengthCm"
                        , "SepalWidthCm"
                        , "MPetalLengthCm"
                        , "PetalWidthCm"}))
                .setOutputCol("features");

        Dataset<Row>[] split = df
                .randomSplit(new double[] {0.7, 0.3});
        Dataset<Row> dfTrain = split[0];
        Dataset<Row> dfTest  = split[1];

        dfTrain.show();
        dfTest.show();

        LogisticRegression lr = new LogisticRegression();

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {
                         ireIndexer
                        ,encoder
                        ,assembler
                        ,lr});

        //Fit the pipeline to training documents.
        PipelineModel model = pipeline.fit(dfTrain);

        //Get Results on Test Set
        Dataset<Row> predictions = model.transform(dfTest);

        predictions.show();
        predictions.printSchema();
    }
}
