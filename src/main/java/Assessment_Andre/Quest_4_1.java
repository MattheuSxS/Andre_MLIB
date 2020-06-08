package Assessment_Andre;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.col;

public class Quest_4_1 {

        public static SparkSession spark;
        public static JavaSparkContext sc;

        public static void main(String[] args) throws AnalysisException {

            spark = SparkSession.builder().appName("Quest 1").master("local[*]").getOrCreate();

            sc = new JavaSparkContext(spark.sparkContext());
            sc.setLogLevel("WARN");

            Dataset<Row> FileBase = spark.read()
                    .option("header", true)
                    .option("inferSchema", true)
                    .csv("hdfs://localhost:9000/Aula_Andre/mushrooms.csv");

            // Show DataFrake
//            FileBase.show();

            // Index to categorical columns
            StringIndexer C00 = new StringIndexer()
                    .setInputCol("class")
                    .setOutputCol("label");
            StringIndexer C01 = new StringIndexer()
                    .setInputCol("cap-shape")
                    .setOutputCol("cap_shapeIndex");
            StringIndexer C02 = new StringIndexer()
                    .setInputCol("cap-surface")
                    .setOutputCol("cap_surfaceIndex");
            StringIndexer C03 = new StringIndexer()
                    .setInputCol("cap-color")
                    .setOutputCol("cap_colorIndex");
            StringIndexer C04 = new StringIndexer()
                    .setInputCol("bruises")
                    .setOutputCol("bruisesIndex");
            StringIndexer C05 = new StringIndexer()
                    .setInputCol("odor")
                    .setOutputCol("odorIndex");
            StringIndexer C06 = new StringIndexer()
                    .setInputCol("gill-attachment")
                    .setOutputCol("gill_attachmentIndex");
            StringIndexer C07 = new StringIndexer()
                    .setInputCol("gill-spacing")
                    .setOutputCol("gill_spacingIndex");
            StringIndexer C08 = new StringIndexer()
                    .setInputCol("gill-size")
                    .setOutputCol("gill_sizeIndex");
            StringIndexer C09 = new StringIndexer()
                    .setInputCol("gill-color")
                    .setOutputCol("gill_colorIndex");
            StringIndexer C10 = new StringIndexer()
                    .setInputCol("stalk-shape")
                    .setOutputCol("stalk_shapeIndex");
            StringIndexer C11 = new StringIndexer()
                    .setInputCol("stalk-root")
                    .setOutputCol("stalk_rootIndex");
            StringIndexer C12 = new StringIndexer()
                    .setInputCol("stalk-surface-above-ring")
                    .setOutputCol("stalk_surface_above_ringIndex");
            StringIndexer C13 = new StringIndexer()
                    .setInputCol("stalk-surface-below-ring")
                    .setOutputCol("stalk_surface_below_ringIndex");
            StringIndexer C14 = new StringIndexer()
                    .setInputCol("stalk-color-above-ring")
                    .setOutputCol("stalk_color_above_ringIndex");
            StringIndexer C15 = new StringIndexer()
                    .setInputCol("stalk-color-below-ring")
                    .setOutputCol("stalk_color_below_ringIndex");
            StringIndexer C16 = new StringIndexer()
                    .setInputCol("veil-type")
                    .setOutputCol("veil_typeIndex");
            StringIndexer C17 = new StringIndexer()
                    .setInputCol("veil-color")
                    .setOutputCol("veil_colorIndex");
            StringIndexer C18 = new StringIndexer()
                    .setInputCol("ring-number")
                    .setOutputCol("ring_numberIndex");
            StringIndexer C19 = new StringIndexer()
                    .setInputCol("ring-type")
                    .setOutputCol("ring_typeIndex");
            StringIndexer C20 = new StringIndexer()
                    .setInputCol("spore-print-color")
                    .setOutputCol("spore_print_colorIndex");
            StringIndexer C21 = new StringIndexer()
                    .setInputCol("population")
                    .setOutputCol("populationIndex");
            StringIndexer C22 = new StringIndexer()
                    .setInputCol("habitat")
                    .setOutputCol("habitatIndex");

            // Vectorizing the indexes
            OneHotEncoderEstimator encoder = new OneHotEncoderEstimator()
                    .setInputCols(new String[] {"cap_shapeIndex", "cap_surfaceIndex", "cap_colorIndex", "bruisesIndex"
                            , "odorIndex", "gill_attachmentIndex", "gill_spacingIndex", "gill_sizeIndex", "gill_colorIndex"
                            , "stalk_shapeIndex", "stalk_rootIndex", "stalk_surface_above_ringIndex"
                            , "stalk_surface_below_ringIndex", "stalk_surface_above_ringIndex", "stalk_color_below_ringIndex"
                            , "veil_colorIndex", "ring_numberIndex", "ring_typeIndex", "spore_print_colorIndex"
                            , "populationIndex", "habitatIndex"})
                    .setOutputCols(new String[] {"cap_shapeVec", "cap_surfaceVec", "cap_colorVec", "bruisesVec"
                            , "odorVec", "gill_attachmentVec", "gill_spacingVec", "gill_sizeVec", "gill_colorVec"
                            , "stalk_shapeVec", "stalk_rootVec", "stalk_surface_above_ringVec","stalk_surface_below_ringVec"
                            , "stalk-color-above-ringVec", "stalk_color_below_ringVec"
                            , "veil_colorVec", "ring_numberVec", "ring_typeVec", "spore_print_colorVec"
                            , "populationVec", "habitatVec"});

            // Vectorizing for the model
            VectorAssembler assembler = (new VectorAssembler()
                    .setInputCols(new String[]{"cap_shapeVec", "cap_surfaceVec", "cap_colorVec", "bruisesVec"
                            , "odorVec", "gill_attachmentVec", "gill_spacingVec", "gill_sizeVec", "gill_colorVec"
                            , "stalk_shapeVec", "stalk_rootVec", "stalk_surface_above_ringVec","stalk_surface_below_ringVec"
                            , "stalk-color-above-ringVec", "stalk_color_below_ringVec"
                            , "veil_typeIndex", "veil_colorVec", "ring_numberVec", "ring_typeVec", "spore_print_colorVec"
                            , "populationVec", "habitatVec"})
                    .setOutputCol("features"));

            // Dividing the base
            Dataset<Row>[] split = FileBase.randomSplit(new double[] {0.7, 0.3});
            Dataset<Row> dfTrain = split[0];
            Dataset<Row> dfTest  = split[1];

            // Model used
            LogisticRegression lr = new LogisticRegression()
                    .setLabelCol("label")
                    .setFeaturesCol("features");

            // Indexing the columns in the pipeline
            Pipeline pipeline = new Pipeline()
                    .setStages(new PipelineStage[]{C00, C01, C02, C03, C04, C05
                            ,C06, C07, C08, C09, C10
                            ,C11, C12, C13, C14, C15
                            ,C16, C17, C18, C19, C20
                            ,C21, C22
                            ,encoder
                            ,assembler
                            ,lr});

            // Using pipeline
            PipelineModel lr_model = pipeline.fit(dfTrain);

            // Using the model on the test base
            Dataset<Row> predictions = lr_model.transform(dfTest);
            predictions.groupBy(col("label")).count().show();
            predictions.select("label", "prediction", "probability", "rawPrediction", "features").show();

            // Correct answers;
            predictions.groupBy(col("label"),col("prediction")).count().show();

            // Accuracy computation
            MulticlassClassificationEvaluator evaluator1 = new MulticlassClassificationEvaluator()
                    .setLabelCol("label")
                    .setPredictionCol("prediction")
                    .setMetricName("accuracy");
            double accuracy = evaluator1.evaluate(predictions);
            System.out.println("Accuracy of logistic regression in the training base = " + Math.round( accuracy * 100) + "%" );

        }
}
