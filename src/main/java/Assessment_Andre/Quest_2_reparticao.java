package Assessment_Andre;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.col;

public class Quest_2_reparticao {

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

        // Colunas modificadas, dado que, tive alguns problemas no JavaBean com os nomes das variáveis.
        // Já repartir o DataFrame em 8 partes iguais.
        Dataset<Row> df = (FileBase
                .select(col("class").as("answer")
                        ,col("cap-shape").as("cap_shape")
                        ,col("cap-surface").as("cap_surface")
                        ,col("cap-color").as("cap_color")
                        ,col("bruises")
                        ,col("odor")
                        ,col("gill-attachment").as("gill_attachment")
                        ,col("gill-spacing").as("gill_spacing")
                        ,col("gill-size").as("gill_size")
                        ,col("gill-color").as("gill_color")
                        ,col("stalk-shape").as("stalk_shape")
                        ,col("stalk-root").as("stalk_root")
                        ,col("stalk-surface-above-ring").as("stalk_surface_above_ring")
                        ,col("stalk-surface-below-ring").as("stalk_surface_below_ring")
                        ,col("stalk-color-above-ring").as("stalk_color_above_ring")
                        ,col("stalk-color-below-ring").as("stalk_color_below_ring")
                        ,col("veil-type").as("veil_type")
                        ,col("veil-color").as("veil_color")
                        ,col("ring-number").as("ring_number")
                        ,col("ring-type").as("ring_type")
                        ,col("spore-print-color").as("spore_print_color")
                        ,col("population")
                        ,col("habitat"))).repartition(8);

        df.show();

        // Criação dos 8 arquivos.
//        df.write().format("parquet").parquet("/home/matt/AulaAndre/DataSetmushrooms");
//
//        System.out.println("Repartição Concluída!");

    }
}
