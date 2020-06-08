package Assessment_Andre;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Quest_1 {

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


        FileBase.createOrReplaceTempView("mushrooms");

        spark.sql("SELECT ROUND(100 * SUM(CASE WHEN class = 'p' THEN 1 ELSE 0 END) / COUNT(class), 2) as Venenoso," +
                "ROUND(100 * SUM(CASE WHEN class = 'e' THEN 1 ELSE 0 END) / COUNT(class), 2) as Comestivel" +
                ", COUNT(class) as Total FROM mushrooms").show();

    }
}
