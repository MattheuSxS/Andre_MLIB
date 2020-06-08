package Assessment_Andre;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.streaming.StreamingQuery;
import org.apache.spark.sql.streaming.StreamingQueryException;
import org.apache.spark.sql.streaming.Trigger;
import org.apache.spark.sql.types.StructType;

public class Quest_2_Streaming {
    public static SparkSession spark;
    public static JavaSparkContext sc;

    public static void main(String[] args){

        spark = SparkSession.builder().appName("FileStreaming").master("local[*]").getOrCreate();

        sc = new JavaSparkContext(spark.sparkContext());
        sc.setLogLevel("WARN");

        StreamingMushrooms(spark);

        spark.stop();

    }

    private static void StreamingMushrooms(SparkSession spark){

        StructType enemShema = new StructType()
                .add("answer","string")
                .add("cap_shape","string")
                .add("cap_surface","string")
                .add("cap_color","string")
                .add("bruises","string")
                .add("odor","string")
                .add("gill_attachment","string")
                .add("gill_spacing","string")
                .add("gill_size","string")
                .add("gill_color","string")
                .add("stalk_shape","string")
                .add("stalk_root","string")
                .add("stalk_surface_above_ring","string")
                .add("stalk_surface_below_ring","string")
                .add("stalk_color_above_ring","string")
                .add("stalk_color_below_ring","string")
                .add("veil_type","string")
                .add("veil_color","string")
                .add("ring_number","string")
                .add("ring_type","string")
                .add("spore_print_color","string")
                .add("population","string")
                .add("habitat", "string");


        Dataset<Quest_2_JavaBean> inLines = spark
                .readStream()
                .format("parquet")
                .schema(enemShema)
                .parquet("hdfs://localhost:9000/Streaming/*.parquet")
                .as(Encoders.bean(Quest_2_JavaBean.class));

        inLines.createOrReplaceTempView("mushrooms");

        Dataset<Row> sqlDFSource = spark.sql("SELECT ROUND(100 * SUM(CASE WHEN answer = 'p' THEN 1 ELSE 0 END) / COUNT(answer), 2) as Venenoso," +
                "SUM(CASE WHEN answer = 'p' THEN 1 ELSE 0 END) as Num_Venenoso, ROUND(100 * SUM(CASE WHEN answer = 'e' THEN 1 ELSE 0 END) / COUNT(answer), 2) as Comestivel" +
                ", SUM(CASE WHEN answer = 'e' THEN 1 ELSE 0 END) as Num_Comestivel, COUNT(answer) as Total FROM mushrooms");

        StreamingQuery querySourceCounts = sqlDFSource.writeStream()
                .outputMode("complete")
                .format("console")
                .trigger(Trigger.ProcessingTime("3 seconds"))
                .start();

        try{
            querySourceCounts.awaitTermination();
        } catch (StreamingQueryException e) {
            e.printStackTrace();
        }

    }
}
