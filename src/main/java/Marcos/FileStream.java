package Marcos;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.*;
import org.apache.spark.sql.streaming.StreamingQuery;
import org.apache.spark.sql.streaming.StreamingQueryException;
import org.apache.spark.sql.streaming.Trigger;
import org.apache.spark.sql.types.StructType;

public class FileStream {

    public static SparkSession spark;
    public static JavaSparkContext sc;

    public static void main(String[] args){

        spark = SparkSession.builder().appName("FileStreaming").master("local[*]").getOrCreate();

        sc = new JavaSparkContext(spark.sparkContext());
        sc.setLogLevel("WARN");

        runStreamingFromDir(spark);

        spark.stop();

    }

    private static void runStreamingFromDir(SparkSession spark){

        // Definindo meu esquema de dados com StructType
        StructType tweetSchema = new StructType()
                .add("msg","string")
                .add("date","string")
                .add("source","string")
                .add("isRetweeted","string")
                .add("user_id","long")
                .add("followers","long");


        Dataset<TweetsBean> inLines = spark
                .readStream()
                .schema(tweetSchema)
                .json("hdfs://localhost:9000/Tweets/*.json")
                .as(Encoders.bean(TweetsBean.class));

        inLines.createOrReplaceTempView("BaseTweets");

//        inLines.printSchema();

        String sqlQuerrySource = "SELECT COUNT(followers) AS TotalLinhas FROM BaseTweets";

        Dataset<Row> sqlDFSource = spark.sql(sqlQuerrySource);

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