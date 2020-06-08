import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Test {

    public static SparkSession spark;
    public static JavaSparkContext sc;

    public static void main(String[] args) throws AnalysisException {

        spark = SparkSession.builder().appName("Perguntas 4 e 5").master("local[*]").getOrCreate();
        sc = new JavaSparkContext(spark.sparkContext());
        sc.setLogLevel("WARN");

        Dataset<Row> FileBase = spark.read()
                .format("parquet")
                .load("hdfs://localhost:9000/BaseTEnem/*.parquet");

        Dataset<Row> FileBase0 = FileBase.na().drop();

        System.out.println("Total: "+FileBase.count());
        System.out.println("Total: "+FileBase0.count());

        FileBase0.show();

    }
}
