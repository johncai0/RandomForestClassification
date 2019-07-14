package org.john.local

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._


/**
  * Created by john on 18-7-2.
  */
object userTarget {
  def main(args: Array[String]) {
    val ss = SparkSession.builder().getOrCreate()
    val base = "hdfs://john:9000/user/RandomForestClassification/*"

    val fieldSchema = StructType(Array(
      StructField("label", DoubleType, true),
      StructField("click", IntegerType, true),
      StructField("sc", IntegerType, true),
      StructField("visits", IntegerType, true),
      StructField("pv", IntegerType, true),
      StructField("tm", IntegerType, true)
    ))
    val data = ss.read.schema(fieldSchema).csv(base)
    val rdf=new RunRDF(data)
    rdf.rdfRun()
  }
}

