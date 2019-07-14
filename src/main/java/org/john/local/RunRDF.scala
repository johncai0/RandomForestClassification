package org.john.local

import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame

import scala.util.Random

class RunRDF(data:DataFrame) extends Serializable {

  val Array(trainData, testData) = data.randomSplit(Array(0.9, 0.1))
  trainData.cache()
  testData.cache()
  def rdfRun(): Unit={
    val inputCols=trainData.columns.filter(_ != "label") //训练的输入列除了label，其他都是特征向量
    val assembler = new VectorAssembler() //将特征转化为特征向量，列名为featureVector
      .setInputCols(inputCols)
      .setOutputCol("featureVector")
    val classifier = new RandomForestClassifier() //随机决策森林模型定义，包括输入列最大深度等
      .setSeed(Random.nextLong())
      .setLabelCol("label")
      .setFeaturesCol("featureVector")
      .setPredictionCol("prediction")
      .setImpurity("entropy")
      .setMaxDepth(30)
      .setMaxBins(300)

    val pipeline = new Pipeline().setStages(Array(assembler, classifier)) //用管道将所有的stag连接起来

    val paramGrid = new ParamGridBuilder() //定义动态参数
      .addGrid(classifier.minInfoGain, Seq(0.0, 0.05))
      .addGrid(classifier.numTrees, Seq(10, 20))
      .build()

    val multiclassEval = new MulticlassClassificationEvaluator() //定义评估器
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val validator = new TrainValidationSplit() //TrainValidationSplit组建将管道和动态超参数训练模型及模型评估关联起来
      .setSeed(Random.nextLong())
      .setEstimator(pipeline)
      .setEvaluator(multiclassEval)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.9)

    val validatorModel = validator.fit(trainData) //训练模型

    val bestModel = validatorModel.bestModel //挑出佳模型

    val forestModel = bestModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[RandomForestClassificationModel] //获取随机森林模型

    println(forestModel.extractParamMap) //打印当前最佳超参数
    println(forestModel.getNumTrees) //打印决策森林的决策树数量
    forestModel.featureImportances.toArray.zip(inputCols)
      .sorted.reverse.foreach(println) //打印因子影响结果的重要程度

    val testAccuracy = multiclassEval.evaluate(bestModel.transform(testData)) //使用验证集测试模型的准确率
    println(testAccuracy) //打印模型准确率

    bestModel.transform(testData.drop("label")).select("prediction").show() //开始预测，打印测试集数据的回归结果
  }
}
