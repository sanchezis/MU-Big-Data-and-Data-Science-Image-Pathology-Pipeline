import logging
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

spark = SparkSession.getActiveSession()
if not spark:
  # Save your AWS Credentials here
  logging.info('****************************CREATING SPARK CLUSTER*******************************')
  aws_access_key_id = ''
  aws_secret_access_key = ''
  aws_session_token = ''

  spark_conf= {
    'spark.executor.memory' : '8g', 
    'spark.driver.memory'  : '16g',
    'spark.driver.maxResultSize' : '6g',
      
    'spark.dynamicAllocation.enabled': 'false',
    'spark.hadoop.io.compression.codecs': "org.apache.hadoop.io.compress.DefaultCodec,org.apache.hadoop.io.compress.BZip2Codec,org.apache.hadoop.io.compress.GzipCodec",
    'spark.yarn.historyServer.allowTracking': 'true',
    'spark.sql.parquet.fs.optimized.committer.optimization-enabled': 'false',
    'spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version': '2',
    'spark.speculation': 'false',
          
    'spark.hadoop.fs.s3.maxRetries': '20',
    'spark.hadoop.fs.s3.maxConnections': '5000',
    'spark.hadoop.fs.s3a.maxRetries': '20',
    'spark.hadoop.fs.s3a.maxConnections': '5000',
    'spark.port.maxRetries': '100',
      
    "spark.ui.showConsoleProgress": "false",
    
    # AWS settings 
    # "spark.jars.packages": "io.delta:delta-core_2.12:2.3.0,org.apache.hadoop:hadoop-aws:3.3.4,graphframes:graphframes:graphframes-0.8.1-spark3.0-s_2.12,com.databricks:spark-avro_2.11-4.0.0.jar",
    "spark.executor.extraJavaOptions":"-Dcom.amazonaws.services.s3.enableV4=true",
    "spark.driver.extraJavaOptions":"-Dcom.amazonaws.services.s3.enableV4=true",
    'spark.hadoop.fs.s3a.aws.credentials.provider': 'org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider'
  }

  conf = SparkConf() \
      .set("spark.sql.shuffle.partitions", "4")\
      .setMaster("local[6]") \
      .setAppName(f"illorens_TFM")  

  for k, v in spark_conf.items():  # type: ignore
      conf.set(k, v)

  conf = conf\
            .set("spark.executorEnv.OPENSLIDE_PATH", "/") \
            .set("spark.executorEnv.OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES") 


  """
  conf\
    .set("spark.hadoop.fs.s3a.access.key", aws_access_key_id)\
    .set("spark.hadoop.fs.s3a.secret.key", aws_secret_access_key)\
    .set("spark.hadoop.fs.s3a.session.token", aws_session_token)\
    .set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")\
    .set("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")\


  conf.set('spark.hadoop.fs.s3a.access.key','1234567890123456')
  conf.set('spark.hadoop.fs.s3a.secret.key', 'AZERTYUIOPQSDFGHJKLMWXCVBN')
  #                      'fs.s3a.aws.credentials.provider', 
  conf.set('spark.hadoop.fs.s3a.aws.credentials.provider', 'org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider') # SimpleAWSCredentialsProvider, AnonymousAWSCredentialsProvider
  conf.set('spark.jars.packages', 'org.apache.hadoop:hadoop-aws:3.2.1')
  """


  context = SparkContext(conf=conf)
  context.setLogLevel(logLevel="INFO")
  spark = SparkSession(context)
  sc = context

  hadoopConf = spark._jsc.hadoopConfiguration()  # type: ignore  # pylint: disable=protected-access  # noqa: N806
  hadoopConf.set("fs.s3.awsAccessKeyId", aws_access_key_id)
  hadoopConf.set("fs.s3.awsSecretAccessKey", aws_secret_access_key)
  sc._jsc.hadoopConfiguration().set("fs.s3n.impl", "org.apache.hadoop.fs.s3native.NativeS3FileSystem")
  sc._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
  sc._jsc.hadoopConfiguration().set("fs.s3.impl", "org.apache.hadoop.fs.s3.S3FileSystem")
  sc._jsc.hadoopConfiguration().set("fs.s3a.aws.credentials.provider.mapping", "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider")

  spark.conf.set('spark.sql.repl.eagerEval.enabled', True)
