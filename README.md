# MSc - Thesis in Data Science and Big Data

<!-- ![Tests](https://img.shields.io/badge/tests-100%25%20passing-brightgreen) -->
![Version](https://img.shields.io/badge/version-1.0.0-blue)
[![Tests](https://github.com/sanchezis/MU-Big-Data-and-Data-Science-Image-Pathology-Pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/sanchezis/MU-Big-Data-and-Data-Science-Image-Pathology-Pipeline/actions/workflows/tests.yml)
![GitHub Stars](https://img.shields.io/github/stars/sanchezis/MU-Big-Data-and-Data-Science-Image-Pathology-Pipeline?logo=github&color=yellow)

<p align="center">
 <a href="https://creativecommons.org/licenses/by/4.0/">
<img src="https://source.roboflow.com/Fa9p6ViXI3XMKf97qo4vQSiQtNF3/51ovu3VZGLOdxckqk8Rp/original.jpg" width="380" />
<img src="https://live.staticflickr.com/7238/7336389498_7c3ef9d443_b.jpg" width="338" />
</a>
</p>


## Generation of a Distributed Pipeline for Feature Extraction in Pathological Medical Images

<!--
<img src="https://oncampus.universidadviu.com/sites/viu/files/logo_crespon_0.png" width="100" />
-->

[Israel Llorens](https://www.linkedin.com/in/israel-llorens/)

## Table of Contents

* [General Information](#general-information)
* [Technologies](#technologies)
* [Setup](#setup)
* [Execution](#execution)

## General Information

This project aims to implement a pipeline where each step performs an information extraction and data mining operation from sources originating from pathological images.

Likewise, various libraries will be used, and their possibilities will be compared, as if they were executed in a real medical environment.

## Technologies

### Prerequisites

The following technologies must be installed or have access to them locally or in a distributed system.

* Spark cluster version ^3.4.4
* Python 3.11
* Java JDK version 11.0.24
* Scala version 2.13
* Libraries: pyspark (3.4.4), pylint(3.1.0), numpy (^1.20.2), pandas (^2.0.0)
* [OpenSlide](https://openslide.org/)
* [TiaToolbox](https://tia-toolbox.readthedocs.io/), [HistomicsTK](https://github.com/DigitalSlideArchive/HistomicsTK/tree/master), [StarDist](https://stardist.net/)

## Setup

### Install Dependencies

Spark can be installed locally by installing code dependencies or by following the script on the machine where you want to have the environment.

> ⚠️ This Spark environment must be configured to use AWS! You must download the JAR files to connect. If using Spark version 3.4.4, these are the files:
> [`aws-java-sdk-bundle-1.12.262.jar`](https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar)
> [`hadoop-aws-3.3.4.jar`](https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar)
> And they must be saved in the Spark JAR directory.

```bash
# Descarga Spark 3.4.4
wget https://archive.apache.org/dist/spark/spark-3.4.4/spark-3.4.4-bin-hadoop3.tgz

# Descomprime
tar -xvzf spark-3.4.4-bin-hadoop3.tgz
sudo mv spark-3.4.4-bin-hadoop3 /opt/spark

# Configura variables de entorno
echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc
echo 'export PATH=$PATH:$SPARK_HOME/bin' >> ~/.bashrc
source ~/.bashrc
```

You can install the necessary Python requirements using `poetry`.

```bash
pip install poetry
poetry install
```

These commands will install all dependencies used in this project.

### Code Preparation

 > All the following fragments must be executed successfully.

The project has a framework called `digital_pathology` which has an associated directory of unit and integration tests that, when executed, maintain proper functioning of the functionalities executed in each feature extraction stage.


* Unit Testing

```bash
poetry run pytest tests/unit
```

* Integration Testing

```bash
poetry run pytest tests/integration
```

* Code style and best practices check execution

```bash
poetry run mypy --ignore-missing-imports --disallow-untyped-calls --disallow-untyped-defs --disallow-incomplete-defs digital_pathology tests

poetry run pylint data_transformations tests
```

## Execution

The project follows the following directory structure.

```bash
/
├─ /digital_pathology # Contains the main Python library
│ # with the code for the transformations
│
├─ /jobs # Contains the entry points to the jobs
│ # performs argument parsing, and is
│ # passed to `spark-submit`
│
├─ /notebooks # Contains the notebooks for databricks
│ # implementation and pipeline execution in cluster/cloud
│ # environment
│
├─ /tests
│ ├─ /units # contains basic unit tests for the code
│ └─ /integration # contains integration tests for the jobs
│ # and the setup
│
├─ .gitignore
├─ .pylintrc # configuration for pylint
├─ LICENCE
├─ poetry.lock
├─ pyproject.toml
└─ README.md # The current file
```
The tests in the repository are performed autonomously and appear as completed through the corresponding `badge`. To execute each process, run the `run.sh` script or follow the instructions below.

> ⚠️ It can be executed locally!

```bash
poetry build && poetry run spark-submit \
    --master local \
    --py-files dist/digital_pathology-*.whl \
    jobs/<JOB_STEP>.py \
    <INPUT_FILE_PATH> \
    <OUTPUT_PATH>
```

* JOB_STEP: pipeline step which performs a specific feature extraction process, image processing, and data mining, possible values can be `download`, `ingest`, `pre_process`, or `extract`.
* INPUT_FILE_PATH: If necessary, you can add the input pathological image data source.
* OUTPUT_PATH: Output directory where all executed steps are stored.

### Creating a Package for Production Environment Execution

Running the script

```bash
scripts/build.sh
```

Will create a `dist` folder with the following files:

```bash
/
...
│
├─ /dist
│ ├─ /digital_pathology-@VERSION.tar # zipped Python library to be used for executors
│ └─ /digital_pathology-@VERSION-py3-none-any # library wheels for setup and installation
│
...
```

Which can be used in any real or production implementation, whether local, cloud, or cluster.

## License

<!-- [MIT](https://choosealicense.com/licenses/mit/) -->
[EUPL](https://raw.githubusercontent.com/sanchezis/MU-Big-Data-and-Data-Science-Image-Pathology-Pipeline/refs/heads/main/LICENSE)

[Universidad Internacional de Valencia](https://www.viu.es) 
<p align="center">
<img src="https://www.universidadviu.com/sites/universidadviu.com/themes/custom/universidadviu_com/logo.webp" width="180" />
</p>
