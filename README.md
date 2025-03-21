# TFM del Máster en Ciencias de Datos y Big Data

[Universidad Internacional de Valencia](https://www.viu.es)

## Generación de una tubería distribuida para la extracción de características en imágenes médicas patológicas

<!-- ![Tests](https://img.shields.io/badge/tests-100%25%20passing-brightgreen) -->
![Version](https://img.shields.io/badge/version-1.0.0-blue)
[![Tests](https://github.com/sanchezis/MU-Big-Data-and-Data-Science-Image-Pathology-Pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/sanchezis/MU-Big-Data-and-Data-Science-Image-Pathology-Pipeline/actions/workflows/tests.yml)
![GitHub Stars](https://img.shields.io/github/stars/sanchezis/MU-Big-Data-and-Data-Science-Image-Pathology-Pipeline?logo=github&color=yellow)

<!--
<img src="https://oncampus.universidadviu.com/sites/viu/files/logo_crespon_0.png" width="100" />
-->

[Israel Llorens](https://www.linkedin.com/in/israel-llorens/)

## Tabla de contenido

* [Información General](#información-general)
* [Tecnologías](#tecnologías)
* [Configuración](#configuración)
* [Ejecución](#ejecución)

## Información General

El tema de desarrollo del presente trabajo, se enfoca en una primera instancia, la de analizar el estado del arte de aplicativos similares. Para más tarde, realizar una solución homogénea para el uso de profesionales de la salud, instituciones y empresas farmacéuticas. El objetivo de este proyecto es el realizar una implementación de una tubería donde cada paso realice una operación de extracción de información y minería de datos.

Para realizar un proceso en el que se llegue a una solución donde se pueda utilizar un conjunto de datos resultado. Este mismo desarrollo brindará un conocimiento adicional de la búsqueda de patrones en imágenes médicas y realizará y mantendrá una base de datos con capacidades de agregar iterativamente mayor información para la búsqueda de conocimiento.

## Tecnologías

### Pre requisitos

Se deben instalar o tener acceso a las siguientes tecnologías instaladas en ya sea en local como en sistema distribuido.

* Spark cluster versión ^3.4.4
* Python 3.11
* Java JDK versión 11.0.24
* Scala versión 2.13
* Librerías: pyspark (3.4.4), pylint(3.1.0), numpy (^1.20.2), pandas (^2.0.0)

## Configuración

### Instalar dependencias

Se puede instalar spark mediante la instalación de las dependencias de código o en forma local siguiendo el siguiente `script` en la máquina donde se desea tener el entorno.

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

Se pueden instalar los requerimientos de Python necesarios usando `poetry`.

```bash
pip install poetry
poetry install
```

los comandos instalarán todas las dependencias utilizadas en este proyecto.

### Ejecución de pruebas de código

 > Todos los siguientes fragmentos se deben ejecutar satisfactoriamente.

El proyecto posee un marco de trabajo o `Framework` denominado `digital_pathology` el cual tiene asociado un directorio de pruebas unitarias y de integración, que al ser ejecutadas, mantienen un correcto funcionamiento de las funcionalidades que se ejecutan en cada etapa de la extracción de características.

* Ejecución de pruebas unitarias

```bash
poetry run pytest tests/unit
```

* Ejecución de pruebas de integración

```bash
poetry run pytest tests/integration
```

* Ejecución de chequeo de estilos y buenas prácticas de código

```bash
poetry run mypy --ignore-missing-imports --disallow-untyped-calls --disallow-untyped-defs --disallow-incomplete-defs digital_pathology tests

poetry run pylint data_transformations tests
```

## Ejecución

El proyecto sigue la siguiente estructura de directorio.

```bash
/
├─ /digital_pathology # Contains the main python library
│ # with the code to the transformations
│
├─ /jobs # Contains the entry points to the jobs
│ # performs argument parsing, and are
│ # passed to `spark-submit`
│
├─ /notebooks # Contains the notebooks for databricks
│ # implementation and pipeline execution in cluster/cloud
│ # environment
│
├─ /tests
│ ├─ /units # contains basic unit tests for the code
│ └─ /integration # contains integrations tests for the jobs
│ # and the setup
│
├─ .gitignore
├─ .pylintrc # configuration for pylint
├─ LICENCE
├─ poetry.lock
├─ pyproject.toml
└─ README.md # The current file
```

Las pruebas en el repositorio se efectúan de forma autónoma y aparecen como realizadas mediante el `badge` correspondiente. Para la ejecución de cada proceso se necesita ejecutar el script `run.sh` o mediante la siguiente instrucción.

> ⚠️ Se puede ejecutar de manera local!

```bash
poetry build && poetry run spark-submit \
    --master local \
    --py-files dist/digital_pathology-*.whl \
    jobs/<JOB_STEP>.py \
    <INPUT_FILE_PATH> \
    <OUTPUT_PATH>
```

* JOB_STEP: paso de la tubería (pipeline) la cual efectúa un proceso específico de extracción de características, procesado de imágenes y minería de datos, los posibles valores pueden ser `download`,`ingest`,`pre_process` or `extract`.
* INPUT_FILE_PATH: De ser necesario, se puede añadir la fuente de datos de imágenes patológicas de entrada.
* OUTPUT_PATH: Directorio de salida donde se almacenan todos los pasos ejecutados.

## License

<!-- [MIT](https://choosealicense.com/licenses/mit/) -->
[EUPL](https://raw.githubusercontent.com/sanchezis/MU-Big-Data-and-Data-Science-Image-Pathology-Pipeline/refs/heads/main/LICENSE)

<p align="center">
<img src="https://www.universidadviu.com/sites/universidadviu.com/themes/custom/universidadviu_com/logo.webp" width="180" />
</p>
