# Sistema multimodal para la detección e identificación de especies de hongos mediante visión por computador y modelos de lenguaje

##  Descripción general
En este proyecto se pretende crear una pipeline que consite en un modelo de clasificación de imagenes para reconocer diferentes especies de setas y un LLM (large language model) que dará información al usuario, esto es pensado para la gente que guste de recoger setas por el campo.

## Arquitectura del sistema
El esquema es el siguiente:

Imagen --> Modelo de reconocimiento de setas(en TPU) --> LLM --> Interfaz

-  Imagen de una camara conectada a la dev board
-  Modelo entrenado con transferlearning
-  LLM modelo de lenguaje empleado para dar razones para no consumir la seta en caso de ser venenosa o recetas en caso contrario
-  Interfaz para hacerlo amigable al usuario.

##  Instrucciones para ejecutar
-  Descargar los ficheros sever.py, main.py y labels.txt.
-  Subir los ficheros sever.py,labels.csv y el modelo de tlflite. Para ello usa el comando `mdt push <NOMBRE DEL FICHERO>`.
-  Entra accede a la placa utilizando `mdt shell` y luego ejecuta el scrit de python con `python3 server.py`.
-  Ejecuta el main.py en el ordenador principal, ya sea desde terminal o desde un editor de código cualquiera.
