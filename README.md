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


