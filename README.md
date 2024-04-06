# Machine Learning

Este es el proyecto entregado del módulo de Machine Learning del Master en Big Data por la Universidad Complutense de Madrid. 
Consiste en predecir el fallo de una bbdd real sobre fallos en bombas de agua en Tanzania. Es una competición de Kaggle
Puedes ver el enunciado en enunciado.pdf

## ¿Cómo he estructurado este proyecto en cuanto a carpetas?

- Primero los archivos que empiezan con V* indican el orden en el que he ido ejecutando los Notebooks
- Estos Notebooks se apoyan en la librería utils.py. Esta librería comprende funciones del módulo de Minería de Datos,
    del módulo de Machine Learning, funciones personales, funciones de la documentación oficial de librerias como sklearn,
    y funciones personales customizadas. 
- Para pasar las transformaciones del dataframe de un archivo Júpiter a otro he recurrido a fuardar los df en .pickels
    Estos archivos se pueden encontrar en la carpeta "./pickels_temp, el nombre coincide con el nombre del Notebook
    que lo crea, por ejemplo 'V4.1_imputar_nulos.ipynb' crea "./pickles_temp/V4_1.pkl" 
- Cada vez que hago una prueba, guardo el resultado .csv en la carpeta './resultados', de forma similar al anterior
- La carpeta 'versiones_preliminares' contiene archivos Júpiter preliminares. Carecen de importancia para el proyecto final.
- Recomiendo leer del V0 al V9.0. 

## ¿Qué orden he seguido?

- Antes de nada, hago una prueba imputando todo con un label_encoder básico a todas las variables. Es muy sorprendente que 
    esta forma tan básica obtenga un rate de 0.8020 cuando el record está en 0.8294. No hay mucha diferencia.
- Primero, he tratado de identificar valores nulos en la base de datos y los he etiquetado como tal. Esto ha llevado
    muchisimo trabajo porque había 50 columnas. Como aprendizaje para el futuro, es mejor eliminar las columnas
    que estén correlacionadas de forma obvia para no tratar columnas que luego voy a eliminar. 
- Luego he borrado columnas que presentaban mucha colinealidad, aunque he dejado algunas que presentaban algo. 
- Vuelvo a tirar el modelo imputando por label encoder y para mi sorpresa ha caido a 0.7863
- En la V4 imputo los nulos a mi discrección, con mucho detalle y dedicación. Quizá tanto detalle haya sido un error
    y me hubiera tenido que centrar más en modifical parametros en el modelo, o a crear nuevas columnas. 
- En el V5 y V6 pruebo a tirar el modelo, pero me doy cuenta que debo aplicar todos y cada uno de los pasos al 'test_set_values.csv'
    lo cual es un problema porque no lo había preparado para que tuviera que ser así y es posible que haya cometido algun fallo al
    copiar y pegar todas las instrucciones de todos los Jupiter. También hago frecuency label_encoder y creo categorias one_hot 
    para las categorías con menos variables. La predicción vuelve a bajar hasta el 0.7611
- En el V7 corrijo el desbalanceo por oversampling y vuelvo a comprobar que el score baja de nuevo a 0.7416, lo cual es muy extraño
    porque el accuracy del rf es de 0.8055, no parece que esté tan mal.
- En el V8 intento seguir los videos de clase pero nuestra variable objetivo no es booleana así que lo hago solo parcialmente
    También obtengo un grafico qué mide la importancia de las variables mediante Fscore. 

#Conclusiones y aprendizajes

- Para realizar este proyecto he utilizado la extensión de Jupiter Notebook con VScode. Ha sido un gran error. Lo hice porque
    estoy muy familiarizado con las ayudas de autocompletado y ayudas visuales por colores a variables y metodos. Además,
    es muy útil el navegador entre archivos que incorpora en el lado izquierdo de la pantalla. 
    Por otra parte, he visto que en comparación con Anaconda, es extraordinariamente lento en la ejecución de los chunks,
    llegando a los 4s para empezar a ejecutar cada uno. Además, requiere de 3 minutos para reiniciar el kernel lo cual ha
    sido muy necesario y para rematar ha generado errores inesperados que han provocado la perdida definitiva de partes del
    proyecto. 
- Al empezar un proyecto, lo primero que se debe mirar antes de todo es el formato en el que se solicitan los datos 
    y reflexionar sobre cómo estructurar el código para hacer esas transformaciones. Es decir, no hacer lo que he hecho en libreria
    V6_tratamiento.py, en la que copio uno a uno todos los pasos. Es largo, tedioso y muy propenso a errores. 
- Si el punto anterior te permite probar un resultado facilmente (lo que no ha sido mi caso) es fácil hacer pruebas a cada
    modificación en el conjunto de datos y corregir si es incorrecto. 
- El machine learning es una rama interesante que requiere bastante tiempo de aprendizaje. Pienso que ese tiempo puede 
    optimizarse con el trabajo en equipo. Creo que con algunos compañeros no hubiera cometido tantos errores y el score hubiera
    sido mejor. Voy a intentar acudir con compañeros a alguna hackathlon con el único objetivo de aprender más.
- Cuando el modelo no requiere explicabilidad (como en los concursos) se debe hacer más incapié en el desarrollo del modelo
    más que en el minado de los datos. Pensaba que era opuesto pero me he dado cuenta que los algoritmos más modernos por fuerza 
    bruta dan buenos rendimientos. Es más útil invertir en modificar hiperparámetros que en eliminar-crear columnas nuevas.

Estoy razonablemente contento para ser una primera aproximación al machine learning.


Autor: JGM
Fecha: 04/04/2024
Módulo de Machine Learning 

Visita mi Github para ver más proyectos!:
https://github.com/jogarman
