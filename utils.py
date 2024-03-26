import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


## Función para gestionar outliers
## Función manual de winsor con clip+quantile 
def winsorize_with_pandas(s, limits):
    """
    s : pd.Series
        Series to winsorize
    limits : tuple of float
        Tuple of the percentages to cut on each side of the array, 
        with respect to the number of unmasked data, as floats between 0. and 1
    """
    return s.clip(lower=s.quantile(limits[0], interpolation='lower'), 
                  upper=s.quantile(1-limits[1], interpolation='higher'))


## Función para gestionar outliers
def gestiona_outliers(col,clas = 'check'):
    
     print(col.name)
     # Condición de asimetría y aplicación de criterio 1 según el caso
     if abs(col.skew()) < 1:
        criterio1 = abs((col-col.mean())/col.std())>3
     else:
        criterio1 = abs((col-col.median())/stats.median_abs_deviation(col))>8 ## Cambio de MAD a stats.median_abs_deviation(col)
     
     # Calcular primer cuartil     
     q1 = col.quantile(0.25)  
     # Calcular tercer cuartil  
     q3 = col.quantile(0.75)
     # Calculo de IQR
     IQR=q3-q1
     # Calcular criterio 2 (general para cualquier asimetría)
     criterio2 = (col<(q1 - 3*IQR))|(col>(q3 + 3*IQR))
     lower = col[criterio1&criterio2&(col<q1)].count()/col.dropna().count()
     upper = col[criterio1&criterio2&(col>q3)].count()/col.dropna().count()
     # Salida según el tipo deseado
     if clas == 'check':
            return(lower*100,upper*100,(lower+upper)*100)
     elif clas == 'winsor':
            return(winsorize_with_pandas(col,(lower,upper)))
     elif clas == 'miss':
            print('\n MissingAntes: ' + str(col.isna().sum()))
            col.loc[criterio1&criterio2] = np.nan
            print('MissingDespues: ' + str(col.isna().sum()) +'\n')
            return(col)

def histogram_boxplot(data, xlabel = None, title = None, font_scale=2, figsize=(9,8), bins = None):
    """ Boxplot and histogram combined
    data: 1-d data array
    xlabel: xlabel 
    title: title
    font_scale: the scale of the font (default 2)
    figsize: size of fig (default (9,8))
    bins: number of bins (default None / auto)

    example use: histogram_boxplot(np.random.rand(100), bins = 20, title="Fancy plot")
    """
    # Definir tamaño letra
    sns.set(font_scale=font_scale)
    # Crear ventana para los subgráficos
    f2, (ax_box2, ax_hist2) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=figsize)
    # Crear boxplot
    sns.boxplot(x=data, ax=ax_box2)
    # Crear histograma
    sns.histplot(x=data, ax=ax_hist2, bins=bins) if bins else sns.histplot(x=data, ax=ax_hist2)
    # Pintar una línea con la media
    ax_hist2.axvline(np.mean(data),color='g',linestyle='-')
    # Pintar una línea con la mediana
    ax_hist2.axvline(np.median(data),color='y',linestyle='--')
    # Asignar título y nombre de eje si tal
    if xlabel: ax_hist2.set(xlabel=xlabel)
    if title: ax_box2.set(title=title, xlabel="")
    # Mostrar gráfico
    plt.show()


# Devuelve 
# - breve descripción
# - gráfico de barras con las 8 más comunes
# - gráfico de pastel con missings, ceros, 8 + comunes y resto
# - Lista de 15-5 más y menos frecuentes con su frecuencia
def descripcion_categorica(df, col, resultados_mostrados=15):
    """
    Función personal con poca ayuda de chatGPT
    """
    if (df[col].dtypes == 'object'):
        print("--------------", col.upper(), "---------------")
        print(df[col].describe())
        print("NaN:   ", df[col].isna().sum())
        print("none:  ", + (df[col] == 'none').sum())
        print("ceros: ", (df[col] == '0').sum())
        print("-------------- Head y Tail ---------------")
        print(df[col].value_counts().head(resultados_mostrados))  #Muestra los resultados
        print(df[col].value_counts().tail(round(resultados_mostrados/3)))

        # Calcular las frecuencias de las 8 categorías principales y el resto
        categorias_principales = df[col].value_counts().nlargest(8).index
        frecuencias = df[col].value_counts()
        frecuencias_resto = frecuencias[~frecuencias.index.isin(categorias_principales)]
        num_ceros = (df[col] == '0').sum()
        num_na = df[col].isna().sum()
        # Calcular las frecuencias de las únicas
        categorias_unicas = df[col].value_counts()[df[col].value_counts() == 1].index
        frecuencias_unicas = sum(df[col].isin(categorias_unicas))
        # Calcular la suma de las frecuencias de las categorías principales
        frecuencia_categorias_principales = frecuencias[categorias_principales].sum()
        # Crear un arreglo con las frecuencias
        frecuencias_totales = np.array([num_ceros, num_na, frecuencia_categorias_principales, frecuencias_unicas, frecuencias_resto.sum()])
        # Crear una lista con las etiquetas
        etiquetas = ['Ceros', 'na/none', '8 Categorías Principales', 'Únicas', 'Resto']
        # Crear el gráfico de quesitos
        plt.figure(figsize=(12, 5))
        # Grafico de barras
        plt.subplot(1, 3, 1)
        sns.countplot(data=df, x=col, order=list(categorias_principales))
        plt.xticks(rotation=60, fontsize=10)  # Rotar las etiquetas del eje x y ajustar tamaño de la fuente
        plt.ylabel('Frecuencia', fontsize=12)
        # Gráfico de quesitos
        plt.subplot(1, 3, 2)
        plt.pie(frecuencias_totales, labels=etiquetas, autopct='%1.1f%%', startangle=0)
        plt.title('Gráfico de Frecuencias')
        # Ajustar el espacio entre subgráficos
        plt.subplots_adjust(wspace=0.2)
        plt.show()


        print("---------------------------------------------")
    else:
        print("error en descripcion_categoricas")
        return(None)
        
def descripcion_numerica(df, col):
    if (df[col].dtype != 'object'): # Analizamos las numéricas, excepto la de tipo datatime
        print("--------------", col.upper(), "---------------")
        print(df[col].describe())
        resultados_outliers = gestiona_outliers(df[col])
        print("Atípicos inferiores:", resultados_outliers[0])
        print("Atípicos superiores:", resultados_outliers[1])
        print("% valores atípicos:", resultados_outliers[2])
        print("missings: ", (df[col].isnull().sum()))
        print("ceros: ", (df[col] == 0).sum())

        plt.hist(df[col], bins=25, color='blue', alpha=0.7)

        plt.title('Histograma de ' + col)
        plt.ylabel('Frecuencia')
        plt.grid(True)

        # Mostrar el histograma
        plt.show()
        print("---------------------------------------------")
    else:
        print("error en descripcion_numerica")
        return(None)


"""
    Esta función calcula el número y el porcentaje de valores nulos en cada columna de un DataFrame.
"""
def missing_values_summary(df):
    """
    Por ChatGPT. Para dar una vision global de missings 
    """
    # Calcula el número de valores nulos en cada columna
    valores_nulos_por_columna = df.isnull().sum()
    
    # Calcula el porcentaje de valores nulos en cada columna
    total_filas = len(df)
    porcentaje_valores_nulos_por_columna = (valores_nulos_por_columna / total_filas) * 100
    
    # Redondea el porcentaje al decimal más cercano
    porcentaje_valores_nulos_por_columna = porcentaje_valores_nulos_por_columna.round(1)
    
    # Crea un DataFrame con el resumen de valores nulos
    resumen_valores_nulos = pd.DataFrame({
        'Valores Nulos': valores_nulos_por_columna,
        '% de Valores Nulos': porcentaje_valores_nulos_por_columna
    })
    
    return resumen_valores_nulos

""" def saca_metricas(y1, y2):
    # saca_metricas(y_test,y_pred)
    print('matriz de confusión')
    print(confusion_matrix(y1, y2))
    print('accuracy')
    print(accuracy_score(y1, y2))
    print('precision')
    print(precision_score(y1, y2))
    print('recall')
    print(recall_score(y1, y2))
    print('f1')
    print(f1_score(y1, y2))
    false_positive_rate, recall, thresholds = roc_curve(y1, y2)
    roc_auc = auc(false_positive_rate, recall)
    print('AUC')
    print(roc_auc)
    plt.plot(false_positive_rate, recall, 'b')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('AUC = %.2f' % roc_auc) """

"""
Devuelve las columnas con 0 o '0' en un df
"""
def zero_searcher(df):
    """Funcion mia"""
    columnas_con_ceros = []
    for columna in df.columns:

        if ((df[columna] == 0).sum() > 0 or (df[columna] == '0').sum() > 0):
            columnas_con_ceros.append(columna)
    return columnas_con_ceros

def numeric_to_categoric(df, col, bins=10): # no testada
    # Crear contenedores y etiquetas
    cut_labels = [f"Categoría_{i}" for i in range(1, bins+1)]
    
    # Aplicar pd.cut() para convertir la columna numérica en categórica
    df[col + '_categoric'] = pd.cut(df[col], bins=bins, labels=cut_labels)