# %%
from utils import *

import pandas as pd
import numpy as np
from pycaret.classification import *
from sklearn.impute import KNNImputer
from sklearn.preprocessing import *
from feature_engine.imputation import RandomSampleImputer

# %%
# Tengo que crear una función que recoja todas las modificaciónes que hemos hecho hasta ahora para implementarla en el df objetivo
# Las ejecuto EXACTAMENTE en el orden en el que las hice

def tratamiento(df):
    # V3
    df2 = df.copy() 
    
    from_zero_to_nan = ['funder','gps_height','installer','longitude','construction_year']
    for fn in from_zero_to_nan:
        a = df2[fn].isnull().sum()
        df2[fn].replace(('0', 0), (np.nan), inplace = True)
        b = df2[fn].isnull().sum()
        del a
        del b

    #lista_col = df2.columns.tolist() # para que lo hice?????? ####################

    df2.loc[df2['gps_height'] < 0, 'gps_height'] = np.nan # Para alturas negativas

    # parece que hay 1812 registros con -2.000000e-08. Los ponemos como missings
    df2['latitude'].replace(-2.000000e-08, np.nan, inplace = True)  

    df3 = df2.copy()    # Estoy copiando y pegando las lineas en las que modificaba los df en los otros jupiter
    del df2             # para que no ocupe memoria

    # V3.1
    del df3['recorded_by']
    del df3['extraction_type']
    del df3['extraction_type_group']
    del df3['scheme_name']
    del df3['management']

    df3['management_group'].replace(['other', 'unknown'], 'other/unknown', inplace = True)
    df3['payment_type'].replace(['other', 'unknown'], 'other/unknown', inplace = True)

    del df3['payment']
    del df3['water_quality']
    del df3['quantity_group']

    df3['quantity'].replace('unknown', np.nan, inplace = True)

    del df3['source']
    df3['source_class']. replace('unknown', np.nan, inplace = True)

    del df3['waterpoint_type_group']
    
    # df3 real contiene status_group

    # V4.1
    df4 = df3.copy()
    del df3

    df4['date_recorded'] = pd.to_datetime(df4['date_recorded'])
    df4['year_recorded'] = df4['date_recorded'].dt.year
    del df4['date_recorded']

    df4_distr_aleatoria = df4.copy()
    df4_distr_aleatoria['funder'].replace('unknown', np.nan, inplace=True)
    imputer_aleatorio = fe_imp.RandomSampleImputer()
    df4_distr_aleatoria['funder'] = imputer_aleatorio.fit_transform(df4_distr_aleatoria[['funder']])

    df4_distr_aleatoria['installer'].replace('unknown', np.nan, inplace=True)
    imputer_aleatorio = fe_imp.RandomSampleImputer()
    df4_distr_aleatoria['installer'] = imputer_aleatorio.fit_transform(df4_distr_aleatoria[['installer']])

    df4_Distinct = df4_distr_aleatoria.copy()
    df4_Distinct['funder'].replace('unknown', np.nan, inplace=True)
    missing_count = df4_Distinct['funder'].isnull().sum()
    random_strings = [''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'), size=4)) for _ in range(missing_count)]
    df4_Distinct.loc[df4_Distinct['funder'].isnull(), 'funder'] = random_strings

    df4_Distinct['installer'].replace('unknown', np.nan, inplace=True)
    missing_count = df4_Distinct['installer'].isnull().sum()
    random_strings = [''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'), size=4)) for _ in range(missing_count)]
    df4_Distinct.loc[df4_Distinct['installer'].isnull(), 'installer'] = random_strings

    df5 = df4_Distinct.copy()

    # V4.2
    df5_geo = df5.copy()
    df5_geo = df5.loc[:, ['id', 'longitude', 'latitude', 'wpt_name', 
        'basin', 'subvillage', 'region', 'region_code', 'district_code', 'lga', 'ward']]
    
    imputer_knn = KNNImputer(n_neighbors=3)
    columns_to_impute = ['longitude', 'latitude']
    df5_geo_subset = df5_geo[columns_to_impute]
    imputer_knn.fit(df5_geo_subset)
    df5_geo_subset_imputed = pd.DataFrame(imputer_knn.transform(df5_geo_subset), columns=columns_to_impute)
    df5_geo_imputed = df5_geo.copy()
    df5_geo_imputed[columns_to_impute] = df5_geo_subset_imputed
    df5_geo_imputed

    df5_altura = df5_geo_imputed.loc[:, ['id', 'longitude', 'latitude']].copy()
    df5_altura = pd.merge(df5_altura, df5[['id', 'gps_height']], on='id', how='inner')

    imputer_knn = KNNImputer(n_neighbors=2) 
    imputer_knn.fit(df5_altura)
    df5_altura_imputed = pd.DataFrame(imputer_knn.transform(df5_altura), columns=df5_altura.columns)


    column_order = df5.columns.tolist()
    df5 = df5.drop(['gps_height', 'longitude', 'latitude'], axis=1)
    df5.head()
    df5 = pd.merge(df5, df5_altura_imputed, on='id', how='inner')
    df5 = df5[column_order]

    df5_sin_missing_location = df5.copy()
    missing_count = df5_sin_missing_location['subvillage'].isnull().sum()
    random_strings = [''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'), size=4)) for _ in range(missing_count)]
    df5_sin_missing_location.loc[df5_sin_missing_location['subvillage'].isnull(), 'subvillage'] = random_strings

    df5_sin_missing_location['wpt_name'].replace('none', np.nan, inplace=True)
    missing_count = df5_sin_missing_location['wpt_name'].isnull().sum()
    random_strings = [''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'), size=4)) for _ in range(missing_count)]
    df5_sin_missing_location.loc[df5_sin_missing_location['wpt_name'].isnull(), 'wpt_name'] = random_strings

    # V4.3
    df5 = df5_sin_missing_location.copy()
    del df5_sin_missing_location

    df5_public_meeting = df5.copy()
    imputer_aleatorio = fe_imp.RandomSampleImputer()
    df5_public_meeting['public_meeting'] = imputer_aleatorio.fit_transform(df5_public_meeting[['public_meeting']])
    del df5

    # V4.4
    df5 = df5_public_meeting.copy()
    del df5_public_meeting
    df5['scheme_management'] = df5['scheme_management'].replace(np.nan, 'Other')

    df5_permit = df5.copy()
    imputer_aleatorio = fe_imp.RandomSampleImputer()
    df5_permit['permit'] = imputer_aleatorio.fit_transform(df5_permit[['permit']])

    df6 = df5_permit.copy()
    imputer_aleatorio = fe_imp.RandomSampleImputer()
    df6['source_class'] = imputer_aleatorio.fit_transform(df6[['source_class']])

    imputer_aleatorio = fe_imp.RandomSampleImputer()
    df6['quantity'] = imputer_aleatorio.fit_transform(df6[['quantity']]) 

    del df5, df5_permit

    # V5.2
    # df6 = df6
    df6_imputed_data = df6.copy()

    imputer = RandomSampleImputer()
    df6_imputed_data['construction_year'] = imputer.fit_transform(df6[['construction_year']])
    missing_values_summary(df6_imputed_data)
  
    return(df6_imputed_data)


# Guardo el base de entrenamiento (training set)
# df6.to_pickle("./pickles_temp/V5_base.pkl")
