#calculo
import numpy as np
import pandas as pd

#grafico
import matplotlib.pyplot as plt

def error_print(text):
    print("*"*50)
    print("ERROR:", text)
    print("*"*50)

class Analizer:
    def __init__(self, ruta='', sep=',', encoding='ISO-8859-1'):
        func_read = pd.read_csv if ruta.lower().endswith('.csv') else None
        if not func_read:
            error_print('archivo .csv no detectado')
            self.data = None
            return
            
        try:
            self.data = pd.read_csv(ruta, encoding=encoding, sep=sep)
        except:
            error_print('verifique ubicacion de archvio')
            self.data = None
            return
        self.test_data = None
        self.col_target = ''
        self.target = None
        self.columns = list(self.data.columns)
        self.columns_set = set(self.columns)

    def null_data(func):
        def proccess(self, *args, **kwargs):
            if self.data is None:
                error_print('data no extraida')
                return
            return func(self, *args, **kwargs)
        return proccess
    
    @null_data
    def head(self, num=5):
        return self.data.head(num)
        
    @null_data
    def set_test_data(self, datus):
        self.test_data = datus.data if isinstance(datus, self.__class__) else (
            datus if isinstance(datus, pd) else None
        )
        return self.test_data.head() if self.test_data is not None else 'Test no asignado'
    
    @null_data
    def set_target(self, col_name):
        if col_name not in self.columns:
            error_print('Columna no identificada')
            return
        
        self.col_target = col_name
        self.target = self.data[col_name]
        self.data = self.data.drop([col_name], axis=1)
        return self.head()
        
    @property
    def columns_null(self):
        return [
            col for col in self.columns_set if self.data[col].isnull().any()
        ]   
    
    def cols_for_types(self, cols=[]):
        if not self.validate_list(cols):
            error_print("Campo requerido como lista invalido")
            return
        
        if not set(cols).issubset(self.columns_set):
            error_print("Columnas invalidas detectadas")
            return 
        
        print('validado')
        return {
            str(k).lower(): [col for col in list(v) if col in cols]
            for k, v in 
            self.data.columns.to_series().groupby(self.data.dtypes).groups.items()
        }

    @property
    def types(self):
        return [str(tipo) for tipo in set(self.data.dtypes)]
    
    
    @null_data
    def counts(self):
        for col in self.data.columns.values:
            print("-"*40)
            print(self.data[col].value_counts(normalize = True, dropna = False))

    
    @null_data
    def null_verificator(self):        
        if self.data.isnull().any().any():
            view_info = pd.DataFrame(
                pd.concat(
                    [self.data.isnull().any(), 
                     self.data.isnull().sum(),
                     self.data.dtypes], 
                    axis=1)
            )
            view_info.columns = ['Nulos', 'Cantidad', 'Tipo Col']
            return view_info
        else:
            return "DATA LIMPIA DE NULOS"
        
    @null_data
    def describe(self):
        return self.data.describe(include='all')
    
    @null_data
    def percentil_verificator(
        self, cols=[], list_percentil=[0,1,3,5,10,20,30,50,60,70,80,90,95,97,99,100]
    ):
        if not isinstance(cols, list):
            print("'cols' debe contener una lista ")
            
        datus = self.data.copy()
        for col in cols:
            plt.plot(
                list_percentil,
                np.nanpercentile(datus[col], list_percentil),
                label=col
            )
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        df_datus = pd.concat(
            [ 
                pd.Series(np.nanpercentile(datus[col], list_percentil),
                          index=list_percentil) 
                for col in cols
            ],
            axis=1
        )
        df_datus.columns = cols
        return df_datus.T
    
    @null_data
    def del_cols(self, cols=[], replicate_in_test=False):
        if not self.validate_list(cols):
            error_print("Campo requerido como lista invalido")
            return
        
        if not set(cols).issubset(self.columns_set):
            error_print("Columnas invalidas detectadas")
            return
        
        if replicate_in_test and self.test_data is None:
            error_print("Data test no especificada")
            return
        
        self.data = self.data.drop(cols, axis=1)
        self.columns = list(self.data.columns)
        print("columns data : ", self.columns)
        
        if replicate_in_test:
            self.test_data = self.test_data.drop(cols, axis=1)
            print("columns test : ", self.test_data.columns)
    
    @null_data
    def cut_col_percentil(self, cols=[], percentile_bigger=95, percentile_smaller=0, 
                          val_bigger=[], val_smaller=[], replicate_in_test=False):

        if not self.validate_list(cols, val_bigger, val_smaller):
            error_print("Campo requerido como lista invalido")
            return
        
        if not self.validate_percentil(percentile_bigger, percentile_smaller):
            error_print("Campo requerido como entero invalido")
            return
        
        if not set(cols).issubset(self.columns_set):
            error_print("Columnas invalidas detectadas")
            return 
        
        if val_bigger and len(val_bigger) != len(cols):
            error_print("la cantidad de valores debe corresponder a cada columna")
            return

        if val_smaller and len(val_smaller) != len(cols):
            error_print("la cantidad de valores debe corresponder a cada columna")
            return 
        
        if replicate_in_test and self.test_data is None:
            error_print("Data test no especificada")
            return            

        for i, col in enumerate(cols):
            copy_col = self.data[col].copy()
            
            percent_cut = val_bigger[i] if val_bigger else copy_col.quantile(percentile_bigger/100)
            self.data.loc[self.data[col] > percent_cut, col] = percent_cut
            if replicate_in_test:
                self.test_data.loc[self.test_data[col] > percent_cut, col] = percent_cut
                
            if val_smaller or percentile_smaller:
                percent_cut = val_smaller[i] if val_smaller else copy_col.quantile(percentile_smaller/100)
                self.data.loc[self.data[col] < percent_cut, col] = percent_cut
                if replicate_in_test:
                    self.test_data.loc[self.test_data[col] < percent_cut, col] = percent_cut
        
        del copy_col
        if not replicate_in_test:
            return self.data[cols].describe(include='all')
        else:
            return pd.DataFrame(
                pd.concat(
                    [self.data[cols].describe(include='all'), 
                     self.test_data[cols].describe(include='all')], 
                    axis=1)
            )
    
    def validate_percentil(self, *args):
        return all([isinstance(_, int) and _ in range(0,100) for _ in args])
        
    
    def validate_list(self, *args):
        return all([isinstance(_, list) for _ in args])

    
    @null_data
    def add_col_dates(self, col, format_match="%d-%b-%y", month=True, day=True, month_day=True,
                      weekday=True, replace_str=False, format_str_replace='%Y/%m/%d', replicate_in_test=False):
        """
        por optimizar en casos separados para data y test_data
        """
        
        self.data['date'] = pd.to_datetime(self.data[col], format=format_match)
        self.data = self.data.drop([col], axis=1)
        if replicate_in_test:
            self.test_data['date'] = pd.to_datetime(self.test_data[col], format=format_match)
            self.test_data = self.test_data.drop([col], axis=1)
        
        if month:
            self.data['month'] = pd.to_numeric(self.data['date'].dt.strftime('%m'), errors='coerce')
            self.data['month'].fillna(-99)
            if replicate_in_test:
                self.test_data['month'] = pd.to_numeric(self.test_data['date'].dt.strftime('%m'), errors='coerce')
                self.test_data['month'].fillna(-99)             
        if day:
            self.data['day'] = pd.to_numeric(self.data['date'].dt.strftime('%d'), errors='coerce')
            self.data['day'].fillna(-99)
            if replicate_in_test:
                self.test_data['day'] = pd.to_numeric(self.test_data['date'].dt.strftime('%d'), errors='coerce')
                self.test_data['day'].fillna(-99)
        if month_day:
            self.data['month_day'] = pd.to_numeric(self.data['date'].dt.strftime('%m%d'), errors='coerce')
            self.data['month_day'].fillna(-99)
            if replicate_in_test:
                self.test_data['month_day'] = pd.to_numeric(self.test_data['date'].dt.strftime('%m%d'), errors='coerce')
                self.test_data['month_day'].fillna(-99)               
        if weekday:
            self.data['weekday'] = pd.to_numeric(self.data['date'].dt.strftime('%w'), errors='coerce')
            self.data['weekday'].fillna(-99)
            if replicate_in_test:
                self.test_data['weekday'] = pd.to_numeric(self.test_data['date'].dt.strftime('%w'), errors='coerce')
                self.test_data['weekday'].fillna(-99)     
        if replace_str:
            self.data['date'] = self.data['date'].dt.strftime(format_str_replace)
            if replicate_in_test:
                self.test_data['date'] = self.test_data['date'].dt.strftime(format_str_replace)
        
        print("columns data : ", self.columns)
        print("columns test : ", self.test_data.columns)
        return(self.data.head(10))
