"""
Classe Dataset para normalizar a manipulação de DataFrames com coluna para rótulos.

Este módulo implementa uma classe "Dataset", que funciona como um wrapper para pandas DataFrames, simplificando
tarefas comuns em tarefas de machine learning, tais como separação entre atributos/rótulo, normalização de dados,
remoção de atributos e divisão do conjunto de dados.

Author: Thiago Barbosa
Created: 2025

Example:
    >>> from dataset import Dataset
    >>> import pandas as pd
    >>> ds = Dataset(df, label_column=-1)
    >>> iris_dataset = Dataset.from_file( filepath = "iris.data", label_column = -1 ).ensure_numeric_labels().normalize()
    >>> print( iris_dataset )
    [1] Dataset(instâncias=150, features=4, classes=3)
    >>> for indice, *features, classe in iris_dataset:
    ...     pass

Main Features:
    - Manipulação da coluna de rótulo.
    - Normalização (Intervalo de [-1, +1]).
    - Separação da base de dados entre treinamento e teste.
    - Remoção de atributos.
    - Conversão da coluna de rótulo para tipo numérico.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Usado para o método split: é possível fazer um split manualmente, mas train_test_split é mais eficiente e há estratificação já implementada.
from sklearn.model_selection import train_test_split

# Realiza alguns imports para a especificação de tipos
from typing import Self, Optional, Union, List


class Dataset:
    """Um wrapper para pandas DataFrames que facilita o uso de base de dados para machine learning.

    Attributes:
        data (pd.DataFrame): O pandas DataFrame com os dados.
        label_column (int): Posição da coluna de rótulo (aceita indexação negativa).
        column_names (Optional[List[str]]): Nome das colunas do DataFrame.
        _label_categories (dict): Mapeia os rótulos codificados para os valores originais.
    """

    def __init__( self, data: pd.DataFrame, *, label_column : int = -1, column_names : Optional[list[str]] = None, _label_categories : dict = None ) -> None:
        """
        Inicializa a classe Dataset.

        Args:
            data: DataFrame com os dados do conjunto de dados.
            label_column: Posição da coluna de rótulo (por padrão é -1, última coluna).
            column_names: Lista opcional com os nomes das colunas.
            _label_categories : Metadado passado entre instâncias, contém o mapeamento inteiro/texto dos rótulos.
        """

        self.data = data                            # Guarda o DataFrame do conjunto de dados
        self.label_column = label_column            # Armazena o índice para a coluna de rótulo
        self.column_names = column_names            # Armazena a lista com os rótulos das colunas do DataFrame
        self._label_categories = _label_categories  # Correspondência inteiro: texto da coluna de rótulo

        # Se for especificado, define o nome das colunas
        if self.column_names is not None:
            self._set_column_names()

    def _set_column_names( self ) -> None:
        """
        Atualiza os rótulos das colunas do DataFrame, conforme o valor no atributo column_names.
        """

        self.data.columns = self.column_names

    def shuffle( self, random_state: Optional[int] = None ) -> Self:
        """
        Retorna uma nova instância da classe com os dados embaralhados.

        Args:
            random_state: Semente opcional para reprodução de resultados.
        
        Returns:
            Uma nova instância do Dataset com os dados embaralhados.
        """

        # Obtém um DataFrame embaralhado e com os índices resetados
        data = self.data.sample(frac=1, random_state=random_state).reset_index( drop=True )

        # Retorna uma nova instância da classe
        return self.__class__( data, label_column = self.label_column, column_names = self.column_names, _label_categories = self._label_categories )
    
    def split( self, train_size: int = 0.8, *, random_state: Optional[int] = None, stratify: bool = False ) -> tuple[Self, Self]:
        """
        Divide o Dataset em conjunto de treinamento e de conjunto de teste.

        Args:
            train_size: Proporção do conjunto de treinamento (0-1).
            random_state: Semente opcional para reprodução de resultados.
            stratify: Se for True, manterá a proporção entre as classes
        
        Returns:
            Tupla( train_set, test_set )
        """

        # Verifica se stratify foi setado
        _stratify = self.y if stratify == True else None

        # Separa o conjunto de dados em dois
        train, test = train_test_split(
            self.data,
            train_size = train_size,
            random_state = random_state,
            stratify = _stratify
        )

        # Retorna as instâncias para Datasets dos conjuntos separados
        return (
            self.__class__( train, label_column = self.label_column, column_names = self.column_names, _label_categories = self._label_categories ),
            self.__class__( test, label_column = self.label_column, column_names = self.column_names, _label_categories = self._label_categories )
        )
    
    def remove_features( self, features_to_remove: List[ Union[int, str] ] ) -> Self:
        """ Remove features do conjunto de dados.
        
        Args:
            features_to_remove: Lista de índices (inteiros) ou de nomes das colunas a serem removidas.
        
        Returns:
            Um novo Dataset com as features restantes.

        Raises:
            ValueError: Formatos ou valores inválidos foram especificados em features_to_remove
        """

        # Realiza uma cópia do DataFrame (garante imutabilidade)
        new_data = self.data.copy( deep = True )
        m, n = new_data.shape

        # Índices a serem removidos
        cols_to_drop = set()

        # Percorre a lista especificada de atributos a serem removidos
        for feature in features_to_remove:
            # O atributo foi especificado como índice
            if isinstance(feature, int):
                # Trata índices negativos
                idx = feature if feature >= 0 else (feature + n)
                
                # Verifica se o índice está fora do intervalo permitido
                if (idx < 0) or (idx >= n):
                    raise ValueError(f"Índice {feature} fora do intervalo permitido.")
                
                cols_to_drop.add( idx )
            
            # O atributo foi especificado como string
            elif isinstance(feature, str):
                # Verifica se o atributo column_names está declarado
                if not self.column_names:
                    raise ValueError("Nomes de colunas não definidos.")
                
                # Recupera o índice do valor especificado
                try:
                    idx = self.column_names.index(feature)
                
                except ValueError:
                    raise ValueError(f"Coluna '{feature}' não encontrada.")
                
                cols_to_drop.add(idx)
            
            else:
                raise ValueError("Os elementos especificador devem ser str ou int")
        
        # Verifica se a coluna de label foi especificada
        if ( (self.label_column >= 0) and (self.label_column in cols_to_drop) ) or ( (self.label_column < 0) and ((self.label_column + n) in cols_to_drop) ):
            raise ValueError("Não é permitido remover a coluna de label")
        
        # Calcula as colunas restantes
        remaining_cols = [i for i in range(n) if i not in cols_to_drop]

        # Atualiza a posição do label = label atual menos a quantidade de atributos anteriores que foram removidos
        resolved_label = self.label_column if self.label_column >= 0 else self.label_column + n
        new_label_column = self.label_column - sum( 1 for col in cols_to_drop if col < resolved_label )

        # Garante que new_label_column seja negativo se o original era
        if self.label_column < 0:
            new_label_column -= (self.data.shape[1] - len(cols_to_drop))

        # Atualiza a lista de nomes, se estiver definida
        remaining_label_column = None

        if self.column_names:
            remaining_label_column = [label for i, label in enumerate(self.column_names) if i not in cols_to_drop]

        # Atualiza o DataFrame para as colunas restantes
        new_data = new_data.iloc[:, remaining_cols]
        
        # Retorna uma nova instância para Dataset com os atributos restantes
        return self.__class__( new_data, label_column = new_label_column, column_names = remaining_label_column, _label_categories = self._label_categories )

    def normalize( self ) -> Self:
        """ Retorna o conjunto de dados com atributos normalizados em [-1, +1] """

        # Cópia do conjunto de dados convertida para float
        new_data = self.data.copy( deep = True ).astype(float)

        # Dimensões do Dataframe
        m, n = new_data.shape

        # Percorre os índices dos atributos
        for idx in range( n ):
            # Se for a coluna de rótulo, pula para a próxima
            if (self.label_column >= 0 and idx == self.label_column) or (self.label_column < 0 and idx == (self.label_column + n)):
                continue

            # Obtém o subconjunto do atributo de índice idx
            column = new_data.iloc[:, idx]

            # Obtém os valores de mínimo e máximo do subconjunto
            _min = np.min( column )
            _max = np.max( column )
            
            # Subconjunto normalizado
            new_column = 2 * (column - _min) / (_max - _min) - 1

            # Ataliza o conjunto de dados com o subconjunto normalizado
            new_data.iloc[:, idx] = new_column
        
        # Retorna uma nova instância normalizada
        return self.__class__( new_data, label_column = self.label_column, column_names = self.column_names, _label_categories = self._label_categories )

    def ensure_numeric_labels( self ) -> Self:
        """ Converte os valores dos rótulos para inteiros. """

        # Cópia do conjunto de dados
        new_data = self.data.copy( deep = True )

        # Obtém o subconjunto do rótulo
        label_column = new_data.iloc[:, self.label_column]

        # Verifica se já é numérico (nada a fazer)
        if pd.api.types.is_numeric_dtype( label_column ):
            return self

        # Usa factorize para converter os rótulos em inteiros
        encoded_labels, uniques = self.data.iloc[:, self.label_column].factorize()

        # Atribui a coluna convertida para 
        new_data.iloc[:, self.label_column] = encoded_labels

        # Dicionário com o mapeamento inteiro-rótulo
        self._label_categories = dict( enumerate(uniques) )

        # Retorna uma nova instância normalizada
        return self.__class__( new_data, label_column = self.label_column, column_names = self.column_names, _label_categories = self._label_categories )
    
    @classmethod
    def from_file( cls, filepath : str, *, comment_marker : str = "#", missing_marker : str = "?", label_column : int = -1, column_names : Optional[list[str]] = None, delimiter : str = "," ) -> None:
        """
        Inicializa a classe a partir de um arquivo CSV.

        Args:
            filepath: Caminho para o arquivo CSV.
            comment_marker: Marcador para linhas de comentário (serão ignoradas).
            missing_marker: Marcador para elementos faltantes (substituídos por NaN).
            Posição da coluna de rótulo (por padrão é -1, última coluna).
            column_names: Lista opcional com o nome das colunas.
            delimiter: Marcador que indica o caracter de separação entre as colunas.
        """

        # Lê o arquivo CSV (Não existe linhas de header no arquivo)
        data = pd.read_csv( filepath, header=None, comment=comment_marker, na_values=missing_marker, delimiter = delimiter )

        # Remover elementos faltantes
        data = data.dropna()

        # Retorna uma chamada para o construtor normal da classe
        return cls( data, label_column = label_column, column_names = column_names )
    
    @property
    def X( self ) -> pd.DataFrame:
        """ Retorna as colunas de atributos """

        return self.data.drop(
            columns = self.data.columns[ self.label_column ]
        )
    
    @property
    def y( self ) -> pd.Series:
        """ Retorna a coluna de rótulo """

        return self.data.iloc[:, self.label_column]
    
    @property
    def shape( self ) -> tuple[int, int, int]:
        """
        Retorna as dimensões do Dataset

        Returns:
            Tupla[m, n, k] onde:
            m = número de instâncias
            n = número de atributos 
            k = número de classes
        """

        m, n_total = self.data.shape
        
        n = n_total - 1 
        k = len( self.y.unique() )

        return (m, n, k)
    
    def __len__( self ):
        """ Retorna o número de instâncias do conjunto de dados. """
        return len( self.data )
    
    def __repr__( self ):
        """ Retorna uma string representando o objeto. """
        m, n, k = self.shape
        return f"Dataset(instâncias={m}, features={n}, classes={k})"
    
    def __iter__( self ):
        """ 
        Gera tuplas no formato (index, *features, classe) para cada instância, garantindo que a classe sempre venha no final, mesmo que label_column não seja -1.
        """

        m, n = self.data.shape

        # Obtém a posição absoluta da coluna de label
        label_column = self.label_column if self.label_column >= 0 else self.label_column + n

        # Remodela o dataset para que a coluna de labels fique no final
        cols = [i for i in range(n) if i != label_column] + [label_column]

        # Percorre o gerador do dataset remodelado
        for row in self.data.iloc[:, cols].itertuples(index = True):
            yield row

# Alguns testes com a classe
if __name__ == "__main__":
    path_dataset = r"datasets\column_3C.dat"
    column_names = ["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius", "degree_spondylolisthesis", "class"]

    vertebral_column_dataset = Dataset.from_file( path_dataset, delimiter=" ", label_column=-1, column_names=column_names ).ensure_numeric_labels()
    print( vertebral_column_dataset )