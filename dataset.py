"""
Classe Dataset para normalizar a manipulação de DataFrames com coluna para rótulos.

Este módulo implementa uma classe "Dataset", que funciona como um wrapper para pandas DataFrames, simplificando
tarefas comuns de machine learning, tais como separação entre atributos/rótulo, normalização de dados,
remoção de atributos e divisão do conjunto de dados.

Autor: Thiago Barbosa
Criado em: 2025

Exemplo de uso:
    >>> from dataset import Dataset
    >>> import pandas as pd
    >>> iris_dataset = Dataset.from_file( filepath = "iris.data", label_column = -1 ).ensure_numeric_labels().normalize()
    >>> print( iris_dataset )
    [1] Dataset(instâncias=150, features=4, classes=3)
    >>> for indice, *features, classe in iris_dataset:
    ...     pass

Principais recursos:
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
from scipy.stats import pearsonr

# Realiza alguns imports para a especificação de tipos
from typing import Self, Optional, Union, List, Iterable


class Dataset:
    """Um wrapper para DataFrames que facilita o uso de base de dados para machine learning.

    Attributes:
        data (pd.DataFrame): O pandas DataFrame com os dados.
        label_column (int): Posição da coluna de rótulo (aceita indexação negativa).
        column_names (Optional[List[str]]): Nome das colunas do DataFrame.
        _label_index_to_name (dict): Mapeia os rótulos codificados para os valores originais.
    """

    def __init__( self, data: pd.DataFrame, *, label_column : int = -1, column_names : Optional[list[str]] = None, _label_index_to_name : dict = None, _label_encodings = None, _label_decodings = None ) -> None:
        """
        Inicializa a classe Dataset.

        Args:
            data: DataFrame com os dados do conjunto de dados.
            label_column: Posição da coluna de rótulo (por padrão é -1, última coluna).
            column_names: Lista opcional com os nomes das colunas.
            _label_index_to_name : Metadado passado entre instâncias, contém o mapeamento inteiro/texto dos rótulos.
            _label_encodings: Metadado passado entre instâncias, garantindo que uma possível codificação seja passado adiante.
            _label_decodings: Metadado passado entre instâncias, garantindo que uma possível codificação seja passado adiante.
        """

        self.data = data                                    # Guarda o DataFrame do conjunto de dados
        self.label_column = label_column                    # Armazena o índice para a coluna de rótulo
        self.column_names = column_names                    # Armazena a lista com os rótulos das colunas do DataFrame
        self._label_index_to_name = _label_index_to_name    # Dicionário com a correspondência do inteiro com o texto da coluna de rótulo

        # Variáveis para uso interno da classe
        self._centroids = None                      # Armazenará os centróides das classes
        self._label_encodings = _label_encodings    # Armazenará o dicionário para codificar classes em vetores
        self._label_decodings = _label_decodings    # Armazenará o dicionário para decodificar vetores em classes

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
        return self.__class__( data, label_column = self.label_column, column_names = self.column_names, _label_index_to_name = self._label_index_to_name )
    
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

        # Reinicia os indices dos subconjuntos
        train = train.reset_index( drop=True )
        test = test.reset_index( drop=True )

        # Retorna as instâncias para Datasets dos conjuntos separados
        return (
            self.__class__( train, label_column = self.label_column, column_names = self.column_names, _label_index_to_name = self._label_index_to_name, _label_encodings = self._label_encodings, _label_decodings = self._label_decodings ),
            self.__class__( test, label_column = self.label_column, column_names = self.column_names, _label_index_to_name = self._label_index_to_name, _label_encodings = self._label_encodings, _label_decodings = self._label_decodings )
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
        m, n_total = new_data.shape

        # Índices a serem removidos
        cols_to_drop = set()

        # Percorre a lista especificada de atributos a serem removidos
        for feature in features_to_remove:
            # O atributo foi especificado como índice
            if isinstance(feature, int):
                # Trata índices negativos
                idx = feature if feature >= 0 else (feature + n_total)
                
                # Verifica se o índice está fora do intervalo permitido
                if (idx < 0) or (idx >= n_total):
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
        if ( (self.label_column >= 0) and (self.label_column in cols_to_drop) ) or ( (self.label_column < 0) and ((self.label_column + n_total) in cols_to_drop) ):
            raise ValueError("Não é permitido remover a coluna de label")
        
        # Calcula as colunas restantes
        remaining_cols = [i for i in range(n_total) if i not in cols_to_drop]

        # Atualiza a posição do label = label atual menos a quantidade de atributos anteriores que foram removidos
        resolved_label = self.label_column if self.label_column >= 0 else self.label_column + n_total
        new_label_column = resolved_label - sum( 1 for col in cols_to_drop if col < resolved_label )

        # Garante que new_label_column seja negativo se o original era
        if self.label_column < 0:
            new_label_column -= (n_total - len(cols_to_drop))

        # Atualiza a lista de nomes, se estiver definida
        remaining_label_column = None

        if self.column_names:
            remaining_label_column = [label for i, label in enumerate(self.column_names) if i not in cols_to_drop]

        # Atualiza o DataFrame para as colunas restantes
        new_data = new_data.iloc[:, remaining_cols]
        
        # Retorna uma nova instância para Dataset com os atributos restantes
        return self.__class__( new_data, label_column = new_label_column, column_names = remaining_label_column, _label_index_to_name = self._label_index_to_name )

    def normalize( self ) -> Self:
        """ Retorna o conjunto de dados com atributos normalizados em [-1, +1] """

        # Cópia do conjunto de dados convertida para float
        new_data = self.data.copy( deep = True ).astype(float)

        # Dimensões do Dataframe
        m, n_total = new_data.shape

        # Percorre os índices dos atributos
        for idx in range( n_total ):
            # Se for a coluna de rótulo, pula para a próxima
            if (self.label_column >= 0 and idx == self.label_column) or (self.label_column < 0 and idx == (self.label_column + n_total)):
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
        return self.__class__( new_data, label_column = self.label_column, column_names = self.column_names, _label_index_to_name = self._label_index_to_name )

    def ensure_numeric_labels( self ) -> Self:
        """ Converte os valores dos rótulos para valores numéricos. """

        # Cópia do conjunto de dados
        new_data = self.data.copy( deep = True )

        # Obtém o subconjunto do rótulo
        label_column = new_data.iloc[:, self.label_column]

        # Verifica se já é numérico (nada a fazer)
        if pd.api.types.is_numeric_dtype( label_column ):
            return self

        # Usa factorize para converter os rótulos em inteiros
        encoded_labels, uniques = self.data.iloc[:, self.label_column].factorize()

        # Substitui a coluna de rótulos pelos valores numéricos
        new_data.iloc[:, self.label_column] = encoded_labels

        # Dicionário com o mapeamento inteiro-rótulo
        self._label_index_to_name = dict( enumerate(uniques) )

        # Retorna uma nova instância normalizada
        return self.__class__( new_data, label_column = self.label_column, column_names = self.column_names, _label_index_to_name = self._label_index_to_name )
    
    def determination_matrix( self ) -> pd.DataFrame:
        """ Retorna a matriz de determinação entre as colunas do dataset """
        m, n_total = self.data.shape

        # Organiza os indices do DataFrame de forma que a coluna de rótulo fique no final
        label_column = self.label_column if self.label_column >= 0 else self.label_column + n_total
        cols = [i for i in range(n_total) if i != label_column] + [label_column]

        dataset = self.data.iloc[:, cols]

        # Organiza a lista de nomes das colunas para que fique no final
        column_names = None

        if self.column_names:
            column_names = np.array(self.column_names)[cols].tolist()

        # Inicializa uma matriz de zeros com a dimensão necessária (#colunas x #colunas)
        determination = pd.DataFrame( 
            np.zeros( (n_total, n_total) ),
            index = column_names,
            columns = column_names
        )

        # Percorre primeiro as linhas
        for i_features in range( n_total ):
            i_column = dataset.iloc[:, i_features]

            # Percorre as colunas
            for j_features in range( n_total ):
                j_column = dataset.iloc[:, j_features]

                # Calcula o coeficiente de correlação de Pearson
                r_pearson = pearsonr( i_column, j_column ).statistic

                # Guarda o valor do quadrado do coef. de Pearson na matriz
                determination.iloc[i_features, j_features] = r_pearson ** 2
        
        # Retorna a matriz de determinação calculada
        return determination
    
    def move_label_to_end(self) -> Self:
        """Move a coluna de label para a última posição, ajustando column_names se existir."""
        m, n_total = self.data.shape

        # Obtém a posição absoluta da coluna de rótulo
        label_column = self.label_column if self.label_column >= 0 else self.label_column + n_total

        # Reorganiza as colunas, deixando o índice do label na última coluna
        cols = [i for i in range(n_total) if i != label_column] + [label_column]

        # Obtém a tabela reorganizada
        new_data = self.data.iloc[:, cols]

        # Reorganiza o nome das colunas
        new_column_names = None

        if self.column_names:
            new_column_names = [self.column_names[i] for i in range(n_total) if i != label_column] + [ self.column_names[label_column] ]

        # Retorna uma nova instância normalizada
        return self.__class__( new_data, label_column = -1, column_names = new_column_names, _label_index_to_name = self._label_index_to_name )

    def vectorize_labels( self ):
        """
            Codifica os rótulos do conjunto de dados em vetores.

            Para cada classe presente no conjunto de dados, este método gera um vetor
            de saída onde a classe correspondente é representada por +1 e todas as demais
            por -1. Requer que os rótulos já estejam em formato numérico.
        """
        
        # Obtém os valores únicos para as classes
        unique_classes = self.y.unique()
        unique_classes = sorted(unique_classes)

        # Codifica as classes na forma de vetores
        encode_classes = np.eye( self.class_count ) 
        encode_classes[ encode_classes == 0 ] = -1

        # Preenche o dicionário para codificação (classe->vetor)
        self._label_encodings = {
            label : encode_classes[i]
            for i, label in enumerate( unique_classes )
        }

        # Preenche o dicionário para decodificação (vetor->classe)
        self._label_decodings = {
            tuple( encode_classes[i] ) : label
            for i, label in enumerate( unique_classes )
        }

    def encode_label( self, label: Union[int, float] ) -> Union[None, np.ndarray]:
        """
            Retorna o vetor correspondente a um rótulo.

            Args:
                label: Valor numérico que representa a classe que se quer obter a codificação.
            
            Returns:
                Retorna o vetor que codifica o rótulo especificado. Retorna None caso não o dicionário
                de codificação não tenha sido inicializado.
        """

        if self._label_encodings is None:
            return None
        
        return self._label_encodings[ label ]
    
    def decode_vector(self, vector: Union[np.ndarray, list, tuple]) -> Union[None, int]:
        """
            Retorna a classe que correspondente ao vetor especificado.

            Args:
                vector: Array, lista ou tupla que especifica o vetor que se quer obter a classe.
            
            Returns:
                Retorna o valor da classe que corresponde ao vator especificado ou retorna None caso não o dicionário
                de decodificação não tenha sido inicializado.
        """

        if self._label_decodings is None:
            return None
        
        return self._label_decodings[ tuple(vector) ]
    
    def decode_prediction(self, output_vector: Union[np.ndarray, list, tuple]) -> Union[None, int]:
        """
            Decodifica um vetor de saída da rede, assumindo codificação (+1, -1), e retorna o rótulo da classe correspondente.

            Args:
                output_vector: Vetor de saída da rede, contendo valores como +1 e -1 para cada classe.
            
            Returns:
                Retorna a classe que corresponde ao vetor especificado ou retorna None caso não o dicionário
                de decodificação não tenha sido inicializado.
        """

        k = self.class_count

        pred_vector = [-1] * k
        idx = np.argmax(output_vector)
        pred_vector[ idx ] = +1
        
        return self.decode_vector( pred_vector )

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
    def class_count(self):
        """ Retorna o número de classes do conjunto de dados """
        return len( self.y.unique() )

    @property
    def features_count(self):
        """ Retorna o número de features do conjunto de dados """
        _, n = self.X.shape
        return n
    
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

        m = self.__len__()
        n = self.features_count
        k = self.class_count

        return (m, n, k)
    
    @property
    def centroids( self ) -> Iterable:
        """ Calcula os centroides do conjunto de dados
        
        Returns:
            Iterable: Um iterável que permite acessar os centroides na forma 
            (index, *features, classe), onde classe é o rótulo do centroide.
        """

        if self._centroids is None:
            # Agrupa por classe e calcula a média (centroide) para cada uma (índice numérico é resetado)
            self._centroids = self.data.groupby( self.y ).mean().reset_index(drop=True)

        return self._centroids.itertuples()
    
    @property
    def label_encodings( self ) -> Union[ None, dict ]:
        """ 
            Método getter para o dicionário de codificação das classes em vetores

            Returns:
                Retorna None caso o método vectorize_labels não tenha sido chamado. Caso
                contrário, retorna o dicionário contendo a codificação usada.
        """

        if self._label_encodings is None:
            return None
        
        return self._label_encodings.copy()
    
    @property
    def label_decodings( self ) -> Union[ None, dict ]:
        """ 
            Método getter para o dicionário de decodificação das classes em vetores

            Returns:
                Retorna None caso o método vectorize_labels não tenha sido chamado. Caso
                contrário, retorna o dicionário contendo a decodificação usada para converter
                os vetores nas classes.
        """

        if self._label_decodings is None:
            return None
        
        return self._label_decodings.copy()
    
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
        for row in self.data.iloc[:, cols].itertuples():
            yield row

# Alguns testes com a classe
if __name__ == "__main__":
    path_dataset = r"datasets\column_3C.dat"
    column_names = ["pelvic_incidence", "pelvic_tilt", "lumbar_lordosis_angle", "sacral_slope", "pelvic_radius", "degree_spondylolisthesis", "class"]

    vertebral_column_dataset = Dataset.from_file( 
        filepath = path_dataset, 
        delimiter = " ", 
        label_column = -1, 
        column_names = column_names 
    ).ensure_numeric_labels()
    
    print( vertebral_column_dataset )

# ======== Carrega os conjuntos de dados na memória

dermatology_dataset = Dataset.from_file(
    filepath = r"datasets\dermatology.data",
    label_column = -1,
    column_names = ["erythema", "scaling", "definite-borders", "itching", "koebner" "phenomenon", "polygonal papules", "follicular papules", "oral-mucosal involvement", "knee elbow involvement", "scalp involvement", "family history", "melanin incontinence", "eosinophils in the infiltrate", "pnl infiltrate", "fibrosis of the papillary dermis", "exocytosis", "acanthosis", "hyperkeratosis", "parakeratosis", "clubbing of the rete ridges", "elongation of the rete ridges", "thinning of the suprapapillary epidermis", "spongiform pustule", "munro microabcess", "focal hypergranulosis", "disappearance of the granular layer", "vacuolisation and damage of the basal layer", "spongiosis", "saw-tooth appearance of retes", "follicular horn plug", "perifollicular parakeratosis", "inflammatory monoluclear infiltrate", "band-like infiltrate", "age", "class"]
).normalize()

wine_dataset = Dataset.from_file( 
    filepath = r"datasets\wine.data", 
    label_column = 0, 
    column_names = ["class", "Alcohol", "Malicacid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols","Proanthocyanins", "Color intensity", "Hue", "0D280 0D315 of diluted wines", "Proline"]
).normalize().move_label_to_end()

iris_dataset = Dataset.from_file( 
    filepath = r"datasets\iris.data", 
    label_column = -1, 
    column_names = ["sepal length", "sepal width", "petal length", "petal width", "class"]
).ensure_numeric_labels().normalize()