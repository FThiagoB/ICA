from dataset import Dataset

import pandas as pd
import numpy as np

from typing import Union

def dist_euclidiana( P1 : Union[ np.ndarray, list[float] ], P2 : Union[ np.ndarray, list[float] ] ) -> float:
    """ 
        Calcula a distância entre dois vetores. 

        Args:
            P1: Array ou lista com as features do primeiro ponto.
            P2: Array ou lista com as features do segundo ponto.
        
        Returns:
            Returna a distância entre os pontos informados.
    """

    distance = np.subtract( P1, P2 ) ** 2
    distance = np.sqrt( np.sum( distance ) )
    
    return distance

def KNN( train_dataset : Dataset, P : np.ndarray, k : int = 3 ) -> float:
    """
        Função que estima uma classe baseado no algoritmo de vizinhos mais próximos.

        Args:
            train_dataset: Classe Dataset que contém os exemplos do conjunto de treinamento.
            P: Array que representa as features do elemento que se quer classificar.
            k: Inteiro que indica a quantidade de vizinhos mais próximos considerada.
        
        Returns:
            Retorna a classe estimada para o ponto especificado.
    """

    distances = []
    nearest_class = None

    # Percorre as instâncias de treinamento
    for index, *features, classe in train_dataset:
        # Armazena o indice do elemento de treinamento e a distância dele até o ponto classificado
        distances.append( (index, dist_euclidiana( features, P )) )
    
    # Ordena a lista em função da distância
    distances = sorted( distances, key = lambda el: el[1] )

    # Obtém a posição dos k elementos mais próximos
    positions = [el[0] for i, el in enumerate(distances) if i < k]

    # Obtém a classe dos k elementos mais próximos
    classes = train_dataset.y[ positions ].to_numpy()

    # Obtém a classe com a maior moda
    classes, moda = np.unique( classes, return_counts = True )
    nearest_class = classes[ np.argmax(moda) ]
    
    return nearest_class

def DMC( train_dataset : Dataset, P : np.ndarray ) -> float:
    """
        Função que estima uma classe baseado no algoritmo mínima distância aos centroides.

        Args:
            train_dataset: Classe Dataset que contém os exemplos do conjunto de treinamento.
            P: Array que representa as features do elemento que se quer classificar.
        
        Returns:
            Retorna a classe estimada para o ponto especificado.
    """

    nearest_class = None
    min_distance = float('inf')
    
    # Percorre os centroides
    for _, *features, classe in train_dataset.centroids:
        current_distance = dist_euclidiana( features, P )

        # Se a distância calculada for menor que menor distância atual, atualiza
        if current_distance < min_distance:
            min_distance = current_distance
            nearest_class = classe
    
    return nearest_class

class PerceptronSimples:
    """ Classe que implementa o Perceptron Simples para problemas de classificação. """

    def __init__( self, train_dataset : Dataset ):
        """
            Inicializa a classe PerceptronSimples.

            Args:
                train_dataset: Classe Dataset que contém os exemplos do conjunto de treinamento.
        """

        self.train_dataset = train_dataset

        # Dimensões do dataset
        self.m, self.n, self.k = self.train_dataset.shape

        # Qual o número de neurônios? É o mesmo que o número de classes
        self.q = self.k

        # Qual o número de entradas? É o número de atributos + 1
        self.p = self.n + 1

        # Inicializa os pesos
        self.W = np.random.normal( size = (self.q, self.p) )
    
    def train( self, max_epocas : int = 1000, *, eta : float = 0.1, reset_weights = False, verbose = True ) -> None:
        """
            Função para o treinamento da rede de neurônios.

            Args:
                max_epocas: Número máximo de épocas de treinamento, pode retornar antes caso haja convergência.
                eta: Taxa de aprendizado.
                reset_weights: Flag que faz com que os pesos sejam reinicializados.
                verbosa: Flag para a exibição de mensagens a cada tantos % do número de épocas.
        """

        # Reseta os pesos
        if reset_weights:
            self.W = np.random.normal( size = (self.q, self.p) )

        # Percorre o total de épocas
        for epoca in range( max_epocas ):
            total_erros = 0

            # Para cada época, embaralha o conjunto de treinamento
            shuffled_dataset = self.train_dataset.shuffle()

            # percorre os exemplos de treinamento embaralhado
            for index, *features, y in shuffled_dataset:
                # Vetor de saída que corresponde à classe verdadeira
                real_output = self.train_dataset.encode_label( y )

                # Vetor de entrada, adiciona o "-1" do viés à lista de features do exemplo
                X_bias = np.r_[features, -1]   

                # Calcula a ativação dos neurônios: ordem de saída = (q x 1)
                activations = self.W @ X_bias

                # Calcula a saída da rede considerando a função signal
                output = np.where( activations >= 0, 1, -1 )

                # Calcula o vetor de erro
                err = real_output - output

                # Se algum elemento não for zero, houve erro.
                if np.any(err != 0):
                    total_erros += 1

                    # Atualiza os pesos usando o produto externo
                    self.W += eta * np.outer(err, X_bias)

            # Se acertou todo o conjunto de treinamento, para o treinamento
            if total_erros == 0:
                print(f"A rede acertou todo o conjunto treinamento em {epoca+1} épocas")
                break

            # Exibe uma mensagem de log a cada 5% do número de épocas totais
            elif ( epoca % (max_epocas * 0.05) == 0 ) and verbose:
                print(f"Época {epoca+1}: erros: {total_erros}")
        
        # Entra quando não houve break, ou seja, ainda houve erros até na última época
        else:
            print(f"Treinamento encerrado com {total_erros} erros após {epoca+1} épocas.")
    
    def predict( self, features : np.ndarray ) -> Union[float, int]:
        """
            Função usada para prever a classe dada as features do ponto.

            Args:
                features: Array com as features do ponto a ser classificado.
            
            Returns:
                Retorna o número que indica a classe estimada.
        """

        # Vetor de entrada, adiciona o "-1" do viés à lista de features passada
        X_bias = np.r_[features, -1]

        # Calcula a ativação dos neurônios
        activations = self.W @ X_bias

        # Calcula a saída da rede considerando a função signal
        output = np.where( activations >= 0, 1, -1 )

        # Inicializa o vetor de saída com -1
        predicted_output = np.full_like(output, -1)

        # Usa argmax para resolver ativações múltiplas
        predicted_output[ np.argmax(output) ] = +1

        # Retorna a classe correspondente ao vetor predito pela rede
        return self.train_dataset.decode_vector( predicted_output ) 