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

        # Qual o número de neurônios? É o mesmo que o número de classes
        self.q = self.train_dataset.class_count

        # Qual o número de entradas? É o número de atributos + 1
        self.p = self.train_dataset.features_count + 1

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
                print(f"A rede acertou todo o conjunto treinamento em {epoca} épocas")
                break

            # Exibe uma mensagem de log a cada 5% do número de épocas totais
            elif ( epoca % (max_epocas * 0.05) == 0 ) and verbose:
                print(f"Época {epoca}: erros: {total_erros}")
        
        # Entra quando não houve break, ou seja, ainda houve erros até na última época
        else:
            print(f"Treinamento encerrado com {total_erros} erros após {epoca} épocas.")
    
    def predict( self, features : np.ndarray ) -> Union[float, int]:
        """
            Função usada para prever a classe, dada as features do ponto.

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

class MultiLayerPerceptron:
    """ Classe que implementa uma MLP(p, q1, m) para problemas de classificação. """

    def __init__( self, train_dataset : Dataset, q : int = 3):
        """
            Inicializa a classe MultiLayerPerceptron.

            Args:
                train_dataset: Classe Dataset que contém os exemplos do conjunto de treinamento.
                q: Número de neurônios da camada oculta.
        """

        self.train_dataset = train_dataset
        
        self.q = q                                          # Número de neurônios ocultos
        self.m = self.train_dataset.class_count             # Número de neurônios de saída
        self.p = self.train_dataset.features_count + 1      # Número de entradas da rede

        # Função de ativação usada
        self.activation = lambda x: np.tanh(x)
        self.ddx_activation = lambda x: 1 - self.activation(x) ** 2

        # Inicializa o vetor de pesos dos neurônios ocultos
        self.W = np.random.normal( size = (self.q, self.p) )     # (q, p)

        # Inicializa o vetor de pesos dos neurônios de saída
        self.M = np.random.normal( size = (self.m, self.q+1) )   # (m, q+1)
    
    def train( self, max_epocas : int = 1000, *, eta : float = 0.1, reset_weights = False, verbose = True ):
        """
            Função para o treinamento da rede de neurônios. Usa como base o gradiente descendente.

            Args:
                max_epocas: Número máximo de épocas de treinamento, pode retornar antes caso haja convergência.
                eta: Taxa de aprendizado.
                reset_weights: Flag que faz com que os pesos sejam reinicializados.
                verbosa: Flag para a exibição de mensagens a cada tantos % do número de épocas.
        """

        # Reseta os pesos
        if reset_weights:
            self.W = np.random.normal( size = (self.q, self.p) )
            self.M = np.random.normal( size = (self.m, self.q+1) )
        
        # Percorre um número qualquer de épocas
        for epoca in range( max_epocas ):
            total_erros = 0

            # Para cada época, embaralha o conjunto de treinamento
            shuffled_dataset = self.train_dataset.shuffle()

            # Percorre os exemplos do conjunto de treinamento
            for index, *features, classe in shuffled_dataset:
                # Vetor que representa a classe 
                real_output = self.train_dataset.encode_label( classe )

                # Monta o vetor de entrada
                X_bias = np.r_[features, -1]

                # Sentido direto - cálculo da ativação e a saída de cada camada
                U = self.W @ X_bias         # Ativação de cada neurônio oculto
                Y = self.activation( U )    # Saída dos neurônios ocultos

                Z = np.r_[ Y, -1 ]          # Prepara as entradas para os neurônios de saída

                A = self.M @ Z              # Ativação dos neurônio de saída
                O = self.activation ( A )   # Saída da camada de saída

                # Sentido inverso - atualização dos pesos das camadas
                err = real_output - O 

                # atualiza os pesos da camada de saída
                delta_output = err * self.ddx_activation( A )           # (m x 1)
                delta_weights_out = eta * np.outer( delta_output, Z )   # (m x (q+1))
                self.M = self.M + delta_weights_out

                # calcula os erros retropagados para a camada oculta oculto
                output_weights_no_bias_T = self.M[:, :-1].T                     # (q x m)
                backpropagated_error = output_weights_no_bias_T @ delta_output  # (q x 1)

                # atualiza os pesos da camada oculta
                delta_hidden = backpropagated_error * self.ddx_activation(U)    # (q x 1)
                delta_weights_hidden = eta * np.outer( delta_hidden, X_bias )   # (q x p)
                self.W = self.W + delta_weights_hidden

                # Verifica se a previsão seria acertada ou não
                if np.argmax(O) != np.argmax(real_output):
                    total_erros += 1

            # Se acertou todo o conjunto de treinamento, para o treinamento
            if total_erros == 0:
                print(f"A rede acertou todo o conjunto treinamento em {epoca} épocas")
                break

            # Exibe uma mensagem de log a cada 5% do número máximo de épocas
            elif verbose and not (epoca% round(max_epocas*0.05)):
                print(f"Época {epoca}: {total_erros} erros.")

        # Entra quando não houve break, ou seja, ainda houve erros até na última época
        else:
            print(f"Treinamento encerrado com {total_erros} erros após {epoca} épocas.")

    def predict( self, features : np.ndarray ) -> Union[float, int]:
        """
            Função usada para prever a classe, dada as features do ponto.

            Args:
                features: Array com as features do ponto a ser classificado.
            
            Returns:
                Retorna o número que indica a classe estimada.
        """

        # Monta o vetor de entrada
        X_bias = np.r_[features, -1]

        # Sentido direto - cálculo da ativação e a saída de cada camada
        U = self.W @ X_bias         # Ativação de cada neurônio oculto
        Y = self.activation( U )    # Saída dos neurônios ocultos

        Z = np.r_[ Y, -1 ]          # Prepara as entradas para os neurônios de saída

        A = self.M @ Z              # Ativação dos neurônio de saída
        O = self.activation ( A )   # Saída da camada de saída

        # Monta um vetor de predição baseado no argmax da saída da rede
        predicted_output = -np.ones_like(O)
        predicted_output[ np.argmax(O) ] = +1

        # Retorna a classe correspondente ao vetor predito pela rede
        return self.train_dataset.decode_vector( predicted_output ) 