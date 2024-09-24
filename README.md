# Introdução às Redes Neurais

## O que é uma Rede Neural?

Uma **rede neural** é um modelo computacional inspirado na estrutura e funcionamento do cérebro humano. Assim como os neurônios biológicos processam informações e tomam decisões, uma rede neural artificial é composta por **neurônios artificiais** (ou "nós") organizados em camadas, que processam dados e aprendem a realizar tarefas complexas, como reconhecimento de padrões, previsão de resultados e classificação de dados.

### Estrutura Básica

Uma rede neural geralmente é composta por três tipos principais de camadas:

1. **Camada de Entrada (Input Layer)**: 
   - Esta camada recebe os dados de entrada. Cada neurônio da camada de entrada representa uma característica ou atributo dos dados.
   
2. **Camada(s) Oculta(s) (Hidden Layer(s))**: 
   - Entre a entrada e a saída, há uma ou mais camadas ocultas que realizam a maior parte do processamento. Nelas, os dados são transformados através de operações matemáticas (geralmente combinações lineares e funções de ativação não lineares).
   
3. **Camada de Saída (Output Layer)**: 
   - Esta camada fornece o resultado final após o processamento. O número de neurônios nesta camada depende da tarefa. Por exemplo, em uma tarefa de classificação com três classes, teríamos três neurônios na camada de saída.

### Como Funciona?

1. **Forward Propagation (Propagação para Frente)**: 
   - Os dados de entrada são enviados pela rede, passando de camada em camada. Cada neurônio recebe um valor de entrada, aplica pesos e somas e, em seguida, passa o resultado por uma **função de ativação** (como ReLU, Sigmoid, Tanh) para introduzir não-linearidade. O objetivo é transformar dados brutos em saídas úteis.
   
2. **Erro e Função de Custo**: 
   - Depois que os dados passam por todas as camadas e chegam à camada de saída, o resultado é comparado com o valor esperado (conhecido como **rótulo**). A diferença entre a saída prevista e o valor correto é chamada de **erro**, e é medida por uma **função de custo** (como o erro quadrático médio).

3. **Backward Propagation (Propagação para Trás)**: 
   - Para melhorar o desempenho da rede, ela ajusta os **pesos** e **biases** dos neurônios. Isso é feito através de um processo chamado **backpropagation**, onde o erro é propagado de volta da camada de saída até a camada de entrada, atualizando os parâmetros para minimizar o erro.

4. **Treinamento**: 
   - O treinamento de uma rede neural consiste em repetir várias vezes o ciclo de **forward propagation** e **backward propagation**, ajustando os pesos e biases a cada iteração até que o erro seja minimizado. Isso é feito com a ajuda de algoritmos de otimização, como o **Gradiente Descendente**.

### Aplicações das Redes Neurais

Redes neurais são extremamente versáteis e têm sido aplicadas em muitas áreas, incluindo:

- **Visão Computacional** (reconhecimento facial, identificação de objetos)
- **Processamento de Linguagem Natural** (tradução automática, análise de sentimentos)
- **Previsão Financeira**
- **Sistemas de Recomendação** (como os usados pela Netflix e Amazon)
- **Diagnóstico Médico**

### Exemplos de Redes Neurais

- **Perceptron**: A forma mais simples de uma rede neural, composta por apenas um neurônio.
- **Multilayer Perceptron (MLP)**: Uma rede neural com uma ou mais camadas ocultas.
- **Convolutional Neural Networks (CNNs)**: Usada principalmente para processar dados visuais.
- **Recurrent Neural Networks (RNNs)**: Ideal para sequências de dados, como texto ou séries temporais.
