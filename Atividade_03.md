Tarefa - Semana 03
Autor: José Lindenberg de Andrade - 20241009460

Parte 01: Análise análise aprofundada do capítulo 0 (zero) do do livro do livro "The
Principles of Deep Learning Theory"

I) Principais Conceitos:

O capítulo fala sobre o aprendizado profundo utilizando redes neurais artificiais como um modelo subjacente para IA, neste modelo as redes são treinadas com dados rotulados, ajustando os pesos das conexões entre neurônios para minimizar a perda. Ele se inspira no modelo de redes neurais biológicas, mas é bem mais pensada. As redes neurais artificiais são estruturas matemáticas compostas por camadas de neurônios interconectados. Cada neurônio recebe entradas, realiza operações matemáticas nelas e produz uma saída que é transmitida para os neurônios da próxima camada. As redes neurais profundas utilizam os modelos de aprendizagem profunda onde são treinados em dados do mundo real e aprendem como resolver problemas, aprendem representações úteis do mundo, pois possuem muitos neurônios organizados em paralelo em camadas computacionais sequenciais. Assim, o aprendizado profundo aproveita as redes neurais para realizar tarefas complexas de forma eficaz.

 Um conceito abordado é a aprendizagem por representação, ela permite que os modelos aprendam automaticamente características relevantes e abstratas dos dados, transformando os dados em formas cada vez mais refinadas.
Uma rede neural normal é constituída por inúmeras unidades computacionais que são denominadas neurônios, esses são organizados paralelamente em camadas. O diferencia uma rede normal normal de uma rede neural profunda é o fato que na profunda os neurônios são organizados por múltiplas camadas em sequência.

Existem dois aspectos que são essenciais de uma rede neural: sua largura e sua profundidade. Elas são cruciais para fazer alterações na rede, como crescer seu tamanho ou aumentar sua profundidade, o que vai depender do limite de largura infinita da rede que é algo que vale destaque, pois pode tornar tudo simples ou mais complicado, sendo necessário ser calculado.
Alguns conceitos são destacados neste capítulo como o princípio de dispersão, o aprendizado de representação e a teoria das perturbações. O princípio da dispersão se refere a abordagem que busca espalhar a representação dos dados em várias unidades neurais ao longo da rede, promovendo uma distribuição mais uniforme dos padrões de ativação ao longo das camadas da rede.

Já o aprendizado de representação,  refere-se ao processo pelo qual uma rede neural aprende a extrair automaticamente características ou representações significativas dos dados de entrada. Por fim, a teoria das perturbações é utilizada para corrigir o limite de largura infinita, ela investiga como pequenas mudanças nos dados de entrada ou nos parâmetros da rede podem afetar o comportamento e o desempenho da rede neural.

II) Implicações Teóricas: 

Esses conceitos abordados são muito importantes no entendimento de redes neurais profundas, pois tratam de questões que podem acontecer no trabalho com redes neurais. Alterações na rede de neurônios, no que se refere sua largura e sua profundidade, trouxeram o entendimento de situações que podem acontecer e de caminhos problemáticos, como é o caso dos limites de largura infinita e a importância de calculá-los. Neste sentido, trazer questões de caminhos percorridos anteriormente para que não se repitam os mesmos erros, é eficaz, pois conduz ao usuário qual via é mais prudente seguir.  As maneiras de fazer a rede neural crescer de tamanho é um exemplo concreto disso.

Aprender como treinar de forma eficaz é algo discutido ao longo do capítulo. Os problemas relatados ao longo do texto são de cenários realistas de aprendizagem profunda, onde o algoritmo de aprendizagem tem que lidar com os ajustes nos parâmetros (treinamento). Os problemas encontrados, como os que conduziram à definição do algoritmo de aprendizado ser iterativo, auxiliando no entendimento do desafio de lidar com as redes neurais profundas, mas elencam algumas dependências, como os os parâmetros treinados dependerem de uma forma muito complicada de todas as quantidades na inicialização.

 

III) Avaliação Crítica: 

A falta de resolução dos problemas relacionados à representação da série de Taylor para estudo da função treinada deixam muitas limitações no uso de redes neurais profundas. E o trabalho abordado neste capítulo se fecha nestas limitações, sendo um ponto negativo pois ignora a busca de outras soluções para o problema.

A explicação sobre aspectos essenciais de uma arquitetura de rede neural, sua largura e sua profundidade, são aspectos positivos, definem bem as limitações no texto e explica como lidar com essas limitações.

Parte  02: Resumo detalhado do Capítulo 5 do Livro "Designing
Machine Learning Systems: An Iterative Process for Production-Ready
Applications, Chip Huyen"

I) Principais Pontos: 
Neste tópico está resumido os principais pontos do Capítulo 5

Engenharia de características:

Às vezes a aprendizagem profunda é designada por aprendizagem de características. Neste sentido, muitas características podem
ser aprendidas e extraídas automaticamente por algoritmos. Essas características podem ser extraídas automaticamente ou podem ainda ser trabalhadas manualmente. Assim, este é um processo de usar o conhecimento para extrair informações, propriedades e atributos relevantes dos dados brutos. Um exemplo disso é o uso de n-gramas para encontrar palavras ou grupos de palavras específicas para identificar um padrão de informação, como é destacado no próximo ponto.

N-grama:

O n-grama é uma sequência contínua de n elementos de uma determinada amostra de texto. Podendo ser fonemas, sílabas, letras ou palavras. Por exemplo, na frase “Eu amo viajar”, os seus 1-gramas de nível de palavra são ["Eu", "amo", "viajar"] e os seus 2-gramas de nível de palavra são ["Eu amo", "amo viajar"]. Se quisermos que n seja 1 e 2, é: ["Eu", "amo", "viajar", "Eu amo", "amo viajar"].
Tratamento de valores em falta: Muito dos valores  de dados estão e nem todos os tipos de valores em falta são iguais. Neste sentido, existem três tipos de valores em falta: Ausência não aleatória (MNAR), Desaparecido ao acaso (MAR) e Desaparecido completamente ao acaso (MCAR). Ausência não aleatória (MNAR) ocorre quando o motivo da falta de um valor é o próprio valor verdadeiro. Desaparecido ao acaso (MAR) ocorre quando a razão pela qual um valor está faltando não é por causa do valor em si, mas a outra variável observada. E o Desaparecido completamente ao acaso (MCAR) acontece quando não existe um padrão na altura em que o valor está em falta.
Para lidar com valores em falta existem duas formas: a imputação que serve para preencher os valores em falta com determinados valores e a eliminação que tem a finalidade de remover os valores em falta. 

Escalonamento: 

Este é um processo que se faz necessário antes de inserir característica nos modelos, escalonando-os para que tenham intervalos semelhantes, processo chamado por escalonamento de características. Isso resulta num aumento de desempenho do modelo, evitando que o modelo faça previsões sem sentido. Embora possa produzir ganhos de desempenho em muitos casos, não funciona em todos os casos. Além disso, é uma fonte comum de fuga de dados e muitas vezes requer estatísticas globais, sendo necessário olhar para a totalidade ou para um subconjunto de dados de treino para calcular o seu mínimo, máximo ou média.
Discretização: Também conhecido como quantização ou binning, este é um processo de transformar uma caraterística contínua numa caraterística discreta. Ela foca em reduzir a complexidade dos dados, tornando-os mais fáceis de interpretar. Cada valor ou variável neste processo é alocado a um intervalo correspondente. A discretização é útil quando lidamos com dados contínuos difíceis de analisar.

Codificação de características categóricas:

Para quem não trabalha com dados em produção as categorias são estáticas, o que infere que as categorias não sofre alterações ao longo do tempo, e isso se encaixa para muitas categorias e seu tratamento é bem simples.. Mas na produção, as categorias mudam. Assim, a codificação de características categóricas permite que algoritmos de ML trabalhem com variáveis categóricas, transformando-as em representações numéricas adequadas para resolver os problemas em ML.

 Cruzamento de características:
 
 Esta é uma técnica útil para modelar as relações não lineares entre características. Ela permite combinar duas ou mais características para gerar novas características. Sendo essencial para modelos que não podem aprender ou são ruins em aprender relações não lineares, como regressão linear, regressão logística e modelos baseados em árvore, pois ajuda a modelar relações não lineares entre variáveis. Não é muito importante em redes neurais, mas pode ser utilizado pois o cruzamento explícito de características ajuda as redes neurais a aprender relações não lineares mais rapidamente.Um ponto negativo do cruzamento de características é o fato de poder aumentar o espaço das características.

Embeddings posicionais:

O Embeddings é definido como sendo um vetor que representa um dado. Um espaço de embeddings, seria então um conjunto de embeddings geradas pelo mesmo algoritmo para um tipo de dados. O embeddings de palavras é um dos mais utilizados, em que cada palavra pode ser representada por um vetor. Os embeddings posicionais  são definidos em Discretos que representam objetos em categorias específicas, como a categoria de um filme (comédia, drama, ação); e Contínuos que representam objetos em um espaço contínuo, como vetores que capturam características de palavras em processamento de linguagem natural (NLP).

Fuga de dados: 

Ocorre quando uma forma do rótulo “vaza” para o conjunto de características utilizadas para fazer previsões, e essa mesma informação não está disponível durante a inferência. Essa fuga é um desafio, pois na maioria dos casos a fuga não é esperada. Sendo muito perigosa, pois pode fazer com que os modelos falhem, mesmo após avaliações e testes. Dentre as causas vale destacar: 
- Dividir os dados correlacionados com o tempo de forma aleatória em vez de os dividir por tempo;
- Escalonamento antes da divisão;
- Preenchimento de dados em falta com estatísticas da divisão do teste;
- Mau tratamento da duplicação de dados antes da divisão;
- Fuga de grupo;
- Fuga do processo de geração de dados.

  Para detectar uma fuga de dados é preciso antes entender que esta fuga de dados pode acontecer durante muitos passos, o que vai desde a geração, recolha, amostragem, divisão e processamento de dados até à engenharia de características. Assim sendo, em um projeto de ML é importante monitorar a fuga de dados durante todo o seu ciclo de vida. Assim neste processo de detecção é importante:
- Medir o poder preditivo de cada caraterística ou de um conjunto de características relativamente à variável-alvo (etiqueta);
- Efetuar estudos para medir a importância de uma caraterística ou de um conjunto de características para o seu modelo.
- Estar  atento às novas funcionalidades adicionadas ao seu modelo.
- Ter muito cuidado sempre que olhar para a divisão de teste, pois se utilizar a divisão de teste de qualquer outra forma que não seja para reportar o desempenho final de um modelo, arrisca-se a deixar escapar informação do futuro para o seu processo de treino.
  
Características de engenharia:

Neste ponto, é preciso entender que mais características nem sempre significam um melhor desempenho do modelo, pois ter demasiadas características pode ser ruim tanto durante o treino como ao serviço do modelo. Neste sentido, é preciso destacar as seguintes razões:
- Quanto mais funcionalidades tiver, mais oportunidades existem para a fuga de dados.
- Demasiadas características podem causar sobreajuste.
- Demasiadas funcionalidades podem aumentar a memória necessária para servir um modelo.
- Demasiadas características podem aumentar a latência da inferência quando se faz uma previsão online.
  
As funcionalidades inúteis tornam-se dívidas técnicas, pois sempre que o seu pipeline de dados é alterado, todas as funcionalidades afectadas têm de ser ajustadas em conformidade.
As características são importantes e existem muitos métodos diferentes para medir a importância de uma caraterística. Outro ponto é que uma vez que o objetivo de um modelo de ML é fazer previsões corretas em dados não vistos, as características utilizadas para o modelo devem ser generalizadas para dados não vistos.

II) Identificação das Melhores Práticas: 

De início, quando se trata de detecção de fuga de dados, o que é essencial para garantir a integridade e a eficácia dos modelos de aprendizado de máquina. É preciso destacar que ao detectar e corrigir a fuga de dados, é possível garantir que o modelo seja treinado de maneira justa e equitativa. A fuga de dados pode resultar em um modelo que não generaliza bem para dados novos e não vistos. Detectar e corrigir a fuga de dados ajuda a garantir que o modelo seja capaz de fazer previsões precisas e confiáveis em situações do mundo real. A detecção de fuga de dados também contribui para manter a consistência dos dados ao longo do tempo. Isso é crucial em cenários onde os dados estão sujeitos a mudanças, como em sistemas de recomendação ou detecção de fraudes. Em muitos casos, a fuga de dados pode expor informações sensíveis ou confidenciais. Ao identificar e corrigir a fuga de dados, é possível proteger a privacidade dos usuários e garantir o cumprimento das regulamentações de proteção de dados, como o GDPR.

Outro ponto, é que o tratamento de valores em falta é fundamental para garantir a eficácia e a confiabilidade dos modelos de aprendizado de máquina. Uma das razões pelas quais essa prática é importante e como contribui para o sucesso de um modelo é que evita a distorção nos dados, pois quando os dados têm valores ausentes, simplesmente ignorá-los pode levar a uma distorção nos resultados do modelo. Isso ocorre porque a falta de dados pode ser sistemática e não aleatória, o que pode levar a conclusões errôneas. Tratar os valores em falta de maneira adequada ajuda a mitigar esse problema. Outra é que em muitos casos, os dados ausentes contêm informações valiosas que podem ser cruciais para a precisão do modelo. Ao tratá-los corretamente, é possível aproveitar ao máximo todas as informações disponíveis, o que pode melhorar significativamente o desempenho do modelo.
Além disso, a falta de dados pode resultar em conjuntos de dados incompletos e inconsistentes, o que pode prejudicar a capacidade do modelo de generalizar para novos dados. Tratar os valores em falta ajuda a manter a integridade dos dados, garantindo que o modelo seja treinado com informações consistentes e confiáveis. E os modelos de aprendizado de máquina robustos são capazes de lidar com uma variedade de situações e cenários, incluindo dados ausentes. Ao implementar estratégias eficazes de tratamento de valores em falta torna o modelo mais robusto e capaz de lidar com dados incompletos de forma adequada.

Por fim, a prática de limitar o número de características em um modelo de aprendizado de máquina é crucial para evitar o overfitting, reduzir o impacto do ruído nos dados e facilitar a interpretação do modelo, além de economizar recursos computacionais. A adição de mais características pode deixar o modelo mais complexo e levar a um ajuste excessivo aos dados de treinamento, comprometendo sua capacidade de generalização para novos dados. Além de que nem todas as características são igualmente informativas, e selecionar cuidadosamente as mais relevantes ajuda a melhorar a eficácia do modelo. Modelos mais simples e com menos características são mais fáceis de interpretar. Ademais reduzir o número de características economiza recursos computacionais, tornando o treinamento do modelo mais eficiente.. Em suma, limitar o número de características contribui significativamente para o sucesso e a confiabilidade dos modelos de aprendizado de máquina, garantindo que sejam mais robustos, generalizáveis, interpretáveis e eficientes em uma variedade de aplicações.

