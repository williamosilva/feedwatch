# Método

O FeedWatch estima a intensidade alimentar aparente de um grupo de peixes em vídeo. As classes são `none` (não observada), `weak` (baixa), `medium` (moderada) e `strong` (alta).

## Classificação

O extrator MobileNetV3 Small, pré-treinado e congelado, produz representações de 16 imagens por janela de dois segundos. Uma GRU treinada no subconjunto utilizado do AV-FFIA classifica cada janela completa. O resultado fica disponível após o fim do intervalo. Caudas incompletas e falhas permanecem sem classificação.

O [AV-FFIA/U-FFIA](https://github.com/FishMaster93/U-FFIA), de Meng Cui e colaboradores, fornece dados de intensidade alimentar. O FeedWatch usa vídeo e parte desse conjunto; não reproduz integralmente o modelo multimodal da pesquisa.

## Panorama

O panorama soma o tempo de cada categoria. Informa a classe predominante, os empates, a distribuição percentual do tempo classificado e a duração sem resultado. A predominância do vídeo só é apresentada quando pelo menos 80% da duração recebeu classificação. Abaixo disso, o resultado é insuficiente; a distribuição bruta e o tempo sem classificação continuam disponíveis. As classes são ordinais: uma média numérica arbitrária de seus nomes não é calculada.

A evolução divide a duração total em três partes iguais: início, meio e final. Janelas que atravessam uma divisão contribuem proporcionalmente ao tempo de interseção. A comparação exige, em cada parte, pelo menos quatro segundos, duas janelas distintas, 80% de cobertura e uma classe em mais de 50% do tempo classificado.

Essas regras são critérios descritivos de apresentação, sem validação biológica. O mínimo técnico de 12 segundos não estabelece uma duração suficiente para avaliar uma alimentação. A comparação informa somente a relação entre as predominâncias do início e do final; não afirma uma tendência contínua.

## Detecção e rastreamento

Um detector Fishial pré-treinado encontra peixes visíveis. O rastreador associa detecções e desenha trajetórias temporárias. Caixas, IDs e classificação alimentar são análises separadas. IDs podem se perder ou mudar; a contagem por imagem não representa a população do tanque. O resumo informa também as imagens sem contagem e as imagens com zero detecções. Zero detecções não comprova ausência de peixes. Precisão do detector e identidade biológica ainda não foram medidas com anotações independentes.

Os relatórios de processamento incluem configuração, origem, hashes, janelas, observações por imagem, falhas e metadados. O deslocamento opcional em pixels/s é uma medida aparente da imagem, sem conversão para velocidade física.

## Mapa de movimento

O painel colorido compara imagens consecutivas da gravação original com o método de [fluxo óptico de Farneback, disponível no OpenCV](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html). A escala é fixa: azul indica menor movimento aparente e vermelho, maior movimento, com saturação em 400 pixels da imagem original por segundo. A resolução de análise é proporcional, com até 640 pixels no lado maior.

A curva mostra o movimento médio da imagem ao longo do tempo. O percentual de área em movimento corresponde aos pixels com deslocamento estimado acima de 20 pixels por segundo. Esses valores incluem água, reflexos e movimento da câmera. Não medem temperatura, velocidade física dos peixes ou atenção da GRU, e não alteram a classificação alimentar.

O primeiro quadro não possui comparação anterior e aparece sem medição. Falhas permanecem indisponíveis. O mapa e as métricas usam os mesmos tempos do vídeo principal; não há normalização independente das cores a cada quadro.

## Simulação de alimentação

O usuário informa a quantidade habitual, a quantidade do cenário e, opcionalmente, o preço por kg, sempre para o mesmo período. O cálculo compara quantidade, variação percentual e custos. A diferença em kg é a quantidade habitual menos a quantidade do cenário. Custo é quantidade multiplicada pelo preço por kg.

O modelo não escolhe a quantidade do cenário nem determina percentuais de ajuste. Classificação baixa não comprova saciedade; classificação alta não determina quanto acrescentar. A simulação não mede economia, ingestão ou desperdício. Uma recomendação de manejo exige validação adicional.
