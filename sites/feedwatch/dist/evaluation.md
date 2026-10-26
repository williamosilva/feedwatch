# Avaliação

## Dados e resultados

Foram utilizados 7.104 clipes do AV-FFIA, separados por dias inteiros: 3.898 para treino (19–20/06/2022), 1.705 para validação (21/06) e 1.501 para o teste histórico (22/06). A GRU operacional utiliza seed 42.

| Avaliação | Acurácia | Macro-F1 | Recall da moderada |
|---|---:|---:|---:|
| Teste histórico, 1.501 clipes | 60,56% | 0,5903 | 18,31% |
| Outros três dias, 224 clipes | 53,57% | 0,4914 | 8,93% |

A amostra adicional foi fixada antes da inferência, com 224 clipes dos dias 27–29/06/2022. São clipes correlacionados do mesmo contexto de pesquisa, com rótulos publicados. Os resultados não certificam transferência para outras fazendas, câmeras ou espécies.

| Modelo na amostra adicional | Acurácia | Macro-F1 | Recall da moderada |
|---|---:|---:|---:|
| GRU operacional | 53,57% | 0,4914 | 8,93% |
| GRU com perda ponderada | 52,68% | 0,4923 | 10,71% |
| Random Forest | 52,23% | 0,5137 | 28,57% |

O modelo operacional foi mantido. Nove treinos posteriores, com três configurações de regularização e três seeds, também não atingiram os critérios de melhoria. O macro-F1 médio de validação da melhor configuração foi 0,5669, frente a 0,5664 da referência. Os dias 23–26/06 continuam reservados, sem avaliação nessa seleção.

A etapa escolhida do modelo operacional registrou 98,46% de acerto no treino e 59,24% na validação. A diferença evidencia dificuldade de generalização. Scores não representam probabilidades de acerto calibradas.

## Evidências

Os resultados completos, protocolos, seleção e procedência estão nos arquivos JSON e CSV de [evidence](evidence/independent-results.json). Os [dados dos gráficos](learning/evidence.json), o [histórico de treinamento](learning/operational-history.csv) e a [matriz de confusão](learning/confusion-matrix.csv) permitem conferir as visualizações.

## Treino com mais dias de gravação

O experimento adicional usou 4.374 clipes de seis dias para treino (13–16 e 19–20/06/2022) e 1.865 clipes de dois dias para validação (17 e 21/06). Foram acrescentados 476 clipes ao treino e 160 à validação, obtidos do registro oficial AV-FFIA com verificação de CRC e SHA256. A amostragem equilibrou dia, sessão e categoria usando somente os dados de treino. O extrator continuou congelado, e a GRU manteve a arquitetura inicial.

Os dois treinamentos foram comparados nos mesmos vídeos de validação, com três inicializações cada. A tabela dá o mesmo peso a cada dia e inicialização; esses números não são diretamente comparáveis às avaliações históricas acima.

| Validação dos dias 17 e 21 | Macro-F1 médio | Recall médio da moderada | Recall médio da alta |
|---|---:|---:|---:|
| Modelo anterior | 0,5913 | 25,90% | 82,46% |
| Novo treino | 0,6068 | 35,40% | 75,27% |

A categoria moderada melhorou 9,49 pontos percentuais, mas a alta perdeu 7,19 pontos. O ganho de Macro-F1 ficou abaixo de 0,03, e a perda de recall de outra categoria ultrapassou o limite de 5 pontos percentuais. O candidato não passou pelos critérios definidos antes do treino. O modelo operacional foi mantido, e o dia 18/06 reservado para exame final não foi baixado nem avaliado. Os testes de 22 e 23–29/06 não foram usados nessa seleção.

O [protocolo](evidence/diverse-training-protocol.json), os [resultados das três inicializações](evidence/diverse-validation-results.json) e as [curvas do novo treino](learning/diverse-evidence.json) registram esse resultado, inclusive os critérios reprovados. Esses dados ampliam os dias observados no mesmo local de pesquisa; não representam novas fazendas ou espécies.

## Detecção e acompanhamento

A configuração atual combina dois detectores Fishial em todas as imagens: YOLO26 nano com limiar 0,60 e YOLOv12s com limiar 0,15. Uma etapa de remoção de duplicatas usa IoU 0,55. Não há troca de detector por vídeo, máscara sobre a mão ou exclusão por posição. A classificação alimentar continua separada dessa detecção.

Uma comparação visual provisória em cinco quadros reservados dos dois exemplos, com 54 caixas de referência, registrou:

| Detecção, IoU mínimo 0,50 | Acertos | Falsos positivos | Peixes não encontrados | Precisão | Recall |
|---|---:|---:|---:|---:|---:|
| Configuração anterior | 25 | 15 | 29 | 62,50% | 46,30% |
| Configuração atual | 33 | 17 | 21 | 66,00% | 61,11% |

As caixas de referência são provisórias, com incerteza sobre peixes parcialmente visíveis ou sobrepostos. São quadros dos mesmos dois vídeos de demonstração, sem revisão humana independente. Esses percentuais servem para uma verificação de regressão local, não como estimativa de precisão geral. Uma primeira reserva foi incorporada ao desenvolvimento após revelar que o ajuste inicial ainda marcava a mão; a segunda configuração foi fixada antes de avaliar esses cinco novos quadros.

Na exportação completa das carpas, as imagens com zero detecções passaram de 347 em 562 contagens disponíveis para 1 em 563. A caixa sobre a mão no quadro 28 (0,93 s) deixou de aparecer, mas outra caixa incorreta ainda aparece sobre parte da mão no quadro 180 (6 s). O exemplo continua identificado como sujeito a falhas. O aquário manteve contagem disponível nas 395 imagens, sem imagens com zero detecções. Esses totais descrevem as saídas do detector, não medem a quantidade real de peixes.

ByteTrack e BoT-SORT foram comparados ao rastreador conservador com as mesmas detecções em 24 quadros consecutivos, acompanhando dois alvos selecionados por cena. Os três registraram 48 observações associadas, sem perdas ou trocas de identificador nessa pequena verificação. As caixas de identidade de referência foram interpoladas entre extremos inspecionados, com conferência no quadro intermediário. Não é uma avaliação independente de rastreamento ou de oclusões longas. Sem ganho demonstrado, `conservative-v2` continua como padrão; os outros dois podem ser selecionados em `tracker.kind` na configuração.

Os [resultados de detecção e rastreamento](evidence/detection-regression.json) descrevem os limites dessa comparação. O exportador também registra o timestamp bruto do decodificador e a origem do tempo usado pelo rastreador. Corrige somente pequenas inconsistências compatíveis com o relógio de quadros; falhas maiores continuam indisponíveis.

## Verificação de software

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

A suíte cobre separação dos dados, treinamento e carregamento, falhas de vídeo, exportação, disponibilidade temporal dos resultados, agregação por duração e aritmética de cenários. Os testes de evolução incluem sessões sintéticas de 50 e 60 segundos. Esses testes verificam o software; não são novas medições biológicas.

## Avaliação contínua pendente

Selecionar gravações autorizadas de 50–60 segundos, sem cortes ou repetição. Separar sessões, dias e câmeras entre desenvolvimento e avaliação final, fixando a seleção antes de consultar as previsões.

Dois revisores, incluindo alguém com conhecimento do domínio, devem rotular os intervalos de dois segundos sem ver a saída do modelo. Usar `none`, `weak`, `medium`, `strong` ou `unobservable`; registrar discordâncias antes do consenso. A [ficha vazia](session-review-template.csv) contém 25 intervalos de um formulário de 50 segundos, sem vídeo ou rótulos atribuídos. Para 60 segundos, acrescentar cinco intervalos.

Registrar também a leitura humana da gravação inteira e das três partes. Medir acerto da predominância, erro da distribuição temporal, concordância início/final, cobertura, macro-F1 e recall por classe. Reportar indisponibilidade e resultados por sessão e câmera. Uma resposta ausente não conta como acerto.

Não há revisão humana concluída para essas sessões contínuas nem economia demonstrada. Medir economia exige comparar ração oferecida, desperdício e desempenho dos peixes em uma avaliação de manejo.

## Ferramenta de revisão por sessão

O comando abaixo prepara dois formulários em branco, uma ficha de consenso e um manifesto com o hash do vídeo. A página de revisão mostra somente a gravação original. Ela permite baixar o CSV e salva rascunhos separados para os revisores A e B. A opção `--copy-video` inclui uma cópia do vídeo para que o formulário possa ser enviado a outro computador.

Substitua os caminhos, a data e os dados de origem pelos da gravação. Faça a seleção antes de consultar as previsões e confirme que o vídeo é contínuo e está autorizado para o estudo:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_sessions.py prepare --video dados/sessao-01.mp4 --output-dir artifacts/session-review/sessao-01 --session-id sessao-01 --camera-id camera-01 --date 2026-10-25 --source-url "Gravacao propria" --authorization "Uso autorizado pelo responsavel" --continuity-note "Gravacao original sem cortes" --selection-basis "Primeira sessao elegivel do dia" --continuous --predictions-unseen --split final --copy-video
```

Cada revisor deve trabalhar em sua própria cópia de `review.html`, sem consultar as respostas do outro. Salve os arquivos como `review-a.csv` e `review-b.csv` na pasta da sessão. Os identificadores dos revisores precisam ser diferentes; pelo menos um deve declarar conhecimento do domínio. Nos pontos em que discordarem, preencham a categoria e a justificativa em `consensus.csv`. Use `unobservable` quando a imagem não permitir avaliar, em vez de atribuir atividade ausente.

Depois das revisões, processe exatamente a gravação selecionada. Copie o `analysis.json` da execução para a pasta da sessão. O avaliador verifica o hash do vídeo, a identificação do modelo, a duração, o FPS e o plano completo das janelas; um CSV de previsões sem procedência não é aceito.

```powershell
.\.venv\Scripts\python.exe -m feedwatch annotate-simple --video dados/sessao-01.mp4 --output-dir artifacts/session-runs/sessao-01
Copy-Item -LiteralPath artifacts/session-runs/sessao-01/analysis.json -Destination artifacts/session-review/sessao-01/analysis.json
.\.venv\Scripts\python.exe scripts/evaluate_sessions.py evaluate --manifest artifacts/session-review/sessao-01/session.json --output-dir artifacts/session-evaluation/resultado-01
```

É possível passar vários manifestos ao mesmo comando. A ferramenta rejeita sobreposição de câmera, dia ou arquivo entre desenvolvimento e avaliação final, além de relatórios que misturem modelos diferentes. Registra concordância entre revisores, acerto por classe, predominância do vídeo, distribuição temporal, comparação entre início e final e cobertura, por sessão e por câmera. Tempo observável sem previsão entra como erro; tempo marcado como não observável não entra como atividade ausente. A predominância automática fica indisponível abaixo de 80% de cobertura.

O modo `--practice --split development` aceita outras durações para aprender a usar o formulário. Resultados desse modo são identificados como prática e não entram na validação final de sessões de 50–60 segundos. As declarações dos revisores e de continuidade são registradas, mas a ferramenta não certifica que houve revisão independente ou que a gravação não contém cortes.
