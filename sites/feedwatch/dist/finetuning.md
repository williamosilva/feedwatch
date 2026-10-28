# Partial fine-tuning

[Português](#português)

This experiment trains MobileNetV3 Small blocks 9–12 together with the existing GRU. Earlier blocks and all BatchNorm running statistics stay fixed. The application keeps its current model until the candidate passes the comparison criteria.

The approach adapts the transfer-learning and image-augmentation principles in [Cui et al.](https://arxiv.org/abs/2309.05058) to FeedWatch's existing network. It does not reproduce their S3D architecture or their reported accuracy. See also the [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

## Recorded result

The three runs completed 48 epochs in total. All three improved validation macro-F1. The table reports their mean scores on the same 1,705 validation clips; category percentages are recall.

| Validation measure | Original | Fine-tuned |
|---|---:|---:|
| Accuracy | 58.53% | 61.06% |
| Macro-F1 | 0.5664 | 0.5972 |
| Not observed | 72.07% | 61.15% |
| Low | 55.32% | 71.21% |
| Moderate | 17.64% | 28.05% |
| High | 88.25% | 78.14% |

The candidate was **not adopted**. Recall fell by 10.92 percentage points for not observed and 10.10 points for high activity, exceeding the 5-point limit. The fixed operational seed also gained only 0.0192 macro-F1, below its 0.03 requirement. The retrospective audit sets were not opened for this experiment. These are development-validation results, not a new estimate of test accuracy.

The recorded runs selected epochs 3, 18 and 9 for seeds 42, 43 and 44. Saved checkpoints reproduced the reported predictions. The early MobileNet layers, BatchNorm running statistics and original GRU checkpoint files remained unchanged. Training used 736,488 encoder parameters and 58,692 GRU parameters, with no new data download.

## Comparison

- Same AV-FFIA clips: 3,898 for training on June 19–20 and 1,705 for validation on June 21.
- Three paired runs, seeds 42, 43 and 44, each initialized with its original trained GRU.
- Sixteen chronological frames per clip. Small crop, brightness, contrast and saturation changes are sampled together for each clip, then shared across its frames. Validation images are unchanged.
- Up to 24 epochs, with early stopping after six epochs without improvement. Epoch zero is the unchanged model and can remain the best result.
- Checkpoints are selected by validation macro-F1: an average that gives the four activity classes equal weight.

The fixed validation criteria require a mean macro-F1 gain of 0.03, a moderate-class recall gain of 10 percentage points, no other class losing more than 5 points of recall, and an improvement in all three runs. Seed 42 must independently gain at least 0.03 macro-F1. Recall measures how many examples of a class the model recognizes correctly.

Only a passing candidate proceeds to the already available June 22 and June 27–29 evaluation sets. Those sets have been examined in earlier experiments, so a new comparison is a **retrospective audit**, not a fresh untouched test. No new data is downloaded, and no hyperparameters are adjusted using those audit results.

[Configuration](evidence/finetune-protocol.json) · [Results](evidence/finetune-results.json) · [Training curves](learning/finetune-learning.en.svg)

## Reproduce

Use Python 3.12 and the pinned project dependencies. The recorded configuration requires a local CUDA GPU; it was sized for an RTX 4060 Ti with 8 GB. The decoded image cache uses approximately 13.5 GB in addition to the dataset and existing experiment files.

With the [June 19–22 dataset](https://zenodo.org/api/records/11059975/files/video_6.19-6.22.zip/content) extracted into `video_6.19-6.22`, first reproduce the original feature extractor and GRU experiments:

```powershell
python -m feedwatch index --data-dir video_6.19-6.22
python -m feedwatch split --config config/splits.json
python -m feedwatch visual-experiment --data-dir video_6.19-6.22 --output-dir artifacts/experiments/visual-full
python -m feedwatch temporal-experiment --data-dir video_6.19-6.22 --output-dir artifacts/experiments/temporal-controlled
python scripts/train_finetune_candidate.py
python scripts/build_finetune_evidence.py
```

Run these commands inside the project's virtual environment. Install the matching CUDA builds of Torch and TorchVision first. The fine-tuning script reads existing local weights and videos, checks the original predictions before training, and saves candidates separately in `artifacts/experiments/finetune-v4`. It does not replace the application's weights. Each completed seed is retained when the command is run again; an interrupted seed starts again from its original initialization.

## Português

O experimento ajusta as últimas camadas da MobileNet e a GRU que o projeto já utiliza. Usa os mesmos vídeos, separados por dia. Pequenas variações de iluminação, cor e recorte ajudam a testar se o modelo depende demais da aparência específica do treino. Cada trecho recebe uma única transformação, aplicada igualmente às 16 imagens, sem mudar sua ordem.

São três treinamentos, comparados com seus modelos anteriores. A troca exige melhora nas quatro classes em conjunto, maior reconhecimento da atividade moderada e perda limitada nas demais classes. O resultado precisa se repetir nas três execuções. Os critérios foram registrados antes do treinamento.

Os comandos acima reproduzem o experimento com os dados já utilizados pelo projeto. Não há serviço pago nem aquisição de novos vídeos. Os candidatos ficam separados do modelo em uso; executar o treinamento não altera a aplicação. Uma comparação melhor em clipes rotulados ainda não mede quantidade de ração ou economia.

O ajuste elevou o acerto médio na validação de 58,53% para 61,06%. Reconheceu mais exemplos de atividade baixa e moderada, mas menos exemplos de atividade não observada e alta. Essas duas perdas ultrapassaram o limite definido antes do treino. Por isso, o modelo anterior continua na aplicação. Os conjuntos reservados para comparação posterior não foram usados neste experimento.
