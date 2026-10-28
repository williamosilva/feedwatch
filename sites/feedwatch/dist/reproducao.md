# Reproduzir o FeedWatch

Código AGPL-3.0. Baixe `feedwatch-source.zip` e `feedwatch-runtime.zip` e extraia ambos na mesma pasta. O pacote de execução usa caminhos relativos e preserva a GRU seed 42; não contém o dataset completo.

No Windows, com Python 3.12, abra PowerShell nessa pasta:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[local]"
.\.venv\Scripts\python.exe -m feedwatch prepare-fish-detector
.\.venv\Scripts\python.exe -m feedwatch serve
```

Para usar GPU, consulte a instalação do Torch/TorchVision no README. O detector é baixado da distribuição Fishial original; o comando verifica o hash configurado. O processamento também funciona em CPU, mas pode levar mais tempo. O pacote inclui um vídeo de exemplo para analisar na aplicação local.

A demonstração web reproduz resultados já calculados. O teste histórico e a avaliação adicional usam conjuntos diferentes de vídeos. Os detalhes estão no relatório de avaliação e nos créditos. Os hashes dos downloads estão em `downloads.json`.
