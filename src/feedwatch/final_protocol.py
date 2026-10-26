"""Registro prospectivo: esta etapa não lê manifest, test.csv ou vídeos."""

import json
from pathlib import Path

from .data import CLASS_IDS, file_hash, utc_now, write_json


def freeze_protocol(config_path: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("O protocolo exige uma pasta nova; não substitua um registro existente")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config["test_date"], config["expected_videos"], config["primary_metric"],
        config["primary_candidate"], config["operational_seed"]) != ("2022_6_22", 1501, "macro_f1", "gru", 42):
        raise ValueError("Decisões finais diferem do protocolo solicitado")
    expected = {"dummy", "random_forest", "mobilenet_linear"} | {
        f"{kind}_seed_{seed}" for kind in ("mlp", "gru") for seed in (42, 43, 44)}
    if {model["id"] for model in config["models"]} != expected or len(config["models"]) != 9:
        raise ValueError("São exigidos os nove checkpoints predefinidos")
    # Somente artefatos históricos de treino/validação. Nenhum acesso ao manifest/teste.
    paths = {Path(config_path), Path(config["motion_config"]), Path(config["visual_config"])}
    for model in config["models"]:
        paths.update((Path(model["checkpoint"]), Path(model["validation_predictions"])))
    visual = json.loads(Path(config["visual_config"]).read_text(encoding="utf-8"))
    paths.add(Path(visual["extraction_identity"]["weights"]["weights_file"]))
    hashes = {str(path): file_hash(path) for path in sorted(paths)}
    config.update({"frozen_at_utc": utc_now(), "class_mapping": CLASS_IDS,
                   "locked_input_sha256": hashes,
                   "extraction_identities": {
                       "motion": json.loads(Path(config["motion_config"]).read_text(encoding="utf-8"))["extraction_identity"],
                       "visual": visual["extraction_identity"]}})
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    text = f"""# Protocolo final prospectivo — FeedWatch

Registrado em UTC: {config['frozen_at_utc']}. Este protocolo e a configuração foram salvos antes de ler os vídeos ou rótulos do teste nesta etapa. A preparação histórica já indexou/auditou dados; isso não foi avaliação de modelos.

- Teste existente: **2022_6_22, 1.501 vídeos**. Treino: dias 19–20; validação: dia 21. Divisões e rótulos não serão alterados.
- Métrica principal: **macro-F1** nas quatro classes. Complementares: acurácia, precisão/recall/F1 por classe, suporte e matriz de confusão (linhas reais, colunas previstas; none=0, weak=1, medium=2, strong=3). Classes sem previsões usam zero_division=0.
- Avaliar Dummy mais frequente do treino, RandomForest, MobileNetV3 Small congelada + linear, e MLP/GRU das seeds **42, 43, 44**, exclusivamente pelos nove checkpoints já existentes listados em `config.json`, com SHA-256.
- Candidata principal: **GRU cronológica**, pelo maior macro-F1 médio da validação entre as cabeças controladas. Isso **não implica benefício temporal demonstrado**; a ablação prévia não mostrou prejuízo médio com embaralhamento. Não repetir busca de ordens no teste.
- Referência operacional para a futura demonstração: **GRU seed 42**, definida agora. Não escolher outra seed pelo teste.
- Não treinar, ajustar hiperparâmetros, fazer fine-tuning, selecionar épocas/modelos/limiares pelo teste ou criar ensemble. Inferência por argmax, com as quatro classes originais. Média de métricas não constitui ensemble.
- Reutilizar exatamente os extratores e pré-processamentos históricos, inclusive ROI, grade temporal, recorte central e normalização ImageNet. Movimento: 5 Hz nos primeiros 2 s, cinza 320×180, diferenças/fluxo óptico. Visual: 16 frames cronológicos dos primeiros 2 s, RGB, resize proporcional lado menor 256 e center crop 224, sem padding. CNN congelada em eval, sem gradientes. Normalização das cabeças carregada dos checkpoints, aprendida somente no treino.
- Cache somente com identidade, configuração, pesos, caminho, tamanho/mtime, forma e cronologia compatíveis. Cache incompatível será registrado e reextraído com os parâmetros originais em pasta nova; falhas serão registradas por arquivo, sem imputação por zeros. Não escrever nos caches antigos.
- Dummy e RF usam a mesma cobertura de características de movimento. Todos os modelos terão métricas sobre sua cobertura e comparação adicional na interseção dos vídeos válidos de todos eles. Informar N/falhas e salvar listas; não ocultar exclusões.
- Salvar previsões por vídeo, probabilidades, métricas, matrizes, configurações e hashes dos checkpoints. Comparar validação e teste, incluindo taxas e contagens medium→strong, strong→none e recall de medium. Recarregar todos os checkpoints e verificar novamente IDs/probabilidades; conferir métricas pelos CSVs.
- MLP e GRU: cada seed e média ± desvio-padrão amostral (ddof=1), incluindo métricas por classe e matrizes. Variação entre seeds mede instabilidade de treinamento, **não incerteza entre dias/fazendas**. Referências anteriores têm uma seed e configurações de treino diferentes.
- Um único dia reservado e clipes correlacionados limitam a interpretação. A seleção repetida na mesma validação pode torná-la otimista. Nenhum resultado autoriza extrapolação para outras fazendas ou atribuição causal ao uso da ordem.
- Não modificar o projeto em resposta aos erros do teste. Não implementar interface. A futura demonstração deverá expor limitações de classes, domínio, recorte e falhas de decodificação. O projeto classifica **intensidade alimentar observada**; não comprova fome, saciedade nem segurança para controlar um alimentador.

Configuração reproduzível: `config.json` (SHA-256: `{file_hash(output / 'config.json')}`). Os hashes e timestamps de `protocol_seal.json` vinculam este registro às decisões anteriores ao acesso ao teste.
"""
    (output / "protocol.md").write_text(text, encoding="utf-8")
    seal = {"sealed_at_utc": utc_now(), "before_test_access": True,
            "config_sha256": file_hash(output / "config.json"),
            "protocol_sha256": file_hash(output / "protocol.md")}
    write_json(output / "protocol_seal.json", seal)
    return seal


def verify_protocol(output: Path) -> dict:
    seal = json.loads((output / "protocol_seal.json").read_text(encoding="utf-8"))
    if not seal["before_test_access"] or any(file_hash(output / f"{name}.{ext}") != seal[f"{name}_sha256"]
        for name, ext in (("config", "json"), ("protocol", "md"))):
        raise ValueError("Protocolo/configuração foram alterados após o registro")
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if any(file_hash(Path(path)) != digest for path, digest in config["locked_input_sha256"].items()):
        raise ValueError("Checkpoint/configuração histórica mudou após o registro")
    return config
