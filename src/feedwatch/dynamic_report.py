"""Measured, deliberately limited report for each reproducible dynamic demonstration."""

import json


def write_dynamic_report(output, report):
    entries = report["examples"]
    total = report["processing_seconds_including_model_load_montage_verification"]
    montage = report["montage"]["verification"]
    counts = {key:sum(e["counters"][key] for e in entries) for key in entries[0]["counters"]
              if key != "max_detections_in_frame"}
    timings = {key:sum(e["timings"][key] for e in entries) for key in entries[0]["timings"]}
    verification = json.loads((output/"verification.json").read_text(encoding="utf-8"))
    memory = [json.loads((output/e["folder"]/"analysis.json").read_text(encoding="utf-8")).get("gpu_peak_allocated_bytes_since_process_start") for e in entries]
    memory = [m for m in memory if m is not None]
    hardware = report["hardware"]
    mapping = ("O **Fishial pré-treinado externamente** fornece caixas da classe Fish e a contagem de detecções no frame. "
               "O **rastreador** associa caixas, atribui IDs temporários e calcula trilhas curtas e deslocamento aparente dos centros em pixels/s. "
               if report["detector"] else "Detector e rastreador estão explicitamente desligados: não existem caixas, IDs ou contagens. ")
    observed_limits = ("A inspeção real confirma peixes visíveis sem caixa, propostas agrupando indivíduos e caixas ambíguas junto à região central. "
        "Não há anotações de detecção/rastreamento para estimar precisão, recall, erro de contagem ou trocas de ID. "
        "Sem caixa significa que o detector não propôs uma caixa válida, não que não há peixe. IDs são associações temporárias; podem fragmentar ou trocar. "
        "Deslocamentos elevados ou irregulares podem refletir jitter e associação incorreta: não são medidas comprovadas de natação ou centímetros/s. "
        if report["detector"] else "O detector está desligado nesta exportação. Não se estimam peixes, identidades, contagens ou velocidades de trajetórias a partir do fluxo. ")
    rows = "\n".join(f'| {i:02d} | {e["class_name"]} | {e["predicted_class"]} | {e["counters"]["frames_with_detections"]}/{e["metadata"]["frame_count"]} | {e["processing_seconds"]:.2f} |'
                     for i,e in enumerate(entries,1))
    text = f"""# FeedWatch — demonstração dinâmica com filmagem real

[Assistir ao MP4](feedwatch-real-montage.mp4) · [Página local](index.html) · [Seleção](selection.csv) · [Linha do tempo](montage-timeline.csv)

Foram exportados **{len(entries)} clipes independentes de validação**, três por classe, com a mesma seleção seed {report['selection_seed']} do MVP anterior, sem filtrar por acerto ou qualidade de detecção. São **{report['sum_real_video_seconds']:.2f} s de filmagem real** e cartões de corte, totalizando **{montage['duration_seconds']:.2f} s**, {montage['width']}×{montage['height']}, {montage['fps']:g} FPS e {montage['frame_count']} frames. Não havia uma fonte contínua adequada: os 5.603 clipes de treino/validação já auditados têm 2 s. A montagem não simula uma sessão contínua; rastreador, fluxo e gráfico reiniciam em cada clipe. Fontes selecionadas sem áudio.

## O que cada elemento significa

A **MobileNet congelada + GRU seed 42** produz somente a classe alimentar **da cena** e os scores por janela, com o recorte/normalização/amostragem históricos. Nesta montagem aparece “Análise do clipe completo”, offline; o rótulo do dataset está separado da previsão na página/cartão. {mapping}O **Farneback** calcula o mapa, vetores azuis e gráfico de movimento da imagem inteira, incluindo água e reflexos. Não localiza peixes e não é saída da GRU. Os cantos do recorte indicam a entrada alimentar, não uma detecção.

O modo contínuo da CLI mantém `analyze_video`/`display_state` originais: só mostra uma previsão após terminar a janela correspondente; cauda menor que 2 s não é preenchida/classificada. Detector, associação, fluxo e gráfico usam somente o frame atual e anteriores. Não houve treinamento, fine-tuning, ensemble, alteração alimentar ou nova avaliação do teste.

## Verificações e medições desta execução

- **{verification['historical_predictions_matched']}/12 previsões alimentares iguais às históricas**; diferença máxima de scores {max(e['maximum_score_difference'] for e in entries):.3g}. A recarga independente da GRU com embeddings cronológicos históricos reproduziu todos os IDs e scores (diferença máxima {max(e['maximum_score_difference'] for e in verification['checkpoint_reload_replays']):.3g}). Pesos/buffers congelados conferidos.
- Cada MP4 foi integralmente decodificado depois de exportado; a montagem também. Originais têm SHA-256 conferido antes/depois. `verification.json` registra a recarga; a auditoria adicional fica em `independent_verification.json` quando executado `scripts/verify_dynamic_demo.py`.
- Tempo total desde o início da função, incluindo carregar modelos, exportar, montar e verificar: **{total:.2f} s** ({total/60:.2f} min). **{report['real_source_frames']} frames reais / {total:.2f} s = {report['effective_processing_fps_end_to_end']:.3f} FPS efetivo**. Soma das exportações: {sum(e['processing_seconds'] for e in entries):.2f} s; por clipe {min(e['processing_seconds'] for e in entries):.2f}–{max(e['processing_seconds'] for e in entries):.2f} s. Importar/iniciar Python e assistir no navegador não entram nesse total.
- Soma por componente: detector {timings['detector_seconds']:.2f} s; associação {timings['tracking_seconds']:.2f} s; fluxo {timings['motion_seconds']:.2f} s; renderização/codificação {timings['render_encode_seconds']:.2f} s. Há ainda decodificação, classificação alimentar, hashes e verificações.
- Hardware: {hardware['gpu'] or 'CPU'}, CPU {hardware['cpu']}, {hardware['platform']}, Python {hardware['python']}, PyTorch {hardware['torch']}, OpenCV {hardware['opencv']}. Pico alocado pelo PyTorch: {str(round(max(memory)/1024**2,2))+' MiB' if memory else 'não disponível'}. Não é a VRAM total utilizada pelo sistema.
- Cobertura computacional do detector: {counts['frames_detector_ok']}/{report['real_source_frames']} frames; {counts['frames_with_detections']} com ao menos uma proposta. Foram {counts['detection_observations']} observações de caixas e {counts['observations_with_speed']} deslocamentos consecutivos calculáveis, **não peixes únicos ou acertos anotados**. Fluxo disponível em {counts['frames_motion_ok']} frames; o primeiro de cada clipe inicializa sem movimento inventado. Falhas visuais registradas: {sum(e['visual_errors'] for e in entries)}.

| Clipe | Rótulo do dataset | Previsão GRU | Frames com detecção | Processamento (s) |
|---|---|---|---:|---:|
{rows}

## Limitações observadas

{observed_limits}O mapa responde também à água e aos reflexos.

Os erros alimentares anteriores permanecem: os três weak são previstos como medium e dois medium como strong nesta seleção seed 42. Estes 12 exemplos não são uma nova estimativa de generalização. A demonstração não corrige o baixo recall de medium nem comprova benefício temporal da GRU. O desempenho entre janelas contínuas não foi avaliado.

**Reprodução a {montage['fps']:g} FPS é diferente de inferência em tempo real**; o FPS efetivo medido nesta execução é muito menor. Na apresentação por `python -m http.server`, o navegador pode não permitir avançar por tempo porque o servidor simples não oferece HTTP Range; para navegar livremente, abra o MP4 localmente. Isso não altera a duração ou os frames do arquivo.

## Barras coloridas da demonstração anterior

O arquivo de entrada era `artifacts/demos/technical-checks/synthetic_5p2s_audio.mp4`, padrão FFmpeg testsrc2 com áudio. A saída era `artifacts/demos/technical-checks/sequential/annotated.mp4`. Ambos mostram as mesmas barras no frame 75 (3 s), com hashes correspondentes ao registro anterior; a diferença média absoluta em uma região sem overlay foi 1,963 níveis BGR, compatível com recodificação. `strong` foi uma classificação fora do domínio em um **teste técnico sintético**, não uma troca de filmagem na exportação. Esses arquivos não integram o vídeo real. [Auditoria](../../dynamic-investigation/source_audit.json) · [Original](../../dynamic-investigation/synthetic_original_3s.jpg) · [Saída](../../dynamic-investigation/synthetic_output_3s.jpg).

## Reprodução e arquivos

Comandos no [README](../../../README.md) e método em [docs/methodology.md](../../../docs/methodology.md). Use uma pasta nova/vazia. `config.json` registra parâmetros; `execution_source_sha256.json` registra o código da exportação. Cada pasta contém vídeo, `analysis.json`, `windows.csv`, `frames.jsonl` e `frames.csv`, com scores não calibrados, caixas, IDs, trilhas, estatísticas, tempos e falhas. O detector oficial tem classe Fish e pesos SHA-256 `{report['detector']['weights_sha256'] if report['detector'] else 'desligado'}`; identificação completa em `demo_report.json`.

**Intensidade alimentar observada — protótipo experimental. Não comprova fome, saciedade nem segurança para controlar um alimentador.**
"""
    (output/"report.md").write_text(text,encoding="utf-8")
