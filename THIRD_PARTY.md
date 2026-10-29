# Créditos e componentes de terceiros

O código do FeedWatch é disponibilizado sob AGPL-3.0, conforme LICENSE. Dependências, datasets, fontes e modelos externos conservam seus próprios termos; não são apresentados como autoria do FeedWatch.

| Componente | Origem / termos |
|---|---|
| AV-FFIA | Meng Cui e colaboradores. https://zenodo.org/records/11059975 e https://zenodo.org/records/11060195 — CC BY 4.0, confirmado nos metadados da API. Exemplos de apresentação foram redimensionados e tiveram áudio omitido. |
| U-FFIA | Referência científica: https://arxiv.org/abs/2309.05058. O FeedWatch é uma implementação própria, não uma reprodução integral do método. |
| Vídeo de aquário | Richard Muller, Pexels 6472322. https://www.pexels.com/video/fishes-swimming-in-aquarium-6472322/ — licença Pexels: https://www.pexels.com/license/. |
| Vídeo de carpas | lam loi, Pexels 3625177. https://www.pexels.com/video/feeding-koi-fish-with-processed-fish-foods-3625177/ — licença Pexels. |
| Fishial | Wye Foundation / Fishial.AI, revisão f43074efba44a7c833457d0ecf2b96a403e8283d. Repositório: https://github.com/Wye-Foundation/Fishial-Fish-Identification — licença MIT para o código do repositório. Os dois detectores pré-treinados são baixados da distribuição original, com hashes em config/simple_video.json. O pacote detector_v26_n3.zip contém o YOLO26 nano; detector_v10_m5.zip é descrito pelo fornecedor como YOLOv12 Medium, mas seus metadados identificam YOLOv12s, Ultralytics 8.3.158 e licença AGPL-3.0. Os termos do checkpoint são distintos dos termos do repositório. |
| Ultralytics | Runtime do detector, obtido separadamente. https://www.ultralytics.com/license — opção aberta AGPL-3.0. |
| MobileNetV3 Small | TorchVision, pesos IMAGENET1K_V1 da distribuição oficial: https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth. A cabeça GRU foi treinada pelo FeedWatch; o extrator permaneceu congelado. |
| Barlow Semi Condensed | Jeremy Tribby e colaboradores, distribuída pelo Google Fonts. SIL Open Font License, incluída em sites/feedwatch/dist/fonts/OFL.txt. |

Os vídeos são utilizados para demonstrar esta aplicação e não como banco de mídia. Não há vínculo ou endosso de seus autores. As versões anotadas e os cartões de apresentação são derivados identificados; os originais locais permanecem preservados. O pacote de execução inclui a GRU do projeto, os pesos MobileNet necessários à reprodução e um vídeo Pexels de exemplo; o detector é obtido separadamente do fornecedor original.

Os GIFs `docs/assets/feedwatch-demo.gif` e `docs/assets/feedwatch-demo-pt.gif` mostram um trecho reduzido, sem áudio, da grade de análise do vídeo de aquário de Richard Muller. As capturas de interface em `docs/assets/` usam essa mesma gravação. As classificações e métricas exibidas foram calculadas pelo projeto; os tempos são relativos ao vídeo original.
