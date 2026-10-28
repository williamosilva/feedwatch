import argparse
import json
import sys
from pathlib import Path

from .data import data_root, index_dataset, make_splits, protect_originals


def parser() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser(description="FeedWatch: experimentos e exportação de vídeos anotados")
    commands = app.add_subparsers(dest="command", required=True)
    diagnose = commands.add_parser("diagnose", help="Verifica ambiente, FFmpeg, GPU e operações CUDA")
    diagnose.add_argument("--output", type=Path, default=Path("artifacts/environment.json"))
    index = commands.add_parser("index", help="Indexa data/grupo/classe/arquivo.mp4 e registra problemas")
    index.add_argument("--data-dir", type=Path, required=True)
    index.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    audit = commands.add_parser("audit", help="Audita uma amostra reproduzível; --full é explícito")
    audit.add_argument("--data-dir", type=Path, required=True)
    audit.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    audit.add_argument("--output-dir", type=Path, default=Path("artifacts/audit"))
    audit.add_argument("--seed", type=int, default=42)
    size = audit.add_mutually_exclusive_group()
    size.add_argument("--sample-size", type=int, default=160)
    size.add_argument("--full", action="store_true")
    audit.add_argument("--frames-per-video", type=int, default=5)
    split = commands.add_parser("split", help="Separa dias inteiros; verifica cobertura e vazamento")
    split.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    split.add_argument("--config", type=Path, default=Path("config/splits.json"))
    split.add_argument("--output-dir", type=Path, default=Path("artifacts/splits"))
    summary = commands.add_parser("summary", help="Gera contagens e resumo verificável")
    summary.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    summary.add_argument("--audit-dir", type=Path, default=Path("artifacts/audit"))
    summary.add_argument("--splits-dir", type=Path, default=Path("artifacts/splits"))
    summary.add_argument("--output-dir", type=Path, default=Path("artifacts/summary"))
    examples = commands.add_parser("examples", help="Salva frames com rótulos do dataset, não previsões")
    examples.add_argument("--data-dir", type=Path, required=True)
    examples.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    examples.add_argument("--audit-dir", type=Path, default=Path("artifacts/audit"))
    examples.add_argument("--output-dir", type=Path, default=Path("artifacts/examples"))
    examples.add_argument("--seed", type=int, default=42)
    motion = commands.add_parser("motion-experiment", help="Dummy e RandomForest por movimento; somente treino/validação")
    motion.add_argument("--data-dir", type=Path, required=True)
    motion.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    motion.add_argument("--splits-dir", type=Path, default=Path("artifacts/splits"))
    motion.add_argument("--config", type=Path, default=Path("config/motion.json"))
    motion.add_argument("--cache-dir", type=Path, default=Path("artifacts/motion-cache"))
    motion.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/motion-full"))
    motion.add_argument("--workers", type=int, default=4)
    motion.add_argument("--sample-per-class", type=int, help="Piloto: limita exemplos por classe em cada divisão")
    review = commands.add_parser("motion-review", help="Confere a floresta persistida e prepara clipes dos erros")
    visual = commands.add_parser("visual-experiment", help="MobileNetV3 Small congelada + cabeça linear; sem teste")
    for command in (review, visual):
        command.add_argument("--data-dir", type=Path, required=True)
        command.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
        command.add_argument("--splits-dir", type=Path, default=Path("artifacts/splits"))
        command.add_argument("--baseline-dir", type=Path, default=Path("artifacts/experiments/motion-full"))
    review.add_argument("--output-dir", type=Path, default=Path("artifacts/analyses/motion-review"))
    visual.add_argument("--config", type=Path, default=Path("config/visual.json"))
    visual.add_argument("--cache-dir", type=Path, default=Path("artifacts/visual-cache"))
    visual.add_argument("--weights-dir", type=Path, default=Path("artifacts/pretrained"))
    visual.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/visual-full"))
    visual.add_argument("--workers", type=int, default=4)
    visual.add_argument("--sample-per-class", type=int)
    visual.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    temporal = commands.add_parser("temporal-experiment", help="Controle MLP/GRU com três seeds e ablação de ordem")
    temporal.add_argument("--data-dir", type=Path, required=True)
    temporal.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.csv"))
    temporal.add_argument("--splits-dir", type=Path, default=Path("artifacts/splits"))
    temporal.add_argument("--visual-dir", type=Path, default=Path("artifacts/experiments/visual-full"))
    temporal.add_argument("--motion-dir", type=Path, default=Path("artifacts/experiments/motion-full"))
    temporal.add_argument("--config", type=Path, default=Path("config/temporal.json"))
    temporal.add_argument("--cache-dir", type=Path, default=Path("artifacts/visual-cache"))
    temporal.add_argument("--repair-cache-dir", type=Path, default=Path("artifacts/temporal-repair-cache"))
    temporal.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/temporal-controlled"))
    temporal.add_argument("--sample-per-class", type=int)
    temporal.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    final_protocol = commands.add_parser("final-protocol", help="Registra decisões finais antes de ler o teste")
    final_protocol.add_argument("--config", type=Path, default=Path("config/final_test.json"))
    final_protocol.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/final-test"))
    for name, help_text in (("final-evaluate", "Inferência no teste com os nove checkpoints registrados; sem treinamento"),
                            ("final-replay", "Recarrega e verifica as previsões finais sem treinar")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/final-test"))
    annotate = commands.add_parser("annotate-video", help="Exporta cópia local anotada com a GRU operacional seed 42")
    annotate.add_argument("--video", type=Path, required=True)
    annotate.add_argument("--output-dir", type=Path, required=True)
    annotate.add_argument("--mode", choices=["sequential", "clip"], default="sequential",
                          help="sequential: janelas de 2 s sem antecipação visual; clip: análise offline do clipe completo de até 2 s")
    demo = commands.add_parser("demo-videos", help="Exporta três clipes de validação por classe e galeria HTML local")
    demo.add_argument("--data-dir", type=Path, required=True)
    demo.add_argument("--splits-dir", type=Path, default=Path("artifacts/splits"))
    demo.add_argument("--output-dir", type=Path, default=Path("artifacts/demos/validation-mvp"))
    demo.add_argument("--seed", type=int, default=42)
    for command in (annotate, demo):
        command.add_argument("--protocol-dir", type=Path, default=Path("artifacts/experiments/final-test"))
        command.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    prepare_fish = commands.add_parser("prepare-fish-detector", help="Obtém pesos Fishial oficiais e confere SHA-256, sem treinamento")
    prepare_fish.add_argument("--config", type=Path, default=Path("config/simple_video.json"))
    dynamic = commands.add_parser("annotate-dynamic", help="GRU existente + detector de peixes + rastreamento e movimento da imagem")
    dynamic.add_argument("--video", type=Path, required=True)
    dynamic.add_argument("--output-dir", type=Path, required=True)
    dynamic.add_argument("--mode", choices=["sequential", "clip"], default="sequential")
    dynamic_demo = commands.add_parser("dynamic-demo", help="Montagem identificada de 12 clipes reais de validação, com estado reiniciado")
    dynamic_demo.add_argument("--data-dir", type=Path, required=True)
    dynamic_demo.add_argument("--splits-dir", type=Path, default=Path("artifacts/splits"))
    dynamic_demo.add_argument("--output-dir", type=Path, default=Path("artifacts/demos/dynamic-real"))
    dynamic_demo.add_argument("--seed", type=int, choices=[42], default=42, help="Reutiliza a seleção controlada anterior")
    for command in (dynamic, dynamic_demo):
        command.add_argument("--config", type=Path, default=Path("config/dynamic.json"))
        command.add_argument("--protocol-dir", type=Path, default=Path("artifacts/experiments/final-test"))
        command.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        command.add_argument("--motion-only", action="store_true", help="Desliga o detector explicitamente; não cria caixas")
    simple=commands.add_parser('annotate-simple',help='Vídeo anotado com rastreamento e resumo por intervalo')
    simple.add_argument('--video',type=Path,required=True)
    simple.add_argument('--output-dir',type=Path,required=True)
    simple.add_argument('--config',type=Path,default=Path('config/simple_video.json'))
    simple.add_argument('--protocol-dir',type=Path,default=Path('artifacts/experiments/final-test'))
    simple.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    simple.add_argument('--mode',choices=['sequential','clip'],default='sequential')
    simple.add_argument('--diagnostic',action='store_true',help='Exibe recorte e movimento da imagem; padrão desativado')
    simple.add_argument('--motion-only',action='store_true',help='Detector desligado explicitamente')
    simple.add_argument('--source-record',type=Path,help='Registro opcional de fonte/recorte contínuo e seu SHA-256')
    simple.add_argument('--show-speed',action='store_true',help='Exibe pixels/s apenas para histórico confirmado; padrão desligado')
    commands.add_parser('prepare-example',help='Prepara e verifica o exemplo oficial contínuo de 12 segundos')
    serve=commands.add_parser('serve',help='Abre a ferramenta local Gradio no navegador')
    serve.add_argument('--port',type=int,default=7860)
    serve.add_argument('--no-browser',action='store_true')
    serve.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    return app


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    try:
        # Para saídas derivadas do manifest, reaproveita a raiz registrada na indexação.
        if hasattr(args, "manifest"):
            index_report = args.manifest.parent / "index_report.json"
            if index_report.exists():
                root = Path(json.loads(index_report.read_text(encoding="utf-8"))["data_dir"]).resolve()
                protect_originals(root, args.output_dir)
                if hasattr(args, "data_dir") and data_root(args.data_dir) != root:
                    raise ValueError("--data-dir difere da raiz registrada no manifest; indexe a nova raiz primeiro")
        if args.command == "diagnose":
            from .diagnostics import diagnose
            report = diagnose(args.output)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return int(bool(report["import_errors"]) or bool(report["cuda"].get("error")))
        if args.command == "index":
            frame, issues = index_dataset(args.data_dir, args.output_dir)
            print(f"Vídeos indexados: {len(frame)}; problemas: {len(issues)}. Manifest: {args.output_dir / 'manifest.csv'}")
            return int(bool(issues) or frame.empty)
        if args.command == "split":
            report = make_splits(args.manifest, args.config, args.output_dir)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.command == "audit":
            from .video import audit_dataset
            report = audit_dataset(args.data_dir, args.manifest, args.output_dir, args.sample_size,
                                   args.seed, args.full, args.frames_per_video)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return int(report["failed"] > 0)
        elif args.command == "summary":
            from .reporting import summarize
            report = summarize(args.manifest, args.audit_dir, args.splits_dir, args.output_dir)
            print(f"Resumo: {args.output_dir / 'report.md'}; {report['videos']} vídeos; divisões validadas.")
        elif args.command == "examples":
            from .video import generate_examples
            report = generate_examples(args.data_dir, args.manifest, args.audit_dir, args.output_dir, args.seed)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return int(bool(report["issues"]))
        elif args.command == "motion-experiment":
            from .experiment import run_experiment
            run_experiment(args.data_dir, args.manifest, args.splits_dir, args.config,
                           args.cache_dir, args.output_dir, args.workers, args.sample_per_class)
        elif args.command == "motion-review":
            from .motion_review import review_motion
            review_motion(args.data_dir, args.manifest, args.splits_dir, args.baseline_dir, args.output_dir)
        elif args.command == "visual-experiment":
            from .visual_experiment import run_visual
            run_visual(args.data_dir, args.manifest, args.splits_dir, args.baseline_dir, args.config,
                       args.cache_dir, args.weights_dir, args.output_dir, args.workers, args.sample_per_class, args.device)
        elif args.command == "temporal-experiment":
            from .temporal_experiment import run_temporal
            run_temporal(args.data_dir, args.manifest, args.splits_dir, args.visual_dir, args.motion_dir, args.config,
                         args.cache_dir, args.repair_cache_dir, args.output_dir, args.sample_per_class, args.device)
        elif args.command == "final-protocol":
            from .final_protocol import freeze_protocol
            print(json.dumps(freeze_protocol(args.config, args.output_dir), ensure_ascii=False, indent=2))
        elif args.command == "final-evaluate":
            from .final_evaluation import run_final
            run_final(args.output_dir)
        elif args.command == "final-replay":
            from .final_evaluation import replay_final
            print(json.dumps(replay_final(args.output_dir), ensure_ascii=False, indent=2))
        elif args.command == "annotate-video":
            from .video_inference import OperationalPredictor
            from .video_export import export_video
            result = export_video(args.video, args.output_dir, OperationalPredictor(args.protocol_dir, args.device), args.mode)
            print(json.dumps({key: result.get(key) for key in ("status", "output_video", "processing_seconds", "error")},
                             ensure_ascii=False, indent=2))
            return int(result["status"] == "unavailable" or not any(item["status"] == "ok" for item in result.get("windows", [])))
        elif args.command == "demo-videos":
            from .video_demo import generate_demo
            result = generate_demo(args.data_dir, args.splits_dir, args.output_dir, args.seed, args.device, args.protocol_dir)
            print(f"Galeria: {args.output_dir / 'index.html'}; exportados {result['exported']}/12; {result['processing_seconds_including_model_load']:.2f}s")
            return int(result["exported"] != 12 or result["compatible_with_existing_pipeline"] != 12)
        elif args.command == "prepare-fish-detector":
            from .fish_detector import prepare_detector
            print(json.dumps(prepare_detector(args.config), ensure_ascii=False, indent=2))
        elif args.command == "annotate-dynamic":
            from .video_inference import OperationalPredictor
            from .dynamic_video import export_dynamic, load_dynamic
            predictor = OperationalPredictor(args.protocol_dir, args.device)
            config, detector = load_dynamic(args.config, predictor, args.motion_only)
            result = export_dynamic(args.video,args.output_dir,predictor,config,detector,args.mode)
            print(json.dumps({key: result.get(key) for key in ("status","output_video","processing_seconds","effective_processing_fps","error")},ensure_ascii=False,indent=2))
            return int(result["status"] == "unavailable" or not any(w["status"] == "ok" for w in result.get("windows",[])))
        elif args.command == "dynamic-demo":
            from .dynamic_demo import generate_dynamic_demo
            result = generate_dynamic_demo(args.data_dir,args.splits_dir,args.output_dir,args.config,
                                           args.seed,args.device,args.protocol_dir,args.motion_only)
            print(json.dumps({"video":result["montage"]["path"],"seconds":result["processing_seconds_including_model_load_montage_verification"],
                "effective_fps":result["effective_processing_fps_end_to_end"]},ensure_ascii=False,indent=2))
        elif args.command == 'annotate-simple':
            from .simple_video import run_simple
            tokens=argv if argv is not None else sys.argv[1:]
            import subprocess
            command='.\\.venv\\Scripts\\python.exe -m feedwatch '+subprocess.list2cmdline(tokens)
            result=run_simple(args.video,args.output_dir,args.config,args.protocol_dir,args.device,args.mode,
                args.diagnostic,args.motion_only,args.source_record,command,show_speed=args.show_speed)
            print(json.dumps({key:result.get(key) for key in ('status','output_video','presentation','total_processing_seconds','effective_input_fps','error')},ensure_ascii=False,indent=2))
            return int(result['status']=='unavailable' or not any(w['status']=='ok' for w in result.get('windows',[])))
        elif args.command == 'prepare-example':
            from .local_example import prepare_example
            print(prepare_example()[0])
        elif args.command == 'serve':
            from .local_app import launch_local
            launch_local(args.port,not args.no_browser,args.device)
    except (ValueError, OSError, KeyError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1
    return 0
