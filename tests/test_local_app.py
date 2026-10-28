import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import cv2

from feedwatch.local_app import LocalService
from feedwatch.local_summary import cost_text, plain_summary, simulate_costs
from feedwatch.video_conclusion import build_summary
from feedwatch.video_inference import analyze_video
from test_video_mvp import FakePredictor, make_video


class LocalAppTest(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_arithmetic_is_user_supplied_and_optional(self):
        result=simulate_costs(100,10,1000,10)
        self.assertEqual(result['cost_per_kg'],10)
        self.assertEqual(result['planned_cost'],10000)
        self.assertEqual(result['hypothetical_difference_kg'],100)
        self.assertEqual(result['hypothetical_difference_cost'],1000)
        self.assertIn('R$ 1.000,00',cost_text(100,10,1000,10))
        for args in ((100,0,1000,10),(None,10,1000,10),(100,10,1000,101),(float('nan'),10,1000,10)):
            with self.assertRaises(ValueError):simulate_costs(*args)

    def test_bad_input_never_loads_models_or_produces_none(self):
        with tempfile.TemporaryDirectory() as folder:
            def forbidden(*args):raise AssertionError('Should not load models')
            service=LocalService(Path(folder)/'runs',runtime_factory=forbidden)
            source=Path(folder)/'invalid.mp4';source.write_bytes(b'not a video')
            short=Path(folder)/'incomplete-technical-test.mp4';make_video(short,frames=15)
            for path in (None,source,short):
                with self.assertRaises(ValueError):service.analyze(path)
            self.assertEqual(service.model_load_count,0)
            for failure in service.output_root.glob('*/failure.json'):
                self.assertIsNone(json.loads(failure.read_text())['predicted_class'])

    def test_progress_reports_actual_frames_and_preserves_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            video=Path(folder)/'technical-test-only.mp4';make_video(video,frames=101)
            events=[]
            with_progress=analyze_video(video,FakePredictor(),'sequential',lambda *args:events.append(args))
            original=analyze_video(video,FakePredictor(),'sequential')
            self.assertEqual(with_progress['windows'],original['windows'])
            self.assertEqual(events[0],('food',0,101))
            self.assertEqual(events[-1],('food',101,101))
            self.assertEqual([e[1] for e in events],list(range(102)))

    def test_serial_calls_share_one_runtime_and_create_distinct_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'technical-test-only.mp4';make_video(source,frames=50)
            config_path=Path('config/simple_video.json')
            from feedwatch.data import file_hash
            created=[]
            def factory(*args):
                predictor=FakePredictor();predictor.device=torch.device('cpu')
                predictor.identification={**predictor.identification,'runtime_device':'cpu','integer_map':{0:'none'}}
                runtime=SimpleNamespace(predictor=predictor,detector=None,
                    config=json.loads(config_path.read_text(encoding='utf-8')),
                    config_path=config_path.resolve(),config_sha256=file_hash(config_path))
                created.append(runtime)
                return runtime
            service=LocalService(root/'runs',runtime_factory=factory)
            class FalseyProgress:
                def __init__(self):self.calls=[]
                def __bool__(self):return False
                def __call__(self,*args,**kwargs):self.calls.append((args,kwargs))
            sink=FalseyProgress()
            with ThreadPoolExecutor(2) as pool:
                results=list(pool.map(lambda _:service.analyze(source,progress=sink),range(2)))
            self.assertEqual(len(created),1)
            self.assertTrue(sink.calls)
            self.assertEqual([r['details']['local_app']['runtime_reused'] for r in results],[False,True])
            self.assertNotEqual(results[0]['directory'],results[1]['directory'])
            for result in results:
                self.assertEqual(result['details']['local_app']['model_load_count'],1)
                self.assertFalse(result['details']['reload_verification']['checkpoint_reloaded'])
                self.assertTrue(Path(result['video']).exists())
                self.assertIn('**Medidas:**',result['summary'])
                self.assertIn('**Previsões:**',result['summary'])
                self.assertIn('**Interpretação:**',result['summary'])
                self.assertEqual(result['details']['food_model']['integer_map'],{'0':'none'})
                motion=result['details']['motion']
                self.assertTrue(Path(motion['video']).is_file())
                self.assertEqual(motion['frame_count'],50)
                self.assertEqual(len(motion['frames']),50)
                self.assertEqual(motion['frames'][0]['status'],'initializing')
                self.assertIsNone(motion['frames'][0]['mean_source_px_per_second'])
                progress=json.loads((Path(result['directory'])/'progress.json').read_text())
                render=[e['completed'] for e in progress if e['phase']=='render']
                self.assertEqual(render,list(range(51)))
                self.assertEqual([e['completed'] for e in progress if e['phase']=='motion'],list(range(51)))
            with patch('feedwatch.local_app.export_motion_presentation',side_effect=cv2.error('Map unavailable')):
                fallback=service.analyze(source)
            self.assertTrue(Path(fallback['video']).is_file())
            self.assertEqual(fallback['details']['windows'],results[0]['details']['windows'])
            self.assertEqual(fallback['details']['motion']['status'],'unavailable')
            self.assertEqual(service.model_load_count,1)

    def test_summary_keeps_failed_windows_unavailable_and_weak_is_not_no_feeding(self):
        analysis={'mode':'sequential','metadata':{'duration_seconds':6.,'fps':30},'rendered_frames':180,
            'windows':[{'window_id':i,'start_seconds':i*2.,'end_seconds':(i+1)*2.,
                        'status':'failed' if i==1 else 'ok','error':'decode' if i==1 else '',
                        'predicted_class':None if i==1 else 'weak'} for i in range(3)]}
        summary=build_summary(analysis,[])
        text=plain_summary(summary,analysis)
        self.assertEqual(summary['complete_valid_windows'],2)
        self.assertEqual(summary['class_distribution']['none'],0)
        self.assertIn('não provam que os peixes não se alimentam',text)
        self.assertIn('1 trecho(s) incompleto(s)/com falha',text)


if __name__=='__main__':unittest.main()
