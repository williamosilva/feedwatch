import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.utils.ops import scale_boxes

from feedwatch.conservative_tracking import ConservativeTracker
from feedwatch.dynamic_video import export_dynamic
from feedwatch.fish_detector import postprocess_boxes
from feedwatch.simple_overlay import SimpleOverlay
from feedwatch.video_conclusion import build_summary,render_card
from feedwatch.video_inference import display_state,plan_windows
from test_video_mvp import FakePredictor,make_video


CONFIG=json.loads(Path('config/simple_video.json').read_text(encoding='utf-8'))


def box(x=100.,score=.8):
    return {'xyxy':[x,100.,x+40,120.],'score':score,'detector_class_id':0,'detector_class':'Fish'}


class Predictor(FakePredictor):
    device=torch.device('cpu')


class SimpleVideoTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);cv2.setNumThreads(1)

    def test_sdk_letterbox_coordinates_are_restored_once_landscape_and_portrait(self):
        for original,target,padding in [((1080,1920),(736,1280),(0,8)),((1920,1080),(1280,736),(8,0))]:
            source=torch.tensor([[100.,200.,400.,500.]])
            encoded=source*(2/3)+torch.tensor([*padding,*padding])
            restored=scale_boxes(target,encoded,original)
            detections,_=postprocess_boxes(np.c_[restored.numpy(),[[.8,0]]],original[1],original[0],CONFIG['detector'])
            np.testing.assert_allclose(detections[0]['xyxy'],source.numpy()[0],atol=.001)

    def test_explicit_nms_removes_duplicate_without_location_mask(self):
        values=np.array([[100,100,140,120,.8,0],[101,100,141,120,.7,0],
                         [900,500,1000,600,.9,0]],dtype=float)
        retained,removed=postprocess_boxes(values,1920,1080,CONFIG['detector'])
        self.assertEqual(len(retained),2);self.assertEqual(removed,[1])
        self.assertIn([900.,500.,1000.,600.],[d['xyxy'] for d in retained])
        with self.assertRaises(ValueError):postprocess_boxes(np.array([[1,1,4,4,.8,.5]]),1920,1080,CONFIG['detector'])

    def test_speed_requires_consecutive_history_and_is_net_displacement(self):
        tracker=ConservativeTracker(CONFIG['tracker'])
        first=tracker.update([box()],0.,0)[0]
        self.assertFalse(first['track_confirmed']);self.assertIsNone(first['speed_source_px_per_second'])
        for index in range(1,7):
            observation=tracker.update([box(100+5*index)],index*.04,index)[0]
            self.assertEqual(observation['track_id'],1)
            if index<5:self.assertIsNone(observation['speed_source_px_per_second'])
        self.assertTrue(observation['track_confirmed'])
        self.assertAlmostEqual(observation['speed_source_px_per_second'],125.)
        self.assertFalse(observation['position_predicted_only'])

    def test_jump_and_ambiguity_do_not_create_connections(self):
        tracker=ConservativeTracker(CONFIG['tracker'])
        tracker.update([box()],0.,0)
        jumped=tracker.update([box(160)],.04,1)[0]
        self.assertEqual(jumped['track_id'],2)
        self.assertEqual(jumped['trail_txy'],[]);self.assertIsNone(jumped['speed_source_px_per_second'])
        tracker.reset();tracker.update([box(100),box(130)],0.,0)
        ambiguous=tracker.update([box(114),box(116)],.04,1)
        self.assertTrue(all(d['track_id']>2 and not d['track_confirmed'] for d in ambiguous))

    def test_loss_suspends_track_and_history_restarts_on_reacquisition(self):
        tracker=ConservativeTracker(CONFIG['tracker'])
        for index in range(7):tracker.update([box(100+index)],index*.04,index)
        self.assertEqual(tracker.update([],.28,7),[])
        self.assertIn(1,tracker.suspended_ids)
        recovered=tracker.update([box(108)],.32,8)[0]
        self.assertEqual(recovered['track_id'],1)
        self.assertFalse(recovered['track_confirmed'])
        self.assertEqual(recovered['trail_txy'],[])
        self.assertIsNone(recovered['speed_source_px_per_second'])

    def test_summary_uses_valid_windows_only_and_reports_tie(self):
        windows=plan_windows({'fps':25.,'frame_count':130,'duration_seconds':5.2},'sequential')
        windows[0].update(status='ok',predicted_class='medium')
        windows[1].update(status='ok',predicted_class='strong')
        records=[{'frame_index':i,'detector_status':'ok','detected_count':i%3,'confirmed_track_count':int(i>4),
            'error':'','detections':[],'suspended_track_ids':[]} for i in range(130)]
        analysis={'metadata':{'fps':25.,'duration_seconds':5.2},'windows':windows,'mode':'sequential'}
        summary=build_summary(analysis,records)
        self.assertTrue(summary['tie']);self.assertEqual(summary['complete_valid_windows'],2)
        self.assertEqual(summary['class_distribution'],{'none':0,'weak':0,'medium':1,'strong':1})
        self.assertEqual(summary['valid_analyzed_seconds'],4.)
        self.assertEqual(summary['frames_without_confirmed_tracking'],5)
        self.assertEqual(summary['detections_per_frame_min'],0)
        self.assertEqual(render_card(summary,1920,1310).shape,(1310,1920,3))
        for window in windows:window['status']='failed'
        unavailable=build_summary(analysis,records)
        self.assertEqual(unavailable['predominant_classes'],[])
        self.assertEqual(sum(unavailable['class_distribution'].values()),0)

    def test_clean_overlay_preserves_whole_image_and_appends_footer(self):
        image=np.full((360,640,3),70,np.uint8)
        renderer=SimpleOverlay(640,360,Predictor(),CONFIG)
        window=plan_windows({'fps':25.,'frame_count':50,'duration_seconds':2.},'sequential')
        rendered=renderer.render(image,display_state(window,'sequential',0.),'sequential',[],
            'disabled',{'status':'disabled'},None,0.)
        np.testing.assert_array_equal(rendered[:360],image)
        self.assertGreater(rendered.shape[0],360)
        self.assertFalse(renderer.last_layout['source_image_scaled'])
        self.assertLess(renderer.last_layout['footer_text_bottom'],rendered.shape[0]-2)

    def test_simple_export_keeps_causality_tail_and_missing_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'technical-fixture.mp4';make_video(source,130)
            report=export_dynamic(source,root/'out',Predictor(),CONFIG,None,'sequential',simple=True)
            self.assertIsNotNone(report['output_video'])
            self.assertEqual(report['export_verification']['decoded_frames'],130)
            rows=[json.loads(line) for line in (root/'out'/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(all(r['food_class'] is None for r in rows[:50]))
            self.assertEqual(rows[50]['food_window_id'],0)
            self.assertEqual(rows[100]['food_window_id'],1)
            self.assertEqual(report['windows'][-1]['status'],'incomplete')
            self.assertTrue(all(r['detected_count'] is None and r['motion_status']=='disabled' for r in rows))


if __name__=='__main__':unittest.main()
