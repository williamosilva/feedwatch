"""Presentation checks: historical provenance and no fixed prediction in the UI summary."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from feedwatch.research_presentation import historical_data,historical_table,result_sections


class ResearchPresentationTest(unittest.TestCase):
    def test_historical_values_come_from_saved_aggregates_and_check_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'splits').mkdir()
            model=root/'experiments/final-test/models/gru_seed_42';model.mkdir(parents=True)
            composition={k:{'videos':n,'dates':[f'2022_6_{d}']} for k,n,d in [('train',12,19),('val',8,21),('test',4,22)]}
            (root/'splits/validation.json').write_text(json.dumps({'valid':True,'composition':composition}))
            (model/'identification.json').write_text(json.dumps({'family':'gru','seed':42,'id':'gru_seed_42'}))
            metrics=model/'metrics.json';metrics.write_text(json.dumps({'examples':4,'accuracy':.5,'macro_f1':.4321}))
            data=historical_data(root);html=historical_table(data)
            self.assertEqual(data['total'],24)
            self.assertIn('50,00%',html);self.assertIn('0,4321',html)
            self.assertNotIn('60,56%',html)
            self.assertIn('22/06/2022',html)
            metrics.write_text(json.dumps({'examples':5,'accuracy':.5,'macro_f1':.4321}))
            with self.assertRaises(ValueError):historical_data(root)

    def test_summary_uses_actual_classes_and_never_mutates_inputs_or_maps_failure_to_none(self):
        details={'windows':[{'status':'ok','predicted_class':'weak','start_seconds':0.,'end_seconds':2.},
                            {'status':'failed','predicted_class':None,'start_seconds':2.,'end_seconds':4.}],
                 'coverage':{'duration_seconds':4.,'class_distribution':{'none':0,'weak':1,'medium':0,'strong':0},
                             'conclusion':'Predominou a previsão fraca.','detections_per_frame_min':2,'detections_per_frame_max':5,
                             'frames_with_suspended_tracks':3,'frames_with_visual_errors':0,'invalid_windows':[{'status':'failed'}]}}
        before=copy.deepcopy(details)
        text=result_sections(details)
        self.assertEqual(details,before)
        self.assertIn('**fraca em 1**',text);self.assertNotIn('forte em',text)
        self.assertIn('não demonstram que os peixes deixaram de se alimentar',text)
        self.assertIn('ficaram sem classificação',text)
        for title in ('O que o modelo previu','O que isso significa','Limites desta análise'):self.assertIn(title,text)
        details['coverage']['class_distribution']['none']=1
        with self.assertRaises(ValueError):result_sections(details)


if __name__=='__main__':unittest.main()
