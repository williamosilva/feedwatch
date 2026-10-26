import copy
import unittest

from feedwatch.local_app import language_pair, local_text, localized_markdown, localized_status
from feedwatch.research_presentation import historical_table, result_sections, platform_i18n


class LocalI18nTest(unittest.TestCase):
    def test_local_default_labels_and_bilingual_output_escape_runtime_values(self):
        self.assertEqual(local_text('Analisar vídeo'),'Analyze video')
        status=localized_status('Análise concluída em {seconds} segundos. Veja o comparador no início da página.',seconds='<1>')
        self.assertIn('Analysis completed in &lt;1&gt; seconds.',status)
        self.assertIn('Análise concluída em &lt;1&gt; segundos.',status)
        self.assertNotIn('<1>',status)
        self.assertIn('data-fw-local-language="pt"',status)
        self.assertIn('hidden',status)
        html=localized_markdown('## Resultado da análise')
        self.assertIn('<h2>Analysis result</h2>',html)
        self.assertIn('<h2>Resultado da análise</h2>',html)

    def test_english_result_uses_saved_classes_and_preserves_failures_and_metadata(self):
        details={'windows':[{'status':'ok','predicted_class':'weak','start_seconds':0.,'end_seconds':2.},
                            {'status':'failed','predicted_class':None,'start_seconds':2.,'end_seconds':4.}],
                 'coverage':{'duration_seconds':4.,'class_distribution':{'none':0,'weak':1,'medium':0,'strong':0},
                             'conclusion':'Predominou a previsão fraca.','detections_per_frame_min':2,'detections_per_frame_max':5,
                             'frames_with_suspended_tracks':3,'frames_with_visual_errors':0,'invalid_windows':[{'status':'failed'}]}}
        before=copy.deepcopy(details)
        text=result_sections(details,'en')
        self.assertIn('**low in 1**',text)
        self.assertIn('1 incomplete or failed interval was left unclassified.',text)
        self.assertIn('do not establish that the fish stopped feeding',text)
        self.assertNotIn('not observed in 1',text)
        self.assertEqual(details,before)

    def test_historical_table_changes_display_locale_only(self):
        data={'counts':{'train':1200,'val':500,'test':300},'total':2000,
              'dates':{'train':['2022_6_19'],'val':['2022_6_21'],'test':['2022_6_22']},
              'accuracy':.5,'macro_f1':.4321}
        before=copy.deepcopy(data)
        english=historical_table(data,'en');portuguese=historical_table(data)
        self.assertIn('2,000 videos',english)
        self.assertIn('50.00%',english)
        self.assertIn('2022-06-22',english)
        self.assertIn('50,00%',portuguese)
        self.assertEqual(data,before)

    def test_native_gradio_upload_labels_have_both_languages(self):
        translations=platform_i18n().translations_dict
        self.assertEqual(translations['en']['common.clear'],'Clear')
        self.assertEqual(translations['pt']['common.clear'],'Limpar')
        self.assertEqual(translations['pt-BR'],translations['pt'])


if __name__=='__main__':unittest.main()
