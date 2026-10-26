import copy
import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feedwatch.session_evaluation import (REVIEW_FIELDS, evaluate_session, import_reviews,
    read_review, review_grid, summarize_sessions, validate_session_set)
from feedwatch.session_evaluation import validate_prediction_record


def session(duration=60, **extra):
    return dict(session_id='session-1', camera_id='camera-1', recorded_date='2026-10-01',
                duration_seconds=duration, source_video_sha256='a' * 64,
                authorization='Owner authorization for study', continuity_note='Uncut original capture',
                source_url='https://example.org/recording', continuous=True,
                selection_locked_at_utc='2026-10-02T10:00:00+00:00', selection_basis='First eligible session',
                predictions_unseen_at_selection=True, split='final', purpose='evaluation', **extra)


def review(profile, duration=60, intervals=None, whole='weak', phases=('weak', 'weak', 'weak'), domain=False):
    rows = review_grid('session-1', duration)
    intervals = intervals or ['weak'] * math.ceil(duration / 2)
    labels = iter(intervals)
    for row in rows:
        row['label'] = next(labels) if row['kind'] == 'interval' else whole if row['kind'] == 'whole' else phases[('start', 'middle', 'end').index(row['kind'])]
    return dict(reviewer_id=profile, domain_expertise=domain, blinded=True, rows=rows)


def windows(duration=60, label='weak'):
    return [dict(start_seconds=t, end_seconds=min(t+2, duration), predicted_class=label, status='ok') for t in range(0, duration, 2)]


def inference_record():
    return dict(input_signature={'sha256': 'a' * 64}, mode='sequential',
                metadata=dict(fps=30, frame_count=1800, duration_seconds=60),
                food_model=dict(kind='gru', checkpoint_sha256='c' * 64,
                                class_mapping=dict(none=0, weak=1, medium=2, strong=3), seed=42),
                windows=[dict(w, window_id=i) for i, w in enumerate(windows())])


class SessionEvaluationTests(unittest.TestCase):
    def test_cli_copy_package_and_json_evaluation_use_same_source_without_touching_original(self):
        spec = importlib.util.spec_from_file_location('evaluate_sessions', Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate_sessions.py')
        script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)
        # Temporary software fixtures only: no actual biological references are created.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'synthetic-source.mp4'
            source.write_bytes(b'SOFTWARE FIXTURE: decode is mocked in this integration test')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            package = root / 'package'
            args = argparse.Namespace(output_dir=package, video=source, date='2026-10-01',
                session_id='session-1', camera_id='camera-1', source_url='local synthetic fixture',
                authorization='Synthetic software test', continuity_note='Synthetic software test',
                continuous=True, selection_basis='Synthetic software test', predictions_unseen=True,
                split='final', practice=False, copy_video=True)
            with patch.object(script, '_inspect_source', return_value=dict(duration_seconds=60, frame_count=1800, fps=30)), contextlib.redirect_stdout(io.StringIO()):
                script.prepare(args)
            manifest = json.loads((package / 'session.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['source_video'], 'source.mp4')
            self.assertEqual(source.read_bytes(), (package / 'source.mp4').read_bytes())
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
            self.assertIn('src="source.mp4"', (package / 'review.html').read_text(encoding='utf-8'))
            self.assertFalse(any(r['label'] for r in script.read_csv(package / 'review-a.csv')))
            for role, domain in (('a', True), ('b', False)):
                rows = [dict(row,reviewer_id=f'SOFTWARE_FIXTURE_{role}',domain_expertise=domain,blinded=True)
                        for row in review(role, domain=domain)['rows']]
                script._write_csv(package / f'review-{role}.csv', rows, REVIEW_FIELDS)
            record = inference_record(); record['input_signature']['sha256'] = digest
            script._write_json(package / 'analysis.json', record)
            output = root / 'report'
            with contextlib.redirect_stdout(io.StringIO()):
                script.evaluate(argparse.Namespace(manifest=[package/'session.json'],output_dir=output,allow_practice=False))
            report = json.loads((output/'results.json').read_text(encoding='utf-8'))
            self.assertEqual(report['summary']['overall']['whole']['accuracy'], 1)
            self.assertEqual(report['sessions'][0]['model_identification']['checkpoint_sha256'], 'c'*64)
            self.assertIn('predictions', report['provenance'][0]['input_sha256'])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
            record['input_signature']['sha256'] = 'b'*64
            script._write_json(package / 'analysis.json', record)
            with self.assertRaisesRegex(ValueError, 'source hash'):
                script.evaluate(argparse.Namespace(manifest=[package/'session.json'],output_dir=root/'wrong-source-report',allow_practice=False))
    def test_inference_json_is_bound_to_source_duration_complete_plan_and_model(self):
        record = inference_record()
        result, identity = validate_prediction_record(record, session())
        self.assertEqual(len(result), 30)
        self.assertEqual(identity['checkpoint_sha256'], 'c' * 64)
        self.assertEqual(identity['class_mapping']['medium'], 2)
        # Same-duration predictions copied from another recording must fail.
        bad = copy.deepcopy(record); bad['input_signature']['sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'source hash'):
            validate_prediction_record(bad, session())
        bad = copy.deepcopy(record); bad['metadata']['duration_seconds'] = 59
        with self.assertRaisesRegex(ValueError, 'duration'):
            validate_prediction_record(bad, session())
        bad = copy.deepcopy(record); bad['windows'].pop()
        with self.assertRaisesRegex(ValueError, 'complete'):
            validate_prediction_record(bad, session())
        bad = copy.deepcopy(record); bad['windows'][1]['window_id'] = 0
        with self.assertRaisesRegex(ValueError, 'window plan'):
            validate_prediction_record(bad, session())
        bad = copy.deepcopy(record); bad['food_model'].pop('checkpoint_sha256')
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            validate_prediction_record(bad, session())
        bad = copy.deepcopy(record); bad['food_model']['class_mapping']['medium'] = 1
        with self.assertRaisesRegex(ValueError, 'class mapping'):
            validate_prediction_record(bad, session())
    def test_agreement_is_duration_weighted_and_retains_disagreement_before_consensus(self):
        a = review('a', 51, ['weak'] * 26, domain=True)
        b = copy.deepcopy(a)
        b['reviewer_id'] = 'b'
        b['rows'][25]['label'] = 'medium'  # Last partial interval is one second.
        consensus = [{**b['rows'][25], 'notes': 'Joint review: low activity visible'}]
        consensus[0]['label'] = 'weak'
        rows, agreement = import_reviews(session(51), a, b, consensus)
        self.assertAlmostEqual(agreement['agreement_fraction'], 50/51)
        self.assertEqual(len(agreement['disagreements']), 1)
        self.assertEqual(agreement['disagreements'][0]['reviewer_b'], 'medium')
        self.assertEqual(rows[25]['label'], 'weak')

    def test_disagreement_requires_explicit_documented_consensus(self):
        a, b = review('a', domain=True), review('b')
        b['rows'][0]['label'] = 'strong'
        with self.assertRaisesRegex(ValueError, 'consensus required'):
            import_reviews(session(), a, b)
        with self.assertRaisesRegex(ValueError, 'reason'):
            import_reviews(session(), a, b, [{**b['rows'][0], 'notes': ''}])

    def test_consensus_cannot_override_agreement(self):
        a, b = review('a', domain=True), review('b')
        c = {**a['rows'][0], 'label': 'medium', 'notes': 'changed'}
        with self.assertRaisesRegex(ValueError, 'override'):
            import_reviews(session(), a, b, [c])

    def test_missing_outputs_are_errors_and_distribution_mass_not_renormalized(self):
        a, b = review('a', domain=True), review('b')
        p = windows()[:20]
        result = evaluate_session(session(), p, a, b)
        self.assertAlmostEqual(result['interval_metrics']['accuracy'], 2/3)
        self.assertEqual(result['interval_metrics']['missing_prediction_seconds'], 20)
        self.assertAlmostEqual(result['distribution']['total_variation_error'], 1/3)
        self.assertAlmostEqual(result['distribution']['prediction_fractions_including_missing']['missing'], 1/3)
        self.assertFalse(result['comparison']['correct'])
        self.assertFalse(result['comparison']['available'])
        self.assertIsNone(result['phases']['end']['prediction'])

    def test_unobservable_time_not_counted_as_none_or_model_error(self):
        labels = ['unobservable'] * 10 + ['weak'] * 20
        a, b = review('a', intervals=labels, domain=True), review('b', intervals=labels)
        result = evaluate_session(session(), windows()[10:], a, b)
        self.assertEqual(result['human_unobservable_seconds'], 20)
        self.assertEqual(result['interval_metrics']['observable_seconds'], 40)
        self.assertEqual(result['interval_metrics']['accuracy'], 1)
        self.assertEqual(result['model_coverage_of_observable_time'], 1)
        self.assertEqual(result['distribution']['total_variation_error'], 0)
        self.assertAlmostEqual(result['model_coverage_fraction'], 2/3)
        self.assertEqual(result['interval_metrics']['per_class']['none']['support_seconds'], 0)

    def test_completely_missing_outputs_do_not_count_as_correct_sessions(self):
        result = evaluate_session(session(), [], review('a', domain=True), review('b'))
        self.assertFalse(result['whole']['correct'])
        self.assertFalse(result['comparison']['correct'])
        self.assertEqual(result['interval_metrics']['accuracy'], 0)
        self.assertEqual(result['distribution']['total_variation_error'], 1)
        group = summarize_sessions([result])['overall']
        self.assertEqual(group['whole']['eligible_sessions'], 1)
        self.assertEqual(group['whole']['unavailable_sessions'], 1)
        self.assertEqual(group['whole']['accuracy'], 0)

    def test_global_agreement_cannot_be_credited_from_two_seconds_of_sixty(self):
        result = evaluate_session(session(), windows()[:1], review('a', domain=True), review('b'))
        self.assertEqual(result['whole']['raw_prediction'], 'weak')
        self.assertIsNone(result['whole']['prediction'])
        self.assertFalse(result['whole']['correct'])
        self.assertEqual(result['whole']['unavailable_reason'], 'insufficient_coverage')
        at_limit = evaluate_session(session(), windows()[:24], review('a', domain=True), review('b'))
        self.assertTrue(at_limit['whole']['correct'])

    def test_all_human_time_unobservable_has_no_accuracy_denominator(self):
        a = review('a', intervals=['unobservable']*30, whole='unobservable', phases=['unobservable']*3, domain=True)
        b = copy.deepcopy(a)
        b['reviewer_id'] = 'b'
        result = evaluate_session(session(), windows(), a, b)
        self.assertIsNone(result['interval_metrics']['accuracy'])
        self.assertIsNone(result['distribution']['total_variation_error'])
        self.assertFalse(result['whole']['eligible'])
        self.assertFalse(result['comparison']['eligible'])

    def test_human_holistic_reference_is_separate_from_interval_aggregation(self):
        a = review('a', whole='strong', phases=('strong', 'weak', 'weak'), domain=True)
        b = review('b', whole='strong', phases=('strong', 'weak', 'weak'))
        result = evaluate_session(session(), windows(), a, b)
        self.assertFalse(result['whole']['correct'])
        self.assertTrue(result['interval_derived_predominance']['correct'])
        self.assertEqual(result['comparison']['reference'], 'lower')
        self.assertEqual(result['comparison']['prediction'], 'same')
        self.assertFalse(result['comparison']['correct'])

    def test_two_seconds_do_not_become_independent_three_parts(self):
        a, b = review('a', 2, domain=True), review('b', 2)
        result = evaluate_session(session(2), windows(2), a, b)
        self.assertFalse(result['comparison']['available'])
        self.assertFalse(result['comparison']['correct'])

    def test_invalid_reviews_and_nonblind_duplicate_or_no_domain_reviewer_rejected(self):
        base = review('a', domain=True)
        variants = []
        b = review('a'); variants.append(b)
        b = review('b'); b['blinded'] = False; variants.append(b)
        b = review('b'); b['rows'].pop(); variants.append(b)
        b = review('b'); b['rows'][0]['label'] = ''; variants.append(b)
        b = review('b'); b['rows'][0]['label'] = 'ambiguous'; variants.append(b)
        b = review('b'); b['rows'].append(b['rows'][0]); variants.append(b)
        b = review('b'); b['rows'][0]['start_seconds'] = .1; variants.append(b)
        for bad in variants:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    import_reviews(session(), base, bad)
        with self.assertRaisesRegex(ValueError, 'domain'):
            import_reviews(session(), review('a'), review('b'))

    def test_failed_rows_still_checked_for_overlap_and_out_of_bounds(self):
        for p in ([dict(start_seconds=0,end_seconds=61,status='failed')],
                  [dict(start_seconds=0,end_seconds=4,status='failed'),windows()[1]],
                  [dict(start_seconds=0,end_seconds=2,status='unknown')]):
            with self.assertRaises(ValueError):
                evaluate_session(session(), p, review('a', domain=True), review('b'))

    def test_protocol_rejects_practice_and_split_leakage(self):
        validate_session_set([session()])
        bad = session(12); bad['purpose'] = 'practice'
        with self.assertRaisesRegex(ValueError, 'Practice'):
            validate_session_set([bad])
        validate_session_set([bad], allow_practice=True)
        with self.assertRaisesRegex(ValueError, 'separately'):
            validate_session_set([session(), bad], allow_practice=True)
        with self.assertRaisesRegex(ValueError, '50'):
            validate_session_set([session(12)])
        for field in ('camera_id', 'recorded_date', 'source_video_sha256'):
            other = session(); other.update(session_id='session-2', camera_id='camera-2', recorded_date='2026-10-03', source_video_sha256='b'*64, split='development')
            other[field] = session()[field]
            with self.assertRaisesRegex(ValueError, 'leakage'):
                validate_session_set([session(), other])

    def test_report_groups_sessions_by_camera_without_hiding_missing_output(self):
        result = evaluate_session(session(), windows(), review('a', domain=True), review('b'))
        second = copy.deepcopy(result); second['camera_id'] = 'camera-2'; second['session_id'] = 'session-2'
        report = summarize_sessions([result, second])
        self.assertEqual(report['overall']['session_count'], 2)
        self.assertEqual(report['by_camera']['camera-1']['session_count'], 1)
        self.assertEqual(report['by_split']['final']['whole']['accuracy'], 1)

    def test_csv_import_checks_reviewer_metadata_and_unicode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'review.csv'
            rows = review('Revisão A', domain=True)['rows']
            with path.open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, REVIEW_FIELDS)
                writer.writeheader()
                writer.writerows(dict(row, reviewer_id='Revisão A',domain_expertise='true',blinded='1') for row in rows)
            result = read_review(path)
            self.assertEqual(result['reviewer_id'], 'Revisão A')
            self.assertTrue(result['domain_expertise'])
            self.assertTrue(result['blinded'])

    def test_review_form_has_no_model_output_and_exports_real_csv_lines(self):
        spec = importlib.util.spec_from_file_location('evaluate_sessions', Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate_sessions.py')
        script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)
        s = session(12); s['purpose'] = 'practice'
        form = script._form(s, Path(__file__).resolve())
        self.assertIn('FORMULÁRIO DE PRÁTICA', form)
        self.assertNotIn('predicted_class', form)
        self.assertNotIn('scores', form)
        self.assertIn(".join('\\r\\n')", form)
        self.assertEqual(form.count('<select aria-label="Classificação'), 10) # 6 periods, 4 holistic rows.
        relative = script._form(s, Path('source.mp4'))
        self.assertIn('src="source.mp4"', relative)


if __name__ == '__main__':
    unittest.main()
