import copy
import math
import unittest
from feedwatch.video_overview import summarize_video


def window(start,end,label,status='ok'):
    return {'start_seconds':start,'end_seconds':end,'predicted_class':label,'status':status}


class VideoOverviewTest(unittest.TestCase):
    def test_duration_weighting_is_not_a_vote_per_window_and_keeps_inputs(self):
        windows=[window(0,1,'weak'),window(1,2,'weak'),window(2,6,'medium')]
        original=copy.deepcopy(windows)
        result=summarize_video(windows,6)
        self.assertEqual(result['predominant_classes'],['medium'])
        self.assertAlmostEqual(result['fractions_by_class']['medium'],2/3)
        self.assertEqual(windows,original)

    def test_failed_incomplete_and_uncovered_time_are_not_none(self):
        result=summarize_video([window(0,2,'weak'),window(2,4,'none','failed'),window(4,5,'none','incomplete')],6)
        self.assertEqual(result['analyzed_seconds'],2)
        self.assertEqual(result['unclassified_seconds'],4)
        self.assertEqual(result['seconds_by_class']['none'],0)
        self.assertEqual(result['predominant_classes'],['weak'])
        self.assertFalse(result['summary_available'])
        self.assertEqual(result['summary_reason'],'insufficient_coverage')

    def test_whole_summary_requires_coverage_and_keeps_raw_duration_distribution(self):
        result=summarize_video([window(0,2,'weak')],60)
        self.assertFalse(result['summary_available'])
        self.assertEqual(result['predominant_classes'],['weak'])
        self.assertEqual(result['seconds_by_class']['weak'],2)
        result=summarize_video([window(0,48,'weak')],60)
        self.assertTrue(result['summary_available'])
        self.assertIsNone(result['summary_reason'])
        self.assertEqual(summarize_video([],60)['summary_reason'],'no_results')

    def test_ties_are_explicit_and_missing_results_have_no_winner(self):
        self.assertTrue(summarize_video([window(0,2,'weak'),window(2,4,'medium')],4)['tie'])
        result=summarize_video([window(0,2,None,'failed')],2)
        self.assertEqual(result['predominant_classes'],[])
        self.assertEqual(result['coverage_fraction'],0)
        self.assertIsNone(result['change_fraction'])

    def test_change_count_does_not_bridge_a_missing_period(self):
        result=summarize_video([window(0,2,'weak'),window(4,6,'medium'),window(6,8,'strong')],8)
        self.assertEqual(result['adjacent_pairs'],1)
        self.assertEqual(result['changes'],1)

    def test_zero_detections_are_real_observations_and_missing_counts_are_excluded(self):
        result=summarize_video([],4,[{'count':0},{'count':None},{'count':6}])
        self.assertEqual(result['detection_mean'],3)
        self.assertEqual(result['detection_min'],0)
        self.assertEqual(result['detection_missing_frames'],1)
        self.assertEqual(result['detection_zero_frames'],1)
        self.assertAlmostEqual(result['detection_coverage_fraction'],2/3)
        self.assertIsNone(summarize_video([],2)['detection_mean'])
        self.assertIsNone(summarize_video([],2)['detection_coverage_fraction'])

    def test_overlaps_bad_classes_and_invalid_durations_are_rejected(self):
        for windows,duration in (([window(0,3,'weak'),window(2,4,'medium')],4),
                                  ([window(0,2,'unknown')],2),([window(0,3,'weak')],2),([],math.nan),([],0)):
            with self.assertRaises(ValueError):summarize_video(windows,duration)

    def test_invalid_counts_cannot_be_mistaken_for_measurements(self):
        for count in (-1,math.nan,True,1.5,'2'):
            with self.assertRaises(ValueError):summarize_video([],2,[{'count':count}])

    def test_sixty_second_session_preserves_decline_hidden_by_global_predominance(self):
        windows=[window(t,t+2,'strong' if t<40 else 'weak') for t in range(0,60,2)]
        result=summarize_video(windows,60)
        self.assertEqual(result['predominant_classes'],['strong'])
        self.assertEqual([p['predominant_classes'] for p in result['evolution']['phases']],
                         [['strong'],['strong'],['weak']])
        self.assertEqual(result['evolution']['comparison']['direction'],'lower')

    def test_fifty_second_session_splits_boundary_windows_by_actual_duration(self):
        result=summarize_video([window(t,t+2,'weak' if t<18 else 'strong') for t in range(0,50,2)],50)
        phases=result['evolution']['phases']
        self.assertAlmostEqual(phases[0]['seconds_by_class']['weak'],50/3)
        self.assertAlmostEqual(phases[1]['seconds_by_class']['weak'],18-50/3)
        self.assertAlmostEqual(sum(p['analyzed_seconds'] for p in phases),50)
        self.assertEqual(result['evolution']['comparison']['direction'],'higher')

    def test_equal_endpoints_do_not_claim_the_entire_session_was_stable(self):
        windows=[window(t,t+2,'strong' if 20<=t<40 else 'weak') for t in range(0,60,2)]
        comparison=summarize_video(windows,60)['evolution']['comparison']
        self.assertEqual(comparison['direction'],'same')
        self.assertTrue(comparison['middle_differs'])

    def test_constant_session_can_compare_equal_endpoints(self):
        comparison=summarize_video([window(t,t+2,'weak') for t in range(0,60,2)],60)['evolution']['comparison']
        self.assertTrue(comparison['available'])
        self.assertEqual(comparison['direction'],'same')
        self.assertFalse(comparison['middle_differs'])

    def test_single_short_clip_cannot_become_three_independent_observations(self):
        evolution=summarize_video([window(0,2,'strong')],2)['evolution']
        self.assertFalse(evolution['comparison']['available'])
        self.assertEqual(evolution['comparison']['reason'],'short_video')
        self.assertIsNone(evolution['comparison']['direction'])

    def test_one_long_prediction_per_part_is_not_enough_for_comparison(self):
        evolution=summarize_video([window(0,20,'strong'),window(20,40,'medium'),window(40,60,'weak')],60)['evolution']
        self.assertFalse(evolution['comparison']['available'])
        self.assertEqual(evolution['comparison']['reason'],'too_few_windows')

    def test_failed_tail_does_not_create_a_decline(self):
        windows=[window(t,t+2,'strong') for t in range(0,50,2)]
        windows+=[window(50,60,'none','failed')]
        evolution=summarize_video(windows,60)['evolution']
        self.assertEqual(evolution['phases'][-1]['seconds_by_class']['none'],0)
        self.assertFalse(evolution['comparison']['available'])
        self.assertEqual(evolution['comparison']['reason'],'missing_results')

    def test_tied_or_minority_mode_does_not_support_direction(self):
        for labels in (['weak']*5+['strong']*5,['none']*4+['weak']*3+['medium']*3):
            windows=[window(2*i,2*i+2,label) for i,label in enumerate(labels+['medium']*20)]
            comparison=summarize_video(windows,60)['evolution']['comparison']
            self.assertFalse(comparison['available'])
            self.assertEqual(comparison['reason'],'ambiguous_predominance')

    def test_coverage_gate_accepts_eighty_percent_and_retains_missing_time(self):
        windows=[window(t,t+2,'medium') for t in range(0,60,2) if t%20<16]
        result=summarize_video(list(reversed(windows)),60)
        self.assertTrue(result['evolution']['comparison']['available'])
        self.assertAlmostEqual(result['evolution']['phases'][-1]['coverage_fraction'],.8)
        self.assertEqual(result['unclassified_seconds'],12)
