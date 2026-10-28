"""Motion charts expose observed history without future values or repaired gaps."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which('node'), 'Node.js is required for motion chart checks')
class MotionDashboardUiTest(unittest.TestCase):
    def evaluate(self, expression):
        source = Path('src/feedwatch/assets/portfolio.js').read_text(encoding='utf-8')
        code = source + '\nprocess.stdout.write(JSON.stringify(' + expression + '));'
        result = subprocess.run(['node'], input=code, encoding='utf-8', capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_latest_sample_keeps_initialization_and_does_not_select_future(self):
        result = self.evaluate("(() => {const frames=[{timestamp_seconds:0,status:'initializing',mean_source_px_per_second:null},{timestamp_seconds:1,status:'ok',mean_source_px_per_second:30},{timestamp_seconds:2,status:'ok',mean_source_px_per_second:900}];return [-1,0,.9,1,1.999].map(t=>motionFrameAt(frames,t));})()")
        self.assertIsNone(result[0])
        self.assertEqual(result[1]['status'], 'initializing')
        self.assertIsNone(result[2]['mean_source_px_per_second'])
        self.assertEqual(result[3]['mean_source_px_per_second'], 30)
        self.assertEqual(result[4]['mean_source_px_per_second'], 30)

    def test_future_values_do_not_change_curve_or_its_scale(self):
        result = self.evaluate("motionChartData([{timestamp_seconds:0,status:'initializing',mean_source_px_per_second:null},{timestamp_seconds:1,status:'ok',mean_source_px_per_second:30},{timestamp_seconds:2,status:'ok',mean_source_px_per_second:9000}],1.5,2)")
        self.assertEqual(result['maximum'], 400)
        self.assertEqual(result['cursor'], 225)
        self.assertTrue(result['path'].startswith('M150.00,'))
        self.assertNotIn('300.00', result['path'])

    def test_missing_observation_breaks_line_and_zero_is_a_real_measurement(self):
        result = self.evaluate("motionChartData([{timestamp_seconds:0,status:'ok',mean_source_px_per_second:0},{timestamp_seconds:1,status:'ok',mean_source_px_per_second:10},{timestamp_seconds:2,status:'unavailable',mean_source_px_per_second:null},{timestamp_seconds:3,status:'ok',mean_source_px_per_second:20}],3,3)")
        self.assertTrue(result['path'].startswith('M0.00,66.00L100.00,'))
        self.assertEqual(result['path'].count('M'), 2)
        self.assertIn('M300.00,', result['path'])

    def test_observed_motion_above_map_ceiling_is_not_clipped(self):
        result = self.evaluate("motionChartData([{timestamp_seconds:0,status:'ok',mean_source_px_per_second:488}],0,2)")
        self.assertGreaterEqual(result['maximum'], 488)
        self.assertLess(float(result['path'].split(',')[1]), 66)
        self.assertGreaterEqual(float(result['path'].split(',')[1]), 4)


if __name__ == '__main__':
    unittest.main()
