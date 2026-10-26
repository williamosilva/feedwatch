"""Arithmetic scenarios are supplied by the user and never inferred from a video."""
import json
import shutil
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which('node'),'Node.js is required for browser arithmetic checks')
class FeedingScenarioTest(unittest.TestCase):
    def evaluate(self, expression):
        source=Path('src/feedwatch/assets/portfolio.js').read_text(encoding='utf-8')
        code=source+'\nprocess.stdout.write(JSON.stringify('+expression+'));'
        result=subprocess.run(['node'],input=code,encoding='utf-8',capture_output=True,check=True)
        return json.loads(result.stdout)

    def test_reduction_shows_actual_before_after_quantities_and_costs(self):
        result=self.evaluate('calculateFeedingScenario(100,90,10)')
        self.assertEqual(result['difference_kg'],10)
        self.assertEqual(result['difference_percent'],10)
        self.assertEqual(result['usual_cost'],1000)
        self.assertEqual(result['proposed_cost'],900)
        self.assertEqual(result['difference_cost'],100)

    def test_increase_is_an_increase_not_positive_savings(self):
        result=self.evaluate('calculateFeedingScenario(100,120,10)')
        self.assertEqual(result['difference_kg'],-20)
        self.assertEqual(result['difference_percent'],-20)
        self.assertEqual(result['difference_cost'],-200)

    def test_price_is_optional_and_zero_start_has_no_percentage(self):
        result=self.evaluate('calculateFeedingScenario(0,10)')
        self.assertIsNone(result['difference_percent'])
        self.assertIsNone(result['usual_cost'])
        self.assertEqual(result['proposed_kg'],10)

    def test_zero_price_and_unchanged_quantity_are_valid(self):
        result=self.evaluate('calculateFeedingScenario(12.5,12.5,0)')
        self.assertEqual(result['difference_kg'],0)
        self.assertEqual(result['difference_cost'],0)
        self.assertEqual(result['proposed_cost'],0)

    def test_invalid_and_overflowing_values_are_rejected(self):
        outcomes=self.evaluate("[[-1,2,1],[1,-2,1],[1,2,-1],[NaN,2,1],[1,2,Infinity],[1e308,1,1e308],['10',2,1]].map(args=>{try{calculateFeedingScenario(...args);return false;}catch{return true;}})")
        self.assertTrue(all(outcomes))
