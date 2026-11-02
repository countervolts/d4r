"""Keep startup, feature instances and resolution changes out of shared medians."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('profile_summary', ROOT/'scripts/summarize_d4r_profile.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class ProfileSummaryTests(unittest.TestCase):
    def test_warm_groups_and_unavailable_timings(self):
        def sample(frame, render='853x480', reset=0, gpu='2.2'):
            return (f'D4R_PROFILE frame={frame} render={render} output=2560x1440 '
                    f'preset=11 quality=3 flags=0x43 input_wait=cpu reset={reset} '
                    f'vk_total=2.8 dlss_gpu={gpu} native_motion_gpu=-1 wire_repair_gpu=-1')
        create='NVSDK_NGX_D3D12_CreateFeature(feature=1)'
        log='\n'.join((create, sample(1), sample(300, reset=1), sample(600),
                       sample(900, gpu='nan'), sample(1200, render='1706x960'),
                       create, sample(300)))
        rows=summary.summarize(log)
        self.assertEqual(len(rows), 3)
        self.assertEqual([r['samples'] for r in rows], [2, 1, 1])
        self.assertEqual([r['feature_instance'] for r in rows], [1, 1, 2])
        self.assertEqual(rows[0]['timings_ms']['dlss_gpu']['n'], 1)
        self.assertIsNone(rows[0]['timings_ms']['wire_repair_gpu'])
        self.assertEqual(rows[0]['timings_ms']['vk_total']['median'], 2.8)

    def test_routing_changes_get_separate_groups(self):
        base = ('D4R_PROFILE frame=300 render=853x480 output=2560x1440 '
                'preset=11 quality=3 flags=0x43 input_wait=cpu reset=0 '
                'feature=29184 native_field=1 wire_requested=1 gpu_completion=0 ')
        rows = summary.summarize(base+'native_substitute=0\n'+base+'native_substitute=1')
        self.assertEqual(len(rows), 2)
        self.assertEqual([r['native_substitute'] for r in rows], ['0', '1'])

    def test_scene_window_and_actual_output_mode(self):
        base = ('D4R_PROFILE frame=600 render=853x480 output=2560x1440 '
                'preset=11 quality=3 flags=0x43 input_wait=cpu reset=0 ')
        log = ('[ 10.000] '+base+'vk_total=8 vk_output_mode=transfer\n'
               '[ 20.000] '+base+'vk_total=3 vk_output_mode=compute\n'
               '[ 21.000] '+base+'vk_total=4 vk_output_mode=transfer\n'
               '[ 30.000] '+base+'vk_total=9 vk_output_mode=compute')
        rows = summary.summarize(log, since_clock=19, until_clock=22)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r['vk_output_mode'] for r in rows], ['compute', 'transfer'])
        self.assertEqual([r['timings_ms']['vk_total']['median'] for r in rows], [3, 4])
