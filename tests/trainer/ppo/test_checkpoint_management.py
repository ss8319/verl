import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

# We'll define a MockTrainer that has the same logic as RayPPOTrainer
# to test the checkpoint management logic in isolation.

class MockTrainer:
    def __init__(self, config):
        self.config = config
        self.global_steps = 0
        self.rank = 0
        self._checkpoint_history = []
        self.use_critic = True

    def _manage_checkpoints(self, val_metrics=None):
        """Logic copied from RayPPOTrainer._manage_checkpoints"""
        if self.rank != 0:
            return

        # 1. Update metric for current step if available
        if val_metrics:
            metric_key = self.config.trainer.get('best_metric_key')
            if metric_key:
                current_metric = val_metrics.get(metric_key)
                if current_metric is None:
                    # Try suffix match
                    for k, v in val_metrics.items():
                        if k.endswith(metric_key):
                            current_metric = v
                            break

                if current_metric is not None:
                    # Find current step in history and update metric
                    found = False
                    for ckpt in self._checkpoint_history:
                        if ckpt['step'] == self.global_steps:
                            ckpt['metric'] = current_metric
                            found = True
                            break
                    if not found:
                        pass

        # 2. Determine which checkpoints to keep
        save_best_k = self.config.trainer.get('save_best_k', 0)
        max_latest_to_keep = self.config.trainer.get('max_actor_ckpt_to_keep', 1)
        if save_best_k <= 0:
            return

        mode = self.config.trainer.get('best_metric_mode', 'max')
        
        # Sort by step desc
        self._checkpoint_history.sort(key=lambda x: x['step'], reverse=True)
        
        latest_steps = [ckpt['step'] for ckpt in self._checkpoint_history[:max_latest_to_keep]]
        
        best_steps = []
        if save_best_k > 0:
            # Only consider checkpoints that HAVE a metric
            ckpts_with_metric = [ckpt for ckpt in self._checkpoint_history if ckpt['metric'] is not None]
            ckpts_with_metric.sort(key=lambda x: x['metric'], reverse=(mode == 'max'))
            best_steps = [ckpt['step'] for ckpt in ckpts_with_metric[:save_best_k]]
            
        keep_steps = set(latest_steps) | set(best_steps)
        
        # 3. Delete folders not in keep_steps
        new_history = []
        for ckpt in self._checkpoint_history:
            if ckpt['step'] in keep_steps:
                new_history.append(ckpt)
            else:
                if os.path.exists(ckpt['path']):
                    print(f"Removing checkpoint: {ckpt['path']} (step {ckpt['step']} is neither among latest {max_latest_to_keep} nor best {save_best_k})")
                    shutil.rmtree(ckpt['path'], ignore_errors=True)
        self._checkpoint_history = new_history

    def simulate_save(self, step, metric=None):
        self.global_steps = step
        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")
        os.makedirs(path, exist_ok=True)
        self._checkpoint_history.append({
            'step': step,
            'path': path,
            'metric': metric
        })
        self._manage_checkpoints()

class ConfigDictMock(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"No such attribute: {name}")

class ConfigMock:
    def __init__(self, trainer_dict):
        self.trainer = ConfigDictMock(trainer_dict)

class TestCheckpointManagement(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_checkpoints"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        self.trainer_config = {
            'default_local_dir': self.test_dir,
            'save_best_k': 2,
            'max_actor_ckpt_to_keep': 1,
            'best_metric_key': 'val/acc',
            'best_metric_mode': 'max'
        }
        self.config = ConfigMock(self.trainer_config)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_keep_latest_and_best(self):
        trainer = MockTrainer(self.config)
        
        # Step 10: acc 0.5 (Best so far, also latest)
        trainer.simulate_save(10, 0.5)
        self.assertEqual(len(trainer._checkpoint_history), 1)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "global_step_10")))
        
        # Step 20: acc 0.4 (Latest, but worse than step 10)
        # Should keep 20 (latest) and 10 (best)
        trainer.simulate_save(20, 0.4)
        self.assertEqual(len(trainer._checkpoint_history), 2)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "global_step_10")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "global_step_20")))
        
        # Step 30: acc 0.6 (Latest AND best)
        # History: [30 (latest, best1), 20 (prev latest), 10 (best2)]
        # Should keep 30, 10. 20 is neither latest nor among top 2 best.
        trainer.simulate_save(30, 0.6)
        self.assertEqual(len(trainer._checkpoint_history), 2)
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, "global_step_20")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "global_step_10")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "global_step_30")))
        
        # Step 40: acc 0.3 (Latest)
        # Best are 30 (0.6) and 10 (0.5).
        # History before manage: [40 (latest), 30 (best1), 10 (best2)]
        # Should keep all 3? No, keep_steps = {40} | {30, 10} = {40, 30, 10}
        trainer.simulate_save(40, 0.3)
        self.assertEqual(len(trainer._checkpoint_history), 3)
        
        # Step 50: acc 0.7 (Latest, best1)
        # History before manage: [50 (0.7), 40 (0.3), 30 (0.6), 10 (0.5)]
        # latest = {50}
        # best = {50, 30}
        # keep = {50, 30}
        # Should delete 40 and 10.
        trainer.simulate_save(50, 0.7)
        self.assertEqual(len(trainer._checkpoint_history), 2)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "global_step_50")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "global_step_30")))
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, "global_step_40")))
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, "global_step_10")))

    def test_min_mode(self):
        self.config.trainer['best_metric_mode'] = 'min'
        self.config.trainer['best_metric_key'] = 'val/loss'
        trainer = MockTrainer(self.config)
        
        # Step 10: loss 0.5
        trainer.simulate_save(10, 0.5)
        # Step 20: loss 0.3 (Best)
        trainer.simulate_save(20, 0.3)
        # Step 30: loss 0.4 (Latest, Best2)
        # keep {30} | {20, 30} = {30, 20}
        trainer.simulate_save(30, 0.4)
        
        self.assertEqual(len(trainer._checkpoint_history), 2)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "global_step_20")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "global_step_30")))
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, "global_step_10")))

    def test_suffix_matching(self):
        self.config.trainer['best_metric_key'] = 'reward/mean@1'
        trainer = MockTrainer(self.config)
        
        # Simulate validation with a complex key
        val_metrics = {'val-aux/RotationQA_dermogpt/reward/mean@1': 0.8}
        
        # Step 10: Check that it finds the metric using suffix
        trainer.simulate_save(10)
        trainer._manage_checkpoints(val_metrics)
        
        self.assertEqual(trainer._checkpoint_history[0]['metric'], 0.8)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "global_step_10")))

if __name__ == "__main__":
    unittest.main()
