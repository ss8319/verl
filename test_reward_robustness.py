import sys
import os
import importlib.util

# Function to import a module directly from a file path
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Path to the reward score directory
reward_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "verl/utils/reward_score"))
ssl4rl_path = os.path.join(reward_dir, "ssl4rl.py")
jigsaw_path = os.path.join(reward_dir, "jigsaw.py")

# Import the modules
ssl4rl = import_module_from_path("ssl4rl", ssl4rl_path)
jigsaw = import_module_from_path("jigsaw", jigsaw_path)

def test_ssl4rl_robustness():
    print("--- Testing ssl4rl reward robustness ---")
    gt = "1"
    
    test_cases = [
        {
            "name": "Perfect implementation",
            "pred": "assistant\n<think>The image shows a skin lesion rotated 90 degrees.</think> <answer>1</answer>",
            "expected_min": 0.99, # 0.9 (acc) + 0.1 (format)
        },
        {
            "name": "Correct answer, missing think tags",
            "pred": "assistant\nThe answer is <answer>1</answer>",
            "expected": 0.9 * 1.0 + 0.1 * 0.4, # 0.94
        },
        {
            "name": "Correct answer, missing all tags",
            "pred": "assistant\n1",
            "expected": 0.0, # extract_answer fails -> R_acc=0
        },
        {
            "name": "Wrong answer, perfect format",
            "pred": "assistant\n<think>reasoning</think> <answer>0</answer>",
            "expected": 0.1, # 0.9*0 + 0.1*1.0
        },
        {
            "name": "Correct answer, wrong tag order",
            "pred": "assistant\n<answer>1</answer> <think>reasoning</think>",
            "expected": 0.9 * 1.0 + 0.1 * 0.8, # 0.9 + 0.08 = 0.98 (no order bonus)
        },
        {
            "name": "Correct answer, empty think tags",
            "pred": "assistant\n<think></think> <answer>1</answer>",
            "expected": 0.9 * 1.0 + 0.1 * 0.4, # 0.94 (no think content bonus)
        },
        {
            "name": "Multiple comma-separated ground truths",
            "gt": "1, 2",
            "pred": "assistant\n<think>...</think> <answer>2</answer>",
            "expected_min": 0.99,
        }
    ]

    for tc in test_cases:
        actual = ssl4rl.compute_score(tc["pred"], tc.get("gt", gt))
        exp = tc.get("expected")
        if exp is not None:
            status = "PASS" if abs(actual - exp) < 1e-5 else f"FAIL (Got {actual:.3f}, Expected {exp:.3f})"
        else:
            status = "PASS" if actual >= tc["expected_min"] else f"FAIL (Got {actual:.3f}, Expected >= {tc['expected_min']})"
        print(f"[{status}] {tc['name']}")

def test_jigsaw_robustness():
    print("\n--- Testing jigsaw reward robustness ---")
    gt = "0,1,2"
    
    test_cases = [
        {
            "name": "Perfect Jigsaw sequence",
            "pred": "assistant\n<think>sorting...</think> <answer>0,1,2</answer>",
            "expected_min": 0.99,
        },
        {
            "name": "Partial Jigsaw (1/3 correct)",
            "pred": "assistant\n<think>sorting...</think> <answer>0,2,1</answer>",
            "expected": 0.9 * (1/3) + 0.1 * 1.0, # 0.3 + 0.1 = 0.4
        },
        {
            "name": "Partial Jigsaw, missing think",
            "pred": "assistant\n<answer>0,2,1</answer>",
            "expected": 0.9 * (1/3) + 0.1 * 0.4, # 0.3 + 0.04 = 0.34
        },
        {
            "name": "Empty answer sequence",
            "pred": "assistant\n<think>failed</think> <answer></answer>",
            "expected": 0.1 * 0.4, # 0.04 (only think tag exists)
        },
        {
            "name": "Wrong length sequence",
            "pred": "assistant\n<think>...</think> <answer>0,1</answer>",
            "expected": 0.9 * (2/3) + 0.1 * 1.0, # 0.6 + 0.1 = 0.7
        }
    ]

    for tc in test_cases:
        actual = jigsaw.compute_score(tc["pred"], tc.get("gt", gt))
        exp = tc.get("expected")
        if exp is not None:
            status = "PASS" if abs(actual - exp) < 1e-5 else f"FAIL (Got {actual:.3f}, Expected {exp:.3f})"
        else:
            status = "PASS" if actual >= tc["expected_min"] else f"FAIL (Got {actual:.3f}, Expected >= {tc['expected_min']})"
        print(f"[{status}] {tc['name']}")

if __name__ == "__main__":
    test_ssl4rl_robustness()
    test_jigsaw_robustness()
