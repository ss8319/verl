import re

def extract_answer(text):
    """Extract content between <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_assistant_response(text):
    """Extract everything after the last 'assistant\n' marker."""
    marker = "assistant\n"
    idx = text.rfind(marker)
    if idx != -1:
        return text[idx + len(marker):].strip()
    return text.strip()

def extract_format_score(text):
    """
    Check for <think>...</think> <answer>...</answer> format.
    Returns a score between 0.0 and 1.0.
    """
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    
    score = 0.0
    # 1. Think tags exist and are not empty (0.4)
    if think_match and think_match.group(1).strip():
        score += 0.4
    # 2. Answer tags exist and are not empty (0.4)
    if answer_match and answer_match.group(1).strip():
        score += 0.4
    # 3. Order is correct: think before answer (0.2)
    if think_match and answer_match and think_match.end() <= answer_match.start():
        score += 0.2
    return score

def compute_score(solution_str: str, ground_truth: str, format_weight: float = 0.1) -> dict:
    """
    Compute reward for DermoGPT MCQA.
    Combines format correctness and answer accuracy.
    """
    # Use everything if 'assistant\n' isn't found, otherwise just the assistant's part
    out_text = extract_assistant_response(solution_str)
    
    # 1. Format Reward (0.1 weight)
    format_reward = extract_format_score(out_text)
    
    # 2. Accuracy Reward (0.9 weight)
    acc_reward = 0.0
    pred_answer = extract_answer(out_text)
    
    # Simple string match for MCQA (A, B, C, etc.)
    if pred_answer and pred_answer.upper() == ground_truth.upper():
        acc_reward = 1.0
        
    score = (1.0 - format_weight) * acc_reward + format_weight * format_reward
    
    return {
        "score": score,
        "acc_reward": acc_reward,
        "format_reward": format_reward,
    }
