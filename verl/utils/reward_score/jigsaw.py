import re


def extract_answer(text):
    # 提取<answer>...</answer>之间的内容
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_assistant_response(text):
    """
    提取 assistant 的回复内容（即最后一个 'assistant\n' 之后的内容）。
    """
    # 查找最后一个 'assistant\n'
    marker = "assistant\n"
    idx = text.rfind(marker)
    if idx != -1:
        return text[idx + len(marker):].strip()
    else:
        return text.strip()  # 如果没有找到，返回原文


def extract_format_score(text):
    """
    Check if the response follows the correct format:
    <think> reasoning </think> <answer> answer </answer>
    
    Returns a partial score between 0 and 1.
    """
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    
    score = 0.0
    
    # 1. Think tags check (0.4)
    if think_match and think_match.group(1).strip():
        score += 0.4
        
    # 2. Answer tags check (0.4)
    if answer_match and answer_match.group(1).strip():
        score += 0.4
        
    # 3. Ordering check (0.2)
    # Awarded only if both exist and have content, and think ends before answer begins
    if think_match and think_match.group(1).strip() and answer_match and answer_match.group(1).strip():
        if think_match.end() <= answer_match.start():
            score += 0.2
            
    return score


def compute_score(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1) -> float:
    out_text = extract_assistant_response(predict_str)
    
    # 1. Calculate Accuracy Score (R_acc) - Weight 0.9
    acc_score = 0.0
    pred = extract_answer(out_text)
    
    if pred is not None:
        ground_truth_list = ground_truth.split(',')
        gt_length = len(ground_truth_list)
        try:
            pred_list = pred.split(',')
        except:
            pred_list = []
        
        pred_length = len(pred_list)
        correct_count = 0
        for i in range(gt_length):
            try:
                if i < pred_length and int(ground_truth_list[i]) == int(pred_list[i]):
                    correct_count += 1
            except:
                pass
        acc_score = correct_count / gt_length if gt_length > 0 else 0.0
        
    # 2. Calculate Format Score (R_format) - Weight 0.1
    format_reward = extract_format_score(out_text)
    
    # 3. Combined Score: 0.9 * R_acc + 0.1 * R_format
    total_score = 0.9 * acc_score + 0.1 * format_reward
    
    return total_score