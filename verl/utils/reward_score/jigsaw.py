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


def compute_score(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1) -> float:
    out_text = extract_assistant_response(predict_str)
    pred = extract_answer(out_text)
    if pred is None:
        return 0
    else:
        score = 0
        ground_truth = ground_truth.split(',')
        gt_length = len(ground_truth)
        try:
            pred = pred.split(',')
        except:
            return 0.0
        
        pred_length = len(pred)
        for i in range(gt_length):
            try:
                if i < pred_length and int(ground_truth[i]) == int(pred[i]):
                    score += 1. / gt_length
            except:
                score += 0.
        return score