import torch

def compute_accuracy(pad_outputs, pad_targets, ignore_label):
    """Calculate accuracy.

    Args:
        pad_outputs (LongTensor): Prediction tensors (B, Lmax).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_outputs.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return numerator.float() / denominator.float() #(FIX:MZY):return torch.Tensor type


def compute_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) between reference and hypothesis.
    
    Args:
        reference: Reference text (ground truth)
        hypothesis: Hypothesis text (prediction)
    
    Returns:
        float: WER value (0.0 - 1.0+)
    """
    try:
        from jiwer import wer
        return wer(reference, hypothesis)
    except ImportError:
        # Fallback to simple word-level edit distance if jiwer is not available
        import re
        ref_words = re.findall(r'\S+', reference.lower())
        hyp_words = re.findall(r'\S+', hypothesis.lower())
        
        # Simple Levenshtein distance on words
        if len(ref_words) == 0:
            return 1.0 if len(hyp_words) > 0 else 0.0
        
        # Dynamic programming for edit distance
        d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
        
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j
        
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(
                        d[i-1][j] + 1,      # deletion
                        d[i][j-1] + 1,      # insertion
                        d[i-1][j-1] + 1     # substitution
                    )
        
        return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def compute_wer_batch(references: list, hypotheses: list) -> dict:
    """Calculate WER metrics for a batch of references and hypotheses.
    
    Args:
        references: List of reference texts
        hypotheses: List of hypothesis texts
    
    Returns:
        dict: Dictionary containing WER statistics
    """
    if len(references) != len(hypotheses):
        raise ValueError(f"Length mismatch: {len(references)} references vs {len(hypotheses)} hypotheses")
    
    wers = []
    for ref, hyp in zip(references, hypotheses):
        wer_value = compute_wer(ref, hyp)
        wers.append(wer_value)
    
    avg_wer = sum(wers) / len(wers) if wers else 0.0
    
    return {
        "wer": avg_wer,
        "wer_list": wers,
        "num_samples": len(wers)
    }