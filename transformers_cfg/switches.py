from math import comb
import torch
from torch.nn import functional as F


class TooManyExpertsError(RuntimeError):
    pass


def switch_experts_top_k(
    EXPERTS,
    EXPERTS_PER_TOK,
    TOP_K,
    logits,
    allowed_tokens,
    gating_logits,
    experts_so_far,
):
    max_num = comb(EXPERTS, EXPERTS_PER_TOK)
    # Only switch experts if we haven't reached the maximum number of experts as there are no more experts to switch to
    if len(experts_so_far) >= max_num:
        raise TooManyExpertsError("Too many experts")
    probabilities = F.softmax(logits, dim=-1)
    # Check if one of k best tokens adheres to the grammar
    k_best = torch.topk(probabilities, TOP_K, dim=-1)
    if any([i in allowed_tokens for i in k_best.indices]):
        print(
            "Not switching experts as one of the top k tokens adheres to the grammar",
            list(filter(lambda x: x in allowed_tokens, k_best.indices)),
        )
        return False
    print("Switching experts as none of the top k tokens adheres to the grammar")
    return True


def switch_experts_top_p(
    EXPERTS,
    EXPERTS_PER_TOK,
    TOP_P,
    logits,
    allowed_tokens,
    gating_logits,
    experts_so_far,
):
    max_num = comb(EXPERTS, EXPERTS_PER_TOK)
    # Only switch experts if we haven't reached the maximum number of experts as there are no more experts to switch to
    if len(experts_so_far) >= max_num:
        raise TooManyExpertsError("Too many experts")
    probabilities = F.softmax(logits, dim=-1)
    # Check if one of p best tokens adheres to the grammar
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    for i in range(len(cumulative_probs)):
        if sorted_indices[i] in allowed_tokens:
            print(
                "Not switching experts as one of the top p tokens adheres to the grammar",
                sorted_indices[i],
            )
            print("At probability", sorted_probs[i], "cum:", cumulative_probs[i])
            return False
        if cumulative_probs[i] > TOP_P:
            break
    print("Switching experts as none of the top p tokens adheres to the grammar")
    return True


def switch_experts_top_k_experts(
    EXPERTS,
    EXPERTS_PER_TOK,
    TOP_K,
    logits,
    allowed_tokens,
    gating_logits,
    experts_so_far,
):
    # Only switch experts if we haven't reached the k experts
    if len(experts_so_far) >= TOP_K or len(experts_so_far) >= comb(
        EXPERTS, EXPERTS_PER_TOK
    ):
        raise TooManyExpertsError("Too many experts")
    print("Switching experts as we haven't reached the top k experts")
    return True


def switch_experts_top_p_experts(
    EXPERTS,
    EXPERTS_PER_TOK,
    TOP_P,
    logits,
    allowed_tokens,
    gating_logits,
    experts_so_far,
):
    # Only switch experts if we haven't reached top p experts
    combinations = torch.combinations(torch.arange(EXPERTS), EXPERTS_PER_TOK)

    combinations = combinations.to(gating_logits.device)
    combinations_mask = torch.zeros(
        combinations.shape[:-1], dtype=torch.bool, device=combinations.device
    )
    gating_combs = gating_logits[None, :].expand(combinations.shape[0], -1)
    gating_combs = gating_combs.gather(dim=-1, index=combinations).sum(dim=-1)
    # Normalize the probabilities
    probability_combs = F.softmax(gating_combs, dim=-1)
    for expert in experts_so_far:
        expert_comb = torch.tensor(expert, device=probability_combs.device)
        combinations_mask = combinations_mask | (combinations == expert_comb).all(
            dim=-1
        )
    probability_combs = probability_combs.masked_fill(~combinations_mask, 0)
    # Sum
    final_value = probability_combs.sum(dim=0)

    if final_value.item() >= TOP_P or len(experts_so_far) >= comb(
        EXPERTS, EXPERTS_PER_TOK
    ):
        raise TooManyExpertsError("Too many experts")
    print("Switching experts as we haven't reached the top p experts")
    return True
