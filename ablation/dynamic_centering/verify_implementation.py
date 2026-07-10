import argparse
import json
import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.variant_config import VALID_VARIANTS, variant_tie_word_embeddings


def build_model(variant):
    config = MiniMindConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        max_position_embeddings=64,
        vocab_size=23,
        tie_word_embeddings=True,
        lm_head_bias=True,
        embedding_variant=variant,
    )
    return MiniMindForCausalLM(config)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def run_checks():
    seed = 20260710
    torch.manual_seed(seed)
    s1 = build_model("s1")
    torch.manual_seed(seed)
    centered = build_model("center_dynamic")

    s1_state = s1.state_dict()
    centered_state = centered.state_dict()
    state_keys_equal = tuple(s1_state) == tuple(centered_state)
    initialization_equal = state_keys_equal and all(
        torch.equal(s1_state[name], centered_state[name]) for name in s1_state
    )
    parameter_count_s1 = count_parameters(s1)
    parameter_count_center_dynamic = count_parameters(centered)

    input_ids = torch.tensor([[0, 1, 1], [2, 3, 4]], dtype=torch.long)
    expected_mean = centered.model.embed_tokens.weight.float().mean(dim=0)
    expected_embeddings = centered.model.embed_tokens(input_ids) - expected_mean
    actual_embeddings = centered.model._embed(input_ids)
    embedding_formula_equal = torch.equal(actual_embeddings, expected_embeddings)

    hidden_states = torch.randn(2, 3, centered.config.hidden_size)
    logits_path_unchanged = torch.equal(
        centered._compute_logits(hidden_states),
        centered.lm_head(hidden_states),
    )

    centered.zero_grad(set_to_none=True)
    centered.model._embed(input_ids).sum().backward()
    gradient = centered.model.embed_tokens.weight.grad
    unused_row = centered.config.vocab_size - 1
    expected_unused_gradient = torch.full_like(
        gradient[unused_row],
        -input_ids.numel() / centered.config.vocab_size,
    )
    unused_row_receives_mean_gradient = torch.allclose(
        gradient[unused_row], expected_unused_gradient, atol=1e-7, rtol=0.0
    )
    all_vocab_rows_receive_gradient = bool(gradient.abs().sum(dim=1).gt(0).all().item())

    with torch.no_grad():
        before = centered.model._embed(input_ids).clone()
        centered.model.embed_tokens.weight[unused_row].add_(1.0)
        after = centered.model._embed(input_ids)
    mean_is_recomputed_each_forward = not torch.equal(before, after)

    checks = {
        "variant_registered": "center_dynamic" in VALID_VARIANTS,
        "tie_word_embeddings": variant_tie_word_embeddings("center_dynamic"),
        "shared_weight_storage": (
            centered.model.embed_tokens.weight.data_ptr() == centered.lm_head.weight.data_ptr()
        ),
        "state_keys_equal_to_s1": state_keys_equal,
        "initialization_equal_to_s1": initialization_equal,
        "parameter_count_equal_to_s1": parameter_count_s1 == parameter_count_center_dynamic,
        "embedding_formula_equal": embedding_formula_equal,
        "logits_path_unchanged": logits_path_unchanged,
        "unused_row_receives_mean_gradient": unused_row_receives_mean_gradient,
        "all_vocab_rows_receive_gradient": all_vocab_rows_receive_gradient,
        "mean_is_recomputed_each_forward": mean_is_recomputed_each_forward,
    }
    return {
        "passed": all(bool(value) for value in checks.values()),
        "checks": checks,
        "test_parameter_count_s1": parameter_count_s1,
        "test_parameter_count_center_dynamic": parameter_count_center_dynamic,
        "test_vocab_size": centered.config.vocab_size,
        "test_input_token_count": input_ids.numel(),
    }


def main():
    parser = argparse.ArgumentParser(description="验证 dynamic centering 的实现约束。")
    parser.add_argument("--output", default="", help="可选 JSON 输出路径")
    args = parser.parse_args()

    report = run_checks()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
