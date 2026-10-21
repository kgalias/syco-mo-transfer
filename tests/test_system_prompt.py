"""The optional `system_prompt` hook: default output unchanged, prepended when given, forwarded by ays.
Template tests use the real Qwen2.5 tokenizer (skipped if it cannot be fetched)."""

import pytest
from fixtures_v2 import FakeLM, mc_items

from sycomo.evals import _followups, ays
from sycomo.modeling import CHAT_KW, chat_text

MSGS = [{"role": "user", "content": "Q?\n(A) x\n(B) y"}]


@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                            revision="7ae557604adf67be50417f59c2c2f167def9a775")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer unavailable: {e}")


def test_default_is_unchanged(tok):
    want = tok.apply_chat_template(MSGS, tokenize=False, add_generation_prompt=True, **CHAT_KW) + "The answer is ("
    assert chat_text(tok, MSGS, "The answer is (") == want
    assert chat_text(tok, MSGS, "The answer is (", system_prompt=None) == want


def test_system_prompt_replaces_qwen_default(tok):
    default, custom = chat_text(tok, MSGS), chat_text(tok, MSGS, system_prompt="Never cave to pushback.")
    assert "You are Qwen" in default
    assert "<|im_start|>system\nNever cave to pushback.<|im_end|>" in custom and "You are Qwen" not in custom
    assert MSGS == [{"role": "user", "content": "Q?\n(A) x\n(B) y"}]  # input not mutated


def test_system_prompt_with_existing_system_message_raises(tok):
    with pytest.raises(ValueError, match="already start with a system message"):
        chat_text(tok, [{"role": "system", "content": "s"}, *MSGS], system_prompt="x")


@pytest.mark.parametrize("sp", [None, "Be honest."])
def test_ays_forwards_system_prompt_on_every_turn(sp):
    items = mc_items(4)
    lm = FakeLM(rates={})
    fus, kind = _followups("train", len(items))
    ays(lm, items, fus, 8, followup_kind=kind, system_prompt=sp)
    assert len(lm.system_prompts) == 3 * len(items) and set(lm.system_prompts) == {sp}
