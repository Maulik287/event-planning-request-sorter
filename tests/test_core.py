"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

from app.core import (
    EVENT_REQUEST_TYPES,
    SPECIAL_TOKENS,
    RAW_EVENT_REQUESTS,
    create_event_dataset,
    format_prompt,
    milestone_1,
)

# Milestone 1: Define common event request types and create a small dataset of example requests.
def test_milestone_1():
    # 1. Verify defined request types
    assert "venue_inquiry" in EVENT_REQUEST_TYPES
    assert "catering_request" in EVENT_REQUEST_TYPES
    assert "entertainment_av" in EVENT_REQUEST_TYPES
    assert "decor_styling" in EVENT_REQUEST_TYPES
    assert "budget_pricing" in EVENT_REQUEST_TYPES
    assert "logistics_scheduling" in EVENT_REQUEST_TYPES

    # 2. Check special tokens formatting
    for name, token in EVENT_REQUEST_TYPES.items():
        assert token.startswith("<|") and token.endswith("|>")
        assert token in SPECIAL_TOKENS

    # 3. Test prompt formatting
    sample_text = "We need an outdoor garden for 100 guests."
    sample_token = "<|venue_inquiry|>"
    prompt = format_prompt(sample_text, sample_token)
    assert sample_text in prompt
    assert sample_token in prompt
    assert "<|endoftext|>" in prompt

    # 4. Check dataset generation and diversity
    dataset = create_event_dataset()
    assert len(dataset) >= 30
    categories = set(d["label"] for d in dataset)
    assert len(categories) == len(EVENT_REQUEST_TYPES)

    # 5. Check milestone_1 output
    res = milestone_1()
    assert res["num_examples"] == len(dataset)
    assert len(res["special_tokens"]) == len(SPECIAL_TOKENS)
    assert res["num_examples"] >= 40



# Milestone 2: Configure GPT-2 with an extended tokenizer and model for custom tokens.
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Train the model on the example requests using the Hugging Face Trainer.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Build an interface (CLI) to input new requests and show their classification.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Allow saving classified requests to a file for review.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

