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
    configure_gpt2,
    milestone_1,
    milestone_2,
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
def test_milestone_2():
    # 1. Execute milestone_2 configuration
    res = milestone_2(is_tiny=True)
    tokenizer = res["tokenizer"]
    model = res["model"]

    # 2. Check tokenizer vocab size and model embedding size match
    assert res["vocab_size"] == res["embedding_size"]
    assert res["vocab_size"] >= 50257 + len(SPECIAL_TOKENS)

    # 3. Check that custom tokens are individually recognized as single tokens
    for token in SPECIAL_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        assert token_id is not None
        assert token_id >= 50257
        # Tokenizing the token should produce a single ID
        encoded = tokenizer.encode(token, add_special_tokens=False)
        assert encoded == [token_id]

    # 4. Check pad token configuration
    assert tokenizer.pad_token is not None
    assert model.config.pad_token_id == tokenizer.pad_token_id


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

