"""Comprehensive test suite for the Event Planning Request Sorter milestones."""
import pytest

from app.core import (
    EVENT_REQUEST_TYPES,
    SPECIAL_TOKENS,
    RAW_EVENT_REQUESTS,
    create_event_dataset,
    format_prompt,
    configure_gpt2,
    prepare_training_dataset,
    train_model,
    RequestClassifier,
    save_classified_request,
    load_classified_requests,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
)
from app.cli import main

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
def test_milestone_3(tmp_path):
    # 1. Test dataset tokenization for trainer
    tokenizer, model = configure_gpt2(is_tiny=True)
    dataset = prepare_training_dataset(tokenizer)
    assert len(dataset) >= 30
    assert "input_ids" in dataset.column_names
    assert "labels" in dataset.column_names

    # 2. Run milestone_3 fine-tuning
    out_dir = str(tmp_path / "model_out")
    res = milestone_3(output_dir=out_dir, epochs=1, max_steps=2, is_tiny=True)
    assert res["steps"] > 0
    assert "output_dir" in res
    assert (tmp_path / "model_out" / "config.json").exists()


# Milestone 4: Build an interface (CLI) to input new requests and show their classification.
def test_milestone_4(capsys):
    # 1. Test inference pipeline directly
    classifier = RequestClassifier()

    r1 = classifier.classify("We are inquiring about booking the banquet ballroom for 200 guests.")
    assert r1["category"] == "venue_inquiry"
    assert r1["special_token"] == "<|venue_inquiry|>"
    assert r1["confidence"] > 0.0

    r2 = classifier.classify("Can we arrange a 3-course vegan dinner menu and wine tasting?")
    assert r2["category"] == "catering_request"
    assert r2["special_token"] == "<|catering_request|>"

    r3 = classifier.classify("We need stage lighting, sound speakers, and microphones for a live DJ.")
    assert r3["category"] == "entertainment_av"
    assert r3["special_token"] == "<|entertainment_av|>"

    # 2. Test milestone_4 execution
    m4_res = milestone_4()
    assert len(m4_res["results"]) == 3
    assert "sample_prediction" in m4_res

    # 3. Test CLI commands
    main(["list-types"])
    out1 = capsys.readouterr().out
    assert "venue_inquiry" in out1
    assert "<|venue_inquiry|>" in out1

    main(["classify", "What is the fee or price quote for the wedding package?"])
    out2 = capsys.readouterr().out
    assert "Predicted Type:" in out2
    assert "budget_pricing" in out2


# Milestone 5: Allow saving classified requests to a file for review.
def test_milestone_5(tmp_path, capsys):
    target_file = str(tmp_path / "saved_reviews.json")

    # 1. Test saving and loading records
    rec1 = {
        "request": "Can we deliver rental furniture the day before?",
        "category": "logistics_scheduling",
        "special_token": "<|logistics_scheduling|>",
        "confidence": 0.95,
    }
    saved1 = save_classified_request(rec1, output_path=target_file)
    assert saved1["id"] == 1
    assert "saved_at" in saved1

    rec2 = {
        "request": "We want white rose floral arches and custom linens.",
        "category": "decor_styling",
        "special_token": "<|decor_styling|>",
        "confidence": 0.92,
    }
    saved2 = save_classified_request(rec2, output_path=target_file)
    assert saved2["id"] == 2

    # Load and verify
    records = load_classified_requests(target_file)
    assert len(records) == 2
    assert records[0]["category"] == "logistics_scheduling"
    assert records[1]["category"] == "decor_styling"

    # 2. Test milestone_5 execution
    m5_target = str(tmp_path / "m5_output.json")
    m5_res = milestone_5(output_path=m5_target)
    assert m5_res["total_saved"] == 1
    assert (tmp_path / "m5_output.json").exists()

    # 3. Test CLI saving and review
    cli_file = str(tmp_path / "cli_classified.json")
    main(["classify", "We need a floral arch for the ceremony.", "--save", cli_file])
    out_save = capsys.readouterr().out
    assert "Saved to:" in out_save

    main(["review", "--file", cli_file])
    out_rev = capsys.readouterr().out
    assert "Saved Classified Requests" in out_rev
    assert "decor_styling" in out_rev

