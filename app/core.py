"""Core implementation for Event Planning Request Sorter.

Milestones:
1. Define common event request types and create a small dataset of example requests.
2. Configure GPT-2 with an extended tokenizer and model for custom tokens.
3. Train the model on the example requests using the Hugging Face Trainer.
4. Build an interface (CLI) to input new requests and show their classification.
5. Allow saving classified requests to a file for review.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ==============================================================================
# Milestone 1: Event Request Types & Dataset Curation
# ==============================================================================

EVENT_REQUEST_TYPES: Dict[str, str] = {
    "venue_inquiry": "<|venue_inquiry|>",
    "catering_request": "<|catering_request|>",
    "entertainment_av": "<|entertainment_av|>",
    "decor_styling": "<|decor_styling|>",
    "budget_pricing": "<|budget_pricing|>",
    "logistics_scheduling": "<|logistics_scheduling|>",
}

# Special tokens list used to extend the tokenizer
SPECIAL_TOKENS: List[str] = list(EVENT_REQUEST_TYPES.values())

# Curated dataset of authentic event planning client requests across categories
RAW_EVENT_REQUESTS: List[Tuple[str, str]] = [
    # Venue Inquiry
    ("We are looking for an outdoor garden venue that can accommodate 150 guests for a wedding next June.", "venue_inquiry"),
    ("Is the Grand Ballroom available on Saturday, October 12th for our annual corporate gala?", "venue_inquiry"),
    ("What is the maximum standing capacity of the rooftop terrace for an evening cocktail reception?", "venue_inquiry"),
    ("We would like to schedule an in-person site tour of the banquet facilities this Thursday afternoon.", "venue_inquiry"),
    ("Do you have indoor spaces with natural lighting suitable for a daytime conference of 80 people?", "venue_inquiry"),
    ("Can we rent the historic courtyard for a private anniversary celebration with 50 seated attendees?", "venue_inquiry"),
    ("We need a conference hall equipped with breakout rooms for a 3-day tech symposium in November.", "venue_inquiry"),
    ("Are there any outdoor noise curfews or sound ordinances at your waterfront pavilion?", "venue_inquiry"),

    # Catering Request
    ("Could you provide sample menus for a 3-course plated dinner including gluten-free and vegan alternatives?", "catering_request"),
    ("We would like to arrange a private food tasting session before finalizing the wedding reception banquet.", "catering_request"),
    ("Can you accommodate kosher and halal dietary requirements for approximately 35 attendees?", "catering_request"),
    ("We are interested in an open bar package featuring craft cocktails, local beers, and premium wines.", "catering_request"),
    ("Do you offer late-night snack stations such as gourmet sliders, taco bars, or artisanal pizza?", "catering_request"),
    ("What are your passed hors d'oeuvres options for a two-hour standing networking cocktail reception?", "catering_request"),
    ("We need buffet breakfast catering with fresh pastries, seasonal fruit, and hot coffee for 100 people.", "catering_request"),
    ("Does the per-person catering package include waitstaff service, table linens, and tableware?", "catering_request"),

    # Entertainment & Audio/Visual
    ("We need to hire a professional DJ with a high-end sound system and wireless microphones for speeches.", "entertainment_av"),
    ("Can you provide dual high-definition projectors, presentation clickers, and stage monitors for keynote speakers?", "entertainment_av"),
    ("Does the stage have adequate power outlets and audio patching for a five-piece live jazz band?", "entertainment_av"),
    ("We require dynamic intelligent stage lighting, spotlights, and ambient uplighting around the perimeter.", "entertainment_av"),
    ("Will there be a dedicated on-site AV technician present throughout the awards presentation?", "entertainment_av"),
    ("We would like to set up an interactive 360-degree photo booth with digital sharing for attendees.", "entertainment_av"),
    ("Can we test the audio-visual system and screen connections the evening before the conference?", "entertainment_av"),
    ("Do you supply lavalier lapel microphones and podium mics for panel discussions?", "entertainment_av"),

    # Decor & Styling
    ("We want an elegant floral arch with white roses and eucalyptus for the ceremony backdrop.", "decor_styling"),
    ("Can you provide luxury velvet table linens, metallic charger plates, and custom menu cards?", "decor_styling"),
    ("We are planning a Gatsby 1920s theme with geometric art deco centerpieces and gold accents.", "decor_styling"),
    ("Do you have custom pipe and drape options in navy blue to divide the main hall into two sections?", "decor_styling"),
    ("We would like hanging fairy string lights and floral chandeliers suspended from the ceiling beams.", "decor_styling"),
    ("Can our external florist arrive three hours early to arrange the tall floral table centerpieces?", "decor_styling"),
    ("We need custom branded entrance signage, photo backdrops, and balloon installations for the product launch.", "decor_styling"),
    ("What candle policies do you have regarding open flame votives versus LED battery-operated candles?", "decor_styling"),

    # Budget & Pricing
    ("Could you send us a detailed cost breakdown and pricing quote for an 80-person rehearsal dinner?", "budget_pricing"),
    ("What is the required initial deposit percentage, and what is your installment payment schedule?", "budget_pricing"),
    ("Are service charges, gratuity, and local sales tax already factored into your package estimates?", "budget_pricing"),
    ("Do you offer discounted package rates for events booked on non-peak weekdays or Sunday evenings?", "budget_pricing"),
    ("What is your refund and cancellation policy if weather forces us to postpone the celebration?", "budget_pricing"),
    ("Could you please provide an itemized invoice showing venue rental separate from catering costs?", "budget_pricing"),
    ("We have a strict total budget of $15,000 for 100 guests—can you customize a package to fit that range?", "budget_pricing"),
    ("Is there an overtime fee per hour if our celebration extends past midnight?", "budget_pricing"),

    # Logistics & Scheduling
    ("What time can our vendors, florists, and decorators access the loading dock on event morning?", "logistics_scheduling"),
    ("Is on-site valet parking available for 200 cars, or should we arrange dedicated shuttle buses?", "logistics_scheduling"),
    ("Can you share the master production timeline for ceremony start, cocktail hour, and dinner service?", "logistics_scheduling"),
    ("Where can vendor delivery trucks and charter buses park during the duration of the event?", "logistics_scheduling"),
    ("How much time is allocated at the end of the evening for breakdown, strike, and trash removal?", "logistics_scheduling"),
    ("Is the venue fully wheelchair accessible with elevator access and ADA compliant restrooms?", "logistics_scheduling"),
    ("Can we deliver rental furniture and staging the afternoon prior to the event date for early staging?", "logistics_scheduling"),
    ("Who will be our primary on-site event coordinator and point of contact on the day of the event?", "logistics_scheduling"),
]


def format_prompt(text: str, token: Optional[str] = None) -> str:
    """Format an event request into prompt format for GPT-2 causal language modeling."""
    base = f"Request: {text.strip()}\nCategory: "
    if token:
        return f"{base}{token}<|endoftext|>"
    return base


def create_event_dataset(raw_samples: Optional[List[Tuple[str, str]]] = None) -> List[Dict[str, Any]]:
    """Create a structured dataset of event planning requests tagged with custom tokens."""
    samples = raw_samples if raw_samples is not None else RAW_EVENT_REQUESTS
    dataset: List[Dict[str, Any]] = []

    for idx, (text, label) in enumerate(samples):
        token = EVENT_REQUEST_TYPES.get(label, "<|unknown|>")
        dataset.append({
            "id": idx,
            "text": text,
            "label": label,
            "token": token,
            "formatted_text": format_prompt(text, token),
        })

    return dataset


# --- Milestone 1: Define common event request types and create a small dataset of example requests.
# Hint: Define special tokens for your request types, similar to how legal tags were defined.
def milestone_1() -> Dict[str, Any]:
    """Execute Milestone 1: Define request types, tokens, and curate dataset."""
    dataset = create_event_dataset()
    return {
        "request_types": dict(EVENT_REQUEST_TYPES),
        "special_tokens": list(SPECIAL_TOKENS),
        "dataset": dataset,
        "num_examples": len(dataset),
        "categories": list(EVENT_REQUEST_TYPES.keys()),
    }


# ==============================================================================
# Milestone 2: Configure GPT-2 Extended Tokenizer & Model
# ==============================================================================

def configure_gpt2(
    model_name: str = "gpt2",
    custom_tokens: Optional[List[str]] = None,
    is_tiny: bool = False,
) -> Tuple[Any, Any]:
    """Configure GPT-2 with an extended tokenizer and model containing custom tokens.

    Args:
        model_name: Pretrained model identifier or path.
        custom_tokens: List of special tokens representing event request categories.
        is_tiny: When True, initializes a tiny lightweight GPT-2 architecture suitable
                 for rapid testing, CI validation, and constrained environments.

    Returns:
        Tuple of (extended_tokenizer, configured_model)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

    tokens_to_add = custom_tokens if custom_tokens is not None else list(SPECIAL_TOKENS)

    # 1. Initialize tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Set pad_token if not already set (GPT-2 standard has no pad token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Add custom special tokens for request classification
    if tokens_to_add:
        special_tokens_dict = {"additional_special_tokens": tokens_to_add}
        tokenizer.add_special_tokens(special_tokens_dict)

    # 3. Initialize or load GPT-2 model
    if is_tiny:
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=128,
            n_embd=64,
            n_layer=2,
            n_head=2,
            bos_token_id=tokenizer.bos_token_id or 50256,
            eos_token_id=tokenizer.eos_token_id or 50256,
            pad_token_id=tokenizer.pad_token_id or 50256,
        )
        model = GPT2LMHeadModel(config)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception:
            # Fallback to direct GPT2LMHeadModel
            model = GPT2LMHeadModel.from_pretrained(model_name)
        # Resize token embeddings to include all newly added special tokens
        model.resize_token_embeddings(len(tokenizer))

    # Synchronize pad token id in model config
    model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model


# --- Milestone 2: Configure GPT-2 with an extended tokenizer and model for custom tokens.
# Hint: Extend the tokenizer and configure the GPT-2 model to recognize your custom tokens.
def milestone_2(model_name: str = "gpt2", is_tiny: bool = True) -> Dict[str, Any]:
    """Execute Milestone 2: Configure GPT-2 with extended tokenizer and custom tokens."""
    tokenizer, model = configure_gpt2(model_name=model_name, custom_tokens=SPECIAL_TOKENS, is_tiny=is_tiny)

    token_ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in SPECIAL_TOKENS}
    embedding_size = model.get_input_embeddings().weight.shape[0]

    return {
        "tokenizer": tokenizer,
        "model": model,
        "vocab_size": len(tokenizer),
        "embedding_size": embedding_size,
        "custom_tokens": list(SPECIAL_TOKENS),
        "token_ids": token_ids,
        "pad_token_id": tokenizer.pad_token_id,
    }



# --- Milestone 3: Train the model on the example requests using the Hugging Face Trainer.
# Hint: Use the Hugging Face Trainer to fine-tune your model on the prepared dataset.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Build an interface (CLI) to input new requests and show their classification.
# Hint: Build an inference pipeline to classify new requests using your fine-tuned model.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Allow saving classified requests to a file for review.
# Hint: Implement file I/O to append classified requests to a designated output file.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")

