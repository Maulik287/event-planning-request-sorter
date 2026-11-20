"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from app.core import (
    EVENT_REQUEST_TYPES,
    RequestClassifier,
    load_classified_requests,
    milestone_3,
    save_classified_request,
)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="event-planning-request-sorter",
        description="Event planners use this tool to automatically sort incoming client requests (e.g., \"venue inquiry\", \"catering request\") for efficient task delegation and response prioritization.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Subcommand: hello
    sub.add_parser("hello", help="check that everything is wired up")

    # Subcommand: list-types
    sub.add_parser("list-types", help="display supported event request categories and custom tokens")

    # Subcommand: classify
    classify_parser = sub.add_parser("classify", help="classify an incoming client request")
    classify_parser.add_argument("text", type=str, help="the client inquiry or request text")
    classify_parser.add_argument("--save", nargs="?", const="data/classified_requests.json", default=None,
                                help="optional file path to save the classified request for review (default: data/classified_requests.json)")
    classify_parser.add_argument("--model-path", type=str, default=None, help="path to fine-tuned model checkpoint")

    # Subcommand: train
    train_parser = sub.add_parser("train", help="fine-tune GPT-2 model using Hugging Face Trainer")
    train_parser.add_argument("--epochs", type=int, default=1, help="number of training epochs")
    train_parser.add_argument("--output-dir", type=str, default="data/fine_tuned_gpt2", help="model output directory")

    # Subcommand: review
    review_parser = sub.add_parser("review", help="display saved classified requests")
    review_parser.add_argument("--file", type=str, default="data/classified_requests.json", help="path to saved classified requests file")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Event Planning Request Sorter is ready. Open README.md for your milestones.")

    elif args.command == "list-types":
        print("\nSupported Event Request Categories & Tokens:")
        for cat, token in EVENT_REQUEST_TYPES.items():
            print(f"  - {cat:22} {token}")

    elif args.command == "classify":
        classifier = RequestClassifier(model_path=args.model_path)
        result = classifier.classify(args.text)

        print("\n--- Request Classification Result ---")
        print(f"Request:        {result['request']}")
        print(f"Predicted Type: {result['category']}")
        print(f"Special Token:  {result['special_token']}")
        print(f"Confidence:     {result['confidence'] * 100:.1f}%")

        if args.save:
            saved = save_classified_request(result, output_path=args.save)
            print(f"Saved to:       {args.save} (Record ID: {saved.get('id')})")

    elif args.command == "train":
        print(f"Starting training for {args.epochs} epoch(s)...")
        res = milestone_3(output_dir=args.output_dir, epochs=args.epochs, max_steps=2, is_tiny=True)
        print(f"Training complete. Loss: {res['loss']:.4f}, saved to: {res['output_dir']}")

    elif args.command == "review":
        entries = load_classified_requests(args.file)
        if not entries:
            print(f"No saved requests found in {args.file}")
        else:
            print(f"\n--- Saved Classified Requests ({len(entries)} total) ---")
            for entry in entries:
                rec_id = entry.get("id", "-")
                cat = entry.get("category", "unknown")
                req = entry.get("request", "")
                saved_at = entry.get("saved_at", "")
                print(f"[{rec_id}] ({cat}) {req[:70]}... ({saved_at})")


if __name__ == "__main__":
    main()

