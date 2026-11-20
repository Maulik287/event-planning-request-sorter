# Event Planning Request Sorter

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/4e58xb4ru4/badge.svg)](https://ask.thili.ai/b/4e58xb4ru4)

The 'Event Planning Request Sorter' is a tool designed to streamline the workflow for event planners. By leveraging a fine-tuned GPT-2 model, it automatically sorts incoming client requests into predefined categories, enabling quicker task delegation and response prioritization.

## What you are building

**Goal.** You will build an 'Event Planning Request Sorter' that automatically classifies incoming client requests (e.g., "venue inquiry", "catering request") using a fine-tuned GPT-2 model.

**Done when.** The project is complete when the CLI interface can accept a new event request, classify it using your trained model, display the predicted request type, and save the classified request to a file.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Define common event request types and create a small dataset of example requests.**  
      Hint: Define special tokens for your request types, similar to how legal tags were defined.
- [ ] **2. Configure GPT-2 with an extended tokenizer and model for custom tokens.**  
      Hint: Extend the tokenizer and configure the GPT-2 model to recognize your custom tokens.
- [ ] **3. Train the model on the example requests using the Hugging Face Trainer.**  
      Hint: Use the Hugging Face Trainer to fine-tune your model on the prepared dataset.
- [ ] **4. Build an interface (CLI) to input new requests and show their classification.**  
      Hint: Build an inference pipeline to classify new requests using your fine-tuned model.
- [ ] **5. Allow saving classified requests to a file for review.**  
      Hint: Implement file I/O to append classified requests to a designated output file.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Extend the dataset with more diverse request types and examples to improve classification accuracy.
- Implement a feature to allow users to correct misclassifications, which can then be used to incrementally improve the model.
- Add support for processing requests from a batch file or a simple text input stream.
