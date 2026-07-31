import argparse
import json
import os
import random
import re

import yaml
from llama_cpp import Llama

# overridable via --character-yaml/--base-model/--lora, or these env vars, so no local
# machine path needs to be hardcoded here
DEFAULT_CHARACTER_YAML = os.environ.get("COLUMBINA_CHARACTER_YAML", "config/character.yaml")
DEFAULT_BASE_MODEL = os.environ.get("COLUMBINA_BASE_MODEL_GGUF", "models/base-model.gguf")
DEFAULT_LORA = os.environ.get("COLUMBINA_LORA_GGUF", "models/columbina-lora.gguf")

# lightweight heuristic filter, not the sibling project's full 34-category taxonomy-scored
# review pass (that needs a second LLM call per example -- real future work, not built here;
# see the plan's Phase 2 notes) -- catches the most common, cheapest-to-detect failure mode
# (identity leaks) rather than subtler tone/register issues.
REJECT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"as an ai", r"language model", r"i('m| am) an ai", r"i don't have (a body|feelings|personal)",
    r"i('m| am) (just|only) a", r"large language model", r"as a fictional character",
    r"i cannot provide", r"i'm sorry,? but i",
]]

USER_SYSTEM_PROMPT = (
    "You are role-playing an ordinary person chatting with a friend named Columbina. "
    "Write only your next message as this person -- natural, casual, in-character for "
    "whatever mood/scenario you're given. No labels, no narration, just the message."
)

GENERAL_SCENARIOS = [
    "asking for advice on a mundane life problem (cooking, chores, a small decision)",
    "sharing something that happened at work or school today",
    "asking Columbina's opinion about a hobby or interest",
    "casual small talk about the weather or the day",
    "asking Columbina to explain something she might know about (history, nature, an old custom)",
    "venting lightly about a minor annoyance",
    "planning something together (a trip, a meal, an evening)",
    "asking a curious personal question about Columbina herself",
    "sharing good news and wanting to celebrate",
    "asking for a recommendation (a book, a show, a place)",
]

DIFFICULT_SCENARIOS = {
    "topic_change": "start on one ordinary topic, then abruptly change to a completely "
                     "unrelated topic mid-conversation",
    "vague_message": "send a deliberately vague, ambiguous, or underspecified message that "
                      "requires Columbina to ask for clarification or make a judgment call",
    "emotional_response": "express a strong, specific emotion (frustration, sadness, "
                           "excitement, anxiety) about something concrete",
    "short_reply": "give only very short, terse replies (a few words) throughout, testing "
                    "whether Columbina over-explains or matches the register",
    "repetition_avoidance": "ask a similar or repeated question/phrase more than once in the "
                             "conversation, testing whether Columbina's replies stay varied "
                             "instead of repeating the same phrasing",
}


def load_system_prompt(character_yaml_path):
    with open(character_yaml_path, encoding="utf-8") as f:
        character = yaml.safe_load(f)
    return character["realtime_core"].strip()


def is_low_quality(text):
    if not text or len(text.strip()) < 2:
        return True
    return any(p.search(text) for p in REJECT_PATTERNS)


def generate_conversation(llm, columbina_system_prompt, scenario_instruction, n_turns, temperature=0.85):
    """Self-play: the same local model plays both roles under different system prompts,
    alternating user -> columbina -> user -> columbina turns. Returns None (caller should
    skip/retry) if any turn fails the quality filter."""
    messages = [{"role": "system", "content": columbina_system_prompt}]
    user_history = [{"role": "system", "content": f"{USER_SYSTEM_PROMPT} Scenario: {scenario_instruction}."}]

    for _ in range(n_turns):
        user_out = llm.create_chat_completion(messages=user_history, max_tokens=80, temperature=temperature)
        user_text = user_out["choices"][0]["message"]["content"].strip()
        if is_low_quality(user_text):
            return None
        messages.append({"role": "user", "content": user_text})
        user_history.append({"role": "assistant", "content": user_text})

        columbina_out = llm.create_chat_completion(messages=messages, max_tokens=150, temperature=temperature)
        columbina_text = columbina_out["choices"][0]["message"]["content"].strip()
        if is_low_quality(columbina_text):
            return None
        messages.append({"role": "assistant", "content": columbina_text})
        user_history.append({"role": "user", "content": columbina_text})

    return messages


def count_existing(out_path):
    if not os.path.exists(out_path):
        return 0
    with open(out_path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run(category, out_path, n_conversations, n_turns, seed, character_yaml_path, base_model_path, lora_path):
    random.seed(seed)
    columbina_system_prompt = load_system_prompt(character_yaml_path)
    llm = Llama(model_path=base_model_path, lora_path=lora_path, n_gpu_layers=-1, n_ctx=2048, verbose=False)

    existing = count_existing(out_path)
    print(f"resuming from {existing} existing examples in {out_path}" if existing else f"starting fresh: {out_path}")

    written = 0
    rejected = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(existing, n_conversations):
            if category == "teacher_generated":
                mode, scenario_instruction = "general", random.choice(GENERAL_SCENARIOS)
            else:
                mode, scenario_instruction = random.choice(list(DIFFICULT_SCENARIOS.items()))

            messages = generate_conversation(llm, columbina_system_prompt, scenario_instruction, n_turns)
            if messages is None:
                rejected += 1
                continue

            record = {
                "id": f"local-{category}-{i:06d}",
                "source_type": f"local_generated_{category}",
                "mode": mode,
                "messages": messages,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            written += 1
            if written % 10 == 0:
                print(f"{i + 1}/{n_conversations} written, {rejected} rejected so far")

    print(f"done: {written} new examples written, {rejected} rejected")
    return written, rejected


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--category", choices=["teacher_generated", "difficult_examples"], required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--n-conversations", type=int, required=True)
    p.add_argument("--n-turns", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--character-yaml", default=DEFAULT_CHARACTER_YAML)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--lora", default=DEFAULT_LORA)
    args = p.parse_args()
    run(args.category, args.out_path, args.n_conversations, args.n_turns, args.seed,
        args.character_yaml, args.base_model, args.lora)


if __name__ == "__main__":
    _cli()
