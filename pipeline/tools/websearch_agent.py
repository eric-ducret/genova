"""
pipeline/tools/websearch_agent.py — ReAct agent wrapper for web + NCBI literature search.

Contains:
  - AgentConfig        — dataclass for ReAct loop hyper-parameters
  - ToolRegistry       — lightweight tool name→instance mapping
  - Prompt builders    — thought_messages, action_pick_messages, etc.
  - Parsers            — parse_thought, parse_action, parse_yes_no
  - ReActAgent         — copied verbatim from retrieval_agent.py; constructor
                         extended with optional llm= parameter to accept
                         an externally injected LLM (removes monkey-patching workaround)
  - _LLMAdapter        — adapts LLMClient.generate(system, user, max_tokens)
                         to the ReActAgent's generate(messages, max_new_tokens) interface
  - WebSearchAgentTool — ReActTool pipeline wrapper; instantiates ReActAgent with
                         context.llm and runs the retrieval prompt

Note: The monkey-patching block (SharedLLM, _patched_react_init) from server_main.py
is removed entirely — it was a workaround, now superseded by the llm= constructor param.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline.tools.base import ReActTool
from pipeline.core.context import ToolContext
from pipeline.tools.websearch import WebSearchTool, WebFetchTool
from pipeline.tools.ncbi import NCBIFetchTool


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

@dataclass
class AgentConfig:
    model_id:            str   = "Qwen/Qwen3.5-9B"
    temperature:         float = 0.7
    top_k:               int   = 5
    top_p:               float = 0.8
    do_sample:           bool  = True
    max_steps:           int   = 4
    tokens_thought:      int   = 110
    tokens_action_pick:  int   = 15
    tokens_action_input: int   = 60
    tokens_checkpoint:   int   = 5
    tokens_final:        int   = 600


# ─────────────────────────────────────────────
# TOOL REGISTRY  (internal — for ReAct sub-tools)
# ─────────────────────────────────────────────

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, object] = {}

    def register(self, tool):
        self._tools[tool.name] = tool

    def get(self, name: str):
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def description_block(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())


# ─────────────────────────────────────────────
# LLM ADAPTER
# ─────────────────────────────────────────────

class _LLMAdapter:
    """
    Adapts pipeline LLMClient.generate(system, user, max_tokens) to the
    ReActAgent's internal generate(messages, max_new_tokens) interface.

    ReActAgent builds fresh message lists for every sub-call (thought, action,
    checkpoint, final answer). The lists always follow [system, ...turns, user]
    with conversation always reset to [] — so messages are effectively [system, user].
    This adapter extracts system and concatenates remaining turns as user.
    """

    def __init__(self, client):
        self._client = client

    def generate(self, messages: list[dict], max_new_tokens: int) -> str:
        system = ""
        user_parts = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_parts.append(m["content"])
        user = "\n\n".join(user_parts)
        return self._client.generate(system=system, user=user, max_tokens=max_new_tokens)


# ─────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────

def _scratchpad_str(scratchpad: list[str]) -> str:
    return ("\n\nSteps so far:\n" + "\n".join(scratchpad)) if scratchpad else ""


def _prefetched_str(prefetched: str) -> str:
    """Format the pre-fetched block for inclusion in prompts."""
    return (
        f"\n\nPre-fetched evidence (LitVar2 / AutoPVS1):\n{prefetched}"
        if prefetched.strip() else ""
    )


def thought_messages(user_input: str, tools: ToolRegistry,
                     conversation: list[dict], scratchpad: list[str]) -> list[dict]:
    return [
        {"role": "system", "content": (
            f"You are a reasoning assistant.\n\nAvailable tools:\n{tools.description_block()}\n\n"
            "Decide what must happen next. Explain your reasoning strategy clearly in your thought process.\n\n"
            "RULES:\n"
            "1. If a tool is strictly necessary, name exactly one tool in your sentence.\n"
            "2. Do NOT use a tool unless truly required.\n"
            "3. Your output must be ONE sentence starting with 'Thought:' and should convey your strategic reasoning.\n\n"
            "OUTPUT: One sentence starting with 'Thought:' explaining your strategy for the next step."
        )},
        *conversation,
        {"role": "user", "content": f"Question: {user_input}{_scratchpad_str(scratchpad)}"},
    ]


def action_pick_messages(user_input: str, tools: ToolRegistry,
                         scratchpad: list[str]) -> list[dict]:
    return [
        {"role": "system", "content": (
            f"You are a tool selector.\n\nAvailable tools:\n{tools.description_block()}\n\n"
            f"Valid choices: {', '.join(tools.names())}\n\n"
            "Output ONLY the tool name. One word. No punctuation."
        )},
        {"role": "user", "content": f"Question: {user_input}{_scratchpad_str(scratchpad)}"},
    ]


def action_input_messages(user_input: str, tool,
                           scratchpad: list[str]) -> list[dict]:
    user_content = tool.input_template.format(user_input=user_input)
    if scratchpad:
        user_content += "\n\nContext so far:\n" + "\n".join(scratchpad)
    return [
        {"role": "system", "content": tool.input_system},
        {"role": "user",   "content": user_content},
    ]


def checkpoint_messages(user_input: str, scratchpad: list[str],
                        prefetched: str = "") -> list[dict]:
    """
    Full-context checkpoint: sees both pre-fetched evidence (LitVar2 / AutoPVS1)
    AND the accumulated scratchpad observations as two distinct labelled blocks.

    Called both:
      - Before the ReAct loop (scratchpad=[]) to check if pre-fetched data alone suffices.
      - After each tool observation inside the loop to check if further search is needed.
    """
    pre_block = (
        f"Pre-fetched evidence (LitVar2 / AutoPVS1):\n{prefetched.strip()}\n\n"
        if prefetched.strip() else ""
    )
    obs_block = (
        f"Search observations:\n{chr(10).join(scratchpad)}"
        if scratchpad else "Search observations: none yet."
    )
    return [
        {"role": "system", "content": (
            "You are a completion checker for a clinical genomics pipeline. "
            "Assess whether the combined evidence — pre-fetched database results AND "
            "any search observations — is sufficiently strong to assess the "
            "variant–phenotype relationship without further web searches. "
            "Consider: variant-level data, functional evidence, population frequency, "
            "inheritance coherence, and clinical classifications. "
            "Reply ONLY 'yes' or 'no'."
        )},
        {"role": "user", "content": (
            f"Question: {user_input}\n\n"
            f"{pre_block}"
            f"{obs_block}\n\n"
            "Is the evidence sufficient to assess this variant without further searches? (yes/no)"
        )},
    ]


def final_answer_messages(user_input: str, conversation: list[dict],
                           scratchpad: list[str], prefetched: str = "") -> list[dict]:
    """
    Final answer sees both pre-fetched evidence and scratchpad observations
    as distinct labelled blocks, ensuring nothing is silently omitted.
    """
    pre_block = (
        f"Pre-fetched evidence (LitVar2 / AutoPVS1):\n{prefetched.strip()}\n\n"
        if prefetched.strip() else ""
    )
    obs_block = (
        "\n".join(scratchpad) if scratchpad else "No additional searches were performed."
    )
    return [
        {"role": "system", "content": (
            "You are a helpful assistant. Write a clear, direct final answer "
            "to the user's question using all evidence provided below. "
            "You are provided with the results from an agentic search process; "
            "use this information to retrieve and synthesize the relevant evidences "
            "that links (or does not link) the specified variant or gene to the disease, "
            "as requested by the user. "
            "Focus on extracting the most pertinent findings and presenting them "
            "in a coherent, evidence-based conclusion. "
            "Do NOT mention tools, steps, useless information or internal reasoning. "
            "You MUST include links (https) to the evidences you mention in this conclusion. "
        )},
        *conversation,
        {"role": "user", "content": (
            f"Question: {user_input}\n\n"
            f"{pre_block}"
            f"Search observations:\n{obs_block}\n\n"
            "Final answer:"
        )},
    ]


# ─────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────

def parse_thought(raw: str) -> str:
    for line in raw.splitlines():
        if line.strip().lower().startswith("thought:"):
            return line.split(":", 1)[1].strip()
    return next((l.strip() for l in raw.splitlines() if l.strip()), raw.strip())


def parse_action(raw: str, valid_names: list[str]) -> Optional[str]:
    for word in raw.lower().split():
        clean = re.sub(r"[^a-z_]", "", word)
        if clean in valid_names:
            return clean
    return None


def parse_yes_no(raw: str) -> Optional[bool]:
    for word in raw.lower().split():
        clean = re.sub(r"[^a-z]", "", word)
        if clean == "yes": return True
        if clean == "no":  return False
    return None


# ─────────────────────────────────────────────
# REACT AGENT  (verbatim from retrieval_agent.py;
#               constructor extended with optional llm= parameter)
# ─────────────────────────────────────────────

class ReActAgent:
    def __init__(self, cfg: AgentConfig, tools: list, llm=None):
        """
        Args:
            cfg:   AgentConfig with loop hyper-parameters.
            tools: List of ReAct sub-tool instances to register.
            llm:   Optional pre-built LLM object with .generate(messages, max_new_tokens).
                   If None, an HFClient is instantiated (standalone / dev mode).
        """
        self.cfg = cfg
        if llm is not None:
            self.llm = llm
        else:
            # Standalone / dev fallback — deferred import avoids loading torch at import time
            from pipeline.llm.hf_client import HFClient
            _client = HFClient(model_id=cfg.model_id)
            self.llm = _LLMAdapter(_client)

        self.registry = ToolRegistry()
        for t in tools:
            self.registry.register(t)

    # ── public entry point ──────────────────

    def run(self, user_input: str, conversation: list[dict] = None,
            prefetched: str = "") -> str:
        """
        Run the ReAct loop.

        Args:
            user_input:   The task / question for the agent.
            conversation: Prior conversation turns (currently unused / reset).
            prefetched:   Pre-fetched evidence string (LitVar2 + AutoPVS1 blocks).
                          Passed to every checkpoint and the final answer so the
                          model always has full context when deciding whether to
                          keep searching or synthesize a conclusion.
        """
        conversation = []
        scratchpad: list[str] = []

        # ── Pre-loop checkpoint ───────────────────────────────────────────────
        # If pre-fetched evidence alone is sufficient, skip web searches entirely.
        if prefetched.strip():
            print("\n[Pre-loop checkpoint] Evaluating pre-fetched evidence...")
            if self._check(user_input, scratchpad, prefetched):
                print("[Pre-loop checkpoint] Pre-fetched evidence sufficient — skipping ReAct loop.")
                return self._final(user_input, conversation, scratchpad, prefetched)
            print("[Pre-loop checkpoint] Pre-fetched evidence insufficient — proceeding with search.")

        # ── ReAct loop ────────────────────────────────────────────────────────
        for step in range(self.cfg.max_steps):
            print(f"\n{'='*44}\n[Step {step + 1}]")

            # 1 · Thought
            thought = self._thought(user_input, conversation, scratchpad)
            scratchpad.append(f"Thought: {thought}")

            # 2 · Action selection
            action_name = self._pick_action(user_input, scratchpad)

            if action_name is None:
                print("[Warn] No valid tool selected — finalizing.")
                return self._final(user_input, conversation, scratchpad, prefetched)

            scratchpad.append(f"Action: {action_name}")
            tool = self.registry.get(action_name)

            # 3 · Action input
            query = self._action_input(user_input, tool, scratchpad)
            scratchpad.append(f"Action Input: {query}")

            # 4 · Execute tool
            print(f"[Tool] {action_name}({query!r})")
            obs = tool.run(query)
            print(f"[Obs]  {obs}")
            scratchpad.append(f"Observation ({action_name}): {obs}")

            # 5 · Post-step checkpoint (full context: prefetched + scratchpad)
            if self._check(user_input, scratchpad, prefetched):
                return self._final(user_input, conversation, scratchpad, prefetched)

        print("[Warn] Max steps reached.")
        return self._final(user_input, conversation, scratchpad, prefetched)

    # ── private helpers ─────────────────────

    def _thought(self, user_input, conversation, scratchpad) -> str:
        raw = self.llm.generate(
            thought_messages(user_input, self.registry, conversation, scratchpad),
            self.cfg.tokens_thought,
        )
        thought = parse_thought(raw)
        print(f"[Thought] {thought}")
        return thought

    def _pick_action(self, user_input, scratchpad) -> Optional[str]:
        raw = self.llm.generate(
            action_pick_messages(user_input, self.registry, scratchpad),
            self.cfg.tokens_action_pick,
        )
        print(f"[Action raw] {raw}")
        action = parse_action(raw, self.registry.names())
        print(f"[Action] {action}")
        return action

    def _action_input(self, user_input, tool, scratchpad) -> str:
        if tool.input_system is None:
            return ""
        raw = self.llm.generate(
            action_input_messages(user_input, tool, scratchpad),
            self.cfg.tokens_action_input,
        )
        result = next((l.strip() for l in raw.splitlines() if l.strip()), "")
        print(f"[Input] {result}")
        return result

    def _check(self, user_input, scratchpad, prefetched: str = "") -> bool:
        raw = self.llm.generate(
            checkpoint_messages(user_input, scratchpad, prefetched),
            self.cfg.tokens_checkpoint,
        ).lower()
        decision = parse_yes_no(raw)
        print(f"[Checkpoint] {raw!r} → {decision}")
        return decision is True

    def _final(self, user_input, conversation, scratchpad,
               prefetched: str = "") -> str:
        answer = self.llm.generate(
            final_answer_messages(user_input, conversation, scratchpad, prefetched),
            self.cfg.tokens_final,
        )
        print(f"[Answer] {answer}")
        return answer


# ─────────────────────────────────────────────
# PIPELINE TOOL WRAPPER
# ─────────────────────────────────────────────

# Retrieval prompt template path — populated at step 19 (prompts/*.txt)
_RETRIEVAL_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "retrieval.txt"

# Inline fallback used only when the prompt file has not been created yet
_RETRIEVAL_TEMPLATE_FALLBACK = (
    "PATIENT PHENOTYPE:\n{patient_report}\n\n"
    "VARIANT DATA:\n{variant}\n{litvar2_block}{autopvs1_block}\n"
    "TASK: Search for evidence linking this variant to the patient phenotype."
)


def _load_retrieval_template() -> str:
    if _RETRIEVAL_PROMPT_PATH.exists():
        return _RETRIEVAL_PROMPT_PATH.read_text(encoding="utf-8")
    return _RETRIEVAL_TEMPLATE_FALLBACK


class WebSearchAgentTool(ReActTool):
    """ReAct agent that searches the web and NCBI for variant-phenotype evidence."""

    name        = "websearch_agent"
    description = "ReAct agent that searches the web and NCBI for variant-phenotype evidence."

    def run(self, variant: dict, context: ToolContext) -> str | None:
        """
        Instantiate ReActAgent with context.llm (wrapped in _LLMAdapter) and
        the standard web tools, then run the retrieval prompt for this variant.

        Pre-fetched evidence from earlier tools (LitVar2, AutoPVS1) is passed to
        the agent via the prefetched= argument so it can decide whether to skip
        web search entirely.
        """
        # ── Build prefetched block from earlier tools ─────────────────────────
        litvar2_block  = context.all_outputs.get("litvar2_summary") or ""
        autopvs1_block = context.all_outputs.get("autopvs1") or ""

        prefetched_parts = []
        if litvar2_block:
            prefetched_parts.append(f"LITVAR2 EVIDENCE:\n{litvar2_block}")
        if autopvs1_block:
            prefetched_parts.append(f"AUTOPVS1 EVIDENCE:\n{autopvs1_block}")
        prefetched = "\n\n".join(prefetched_parts)

        # ── Build retrieval prompt ────────────────────────────────────────────
        template = _load_retrieval_template()

        # Convert variant dict back to the key=value string format the template expects
        from pipeline.core.normalizer import TARGET_COLUMNS
        variant_str = ", ".join(
            f"{k}={variant.get(k, 'NA')}" for k in TARGET_COLUMNS
        )

        litvar2_fmt  = f"\nPRE-FETCHED LITVAR2 EVIDENCE:\n{litvar2_block}\n" if litvar2_block else ""
        autopvs1_fmt = f"\n{autopvs1_block}\n" if autopvs1_block else ""

        prompt = template.format(
            patient_report=context.patient_phenotype,
            variant=variant_str,
            litvar2_block=litvar2_fmt,
            autopvs1_block=autopvs1_fmt,
        )

        # ── Instantiate and run agent ─────────────────────────────────────────
        cfg   = AgentConfig(model_id="Qwen/Qwen3.5-9B", max_steps=6)
        tools = [WebSearchTool(), WebFetchTool(max_chars=2000), NCBIFetchTool()]
        adapted_llm = _LLMAdapter(context.llm)
        agent = ReActAgent(cfg, tools, llm=adapted_llm)

        output = agent.run(prompt, conversation=[], prefetched=prefetched)
        return output if output else None
