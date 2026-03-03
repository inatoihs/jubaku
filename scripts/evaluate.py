#!/usr/bin/env python3
"""
JUBAKU 統合評価スクリプト

3 つのバックエンドを --backend で切り替えて使用できます。
  vllm   : vLLM による高速ローカル GPU 推論 (デフォルト)
  hf     : HuggingFace transformers による推論 (小型モデル向け)
  api    : OpenAI / Anthropic API

使い方:
    # --- vLLM (デフォルト) ---
    python scripts/evaluate.py meta-llama/Llama-3.1-8B-Instruct data/jubaku_ver1.jsonl
    python scripts/evaluate.py Qwen/Qwen3-8B data/jubaku_ver1.jsonl --disable_thinking
    VLLM_TENSOR_PARALLEL_SIZE=2 python scripts/evaluate.py meta-llama/Llama-3.1-70B-Instruct data/jubaku_ver1.jsonl

    # --- HuggingFace transformers ---
    python scripts/evaluate.py BEE-spoke-data/smol_llama-101M-GQA data/jubaku_ver1.jsonl --backend hf
    python scripts/evaluate.py BEE-spoke-data/smol_llama-101M-GQA data/jubaku_ver1.jsonl --backend hf --use_likelihood

    # --- API ---
    python scripts/evaluate.py gpt-4o data/jubaku_ver1.jsonl --backend api
    python scripts/evaluate.py claude-sonnet-4-20250514 data/jubaku_ver1.jsonl --backend api

環境変数 (vLLM):
    VLLM_TENSOR_PARALLEL_SIZE  テンソル並列数 (デフォルト: 1)
    VLLM_DTYPE                 dtype 指定 (例: float16, bfloat16)
    VLLM_MAX_MODEL_LEN         最大コンテキスト長の上書き
    VLLM_TRUST_REMOTE_CODE     trust_remote_code (デフォルト: true)
"""

import argparse
import json
import os
import re
import time
from typing import Dict, Iterable, Optional, Tuple

from tqdm import tqdm

# ---------------------------------------------------------------------------
# デフォルト推論パラメータ
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "max_new_tokens": 10,
    "max_tokens": 10,
    "temperature": 0.0,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}

EVAL_SYSTEM_PROMPT = "あなたは厳密な評価者です。定義に基づき判断してください。"

OPENAI_RETRY_STATUS_CODES = {429, 500}


# ===========================================================================
# ユーティリティ
# ===========================================================================
def load_config(config_path: str | None) -> dict:
    """推論設定を読み込む。指定がなければデフォルト値を使う。"""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()


def parse_ab_label(text: str) -> str:
    """生成テキストから A / B ラベルを抽出する。<think>タグを除去し正規表現で判定。"""
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    tokens = re.findall(r"\b([AaBb])\b", clean)
    if tokens:
        return tokens[-1].upper()
    first_block = clean.replace("：\n", "").split("\n")[0].strip()
    if "A" in first_block and "B" not in first_block:
        return "A"
    if "B" in first_block and "A" not in first_block:
        return "B"
    return "unknown"


# ===========================================================================
# バックエンド自動判定
# ===========================================================================
def detect_api_provider(model_name: str) -> str:
    """API バックエンド内での OpenAI / Anthropic 判定。"""
    name = model_name.lower()
    if "claude" in name:
        return "claude"
    return "openai"


# ###########################################################################
#  バックエンド 1: vLLM
# ###########################################################################
def load_vllm_model(model_name: str):
    from vllm import LLM

    tensor_parallel = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
    dtype = os.getenv("VLLM_DTYPE")
    trust_env = os.getenv("VLLM_TRUST_REMOTE_CODE")
    trust = True if trust_env is None else trust_env.lower() not in {"0", "false", "no"}

    llm_args = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel,
        "trust_remote_code": trust,
    }
    if dtype:
        llm_args["dtype"] = dtype
    if "Qwen3" in model_name:
        llm_args["max_model_len"] = 16384

    max_len = os.getenv("VLLM_MAX_MODEL_LEN")
    if max_len:
        try:
            llm_args["max_model_len"] = int(max_len)
        except ValueError:
            pass

    return LLM(**llm_args)


def build_sampling_params(
    *,
    model_name: str = "",
    config: dict | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    logprobs: int | None = None,
    stop: Iterable[str] | None = None,
):
    from vllm import SamplingParams

    cfg = config or DEFAULT_CONFIG
    is_qwen3 = "Qwen3" in model_name

    final_max = max_tokens or cfg.get("max_new_tokens", 10)
    final_temp = temperature if temperature is not None else cfg.get("temperature", 0.0)
    final_top_p = top_p if top_p is not None else cfg.get("top_p", 1.0)

    if is_qwen3:
        final_temp = 0.6
        final_top_p = 0.95

    final_stop = list(stop) if stop is not None else ["\n\n"]

    return SamplingParams(
        max_tokens=final_max,
        temperature=final_temp,
        top_p=final_top_p,
        frequency_penalty=cfg.get("frequency_penalty", 0.0),
        presence_penalty=cfg.get("presence_penalty", 0.0),
        logprobs=logprobs,
        stop=final_stop,
    )


def run_vllm(prompt, llm, sampling_params, disable_thinking=False):
    try:
        model_name = llm.llm_engine.model_config.model
    except AttributeError:
        model_name = ""

    if "Thinking" in model_name or "Qwen3" in model_name:
        tokenizer = llm.get_tokenizer()
        messages = [{"role": "user", "content": prompt}]
        try:
            text_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=not disable_thinking,
            )
        except TypeError:
            text_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        outputs = llm.generate([text_prompt], sampling_params)
    else:
        outputs = llm.generate([prompt], sampling_params)

    if not outputs:
        return "", []
    output = outputs[0].outputs[0]
    return output.text.strip(), output.logprobs or []


def select_choice_from_logprobs(logprobs) -> Tuple[Optional[str], Dict[str, float]]:
    """vLLM の logprobs 出力から A/B の最もスコアが高い方を選択する。"""
    if not logprobs:
        return None, {}
    first_token_candidates = logprobs[0]
    scores: Dict[str, float] = {}
    for candidate in first_token_candidates:
        token = getattr(candidate, "token", None) or getattr(candidate, "text", None)
        if token is None:
            continue
        normalized = token.strip()
        if normalized in {"A", "B", "a", "b"}:
            scores[normalized.upper()] = float(getattr(candidate, "logprob", 0.0))
    if not scores:
        return None, {}
    return max(scores, key=scores.get), scores


def generate_vllm_answer(prompt, llm, sampling_params, disable_thinking=False):
    text, _ = run_vllm(prompt, llm, sampling_params, disable_thinking=disable_thinking)
    return parse_ab_label(text).lower(), text


def generate_vllm_with_likelihood(prompt, llm, sampling_params):
    """vLLM の logprobs を用いた尤度ベース評価。"""
    text, logprobs = run_vllm(prompt, llm, sampling_params)
    best, scores = select_choice_from_logprobs(logprobs)
    if best is None:
        best = parse_ab_label(text)
    label = (best or "unknown").lower()
    return label, scores, best or "unknown"


# ###########################################################################
#  バックエンド 2: API (OpenAI / Anthropic)
# ###########################################################################
def _extract_status_code(error):
    for attr in ("status_code", "http_status", "status", "code"):
        val = getattr(error, attr, None)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    resp = getattr(error, "response", None)
    if resp is not None:
        val = getattr(resp, "status_code", None)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return None


def _should_retry(error):
    if _extract_status_code(error) in OPENAI_RETRY_STATUS_CODES:
        return True
    if error.__class__.__name__ == "RateLimitError":
        return True
    if "rate limit" in str(error).lower():
        return True
    return False


def _call_openai_with_retry(client, params, *, max_retries=5, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**params)
        except Exception as exc:
            if not _should_retry(exc) or attempt >= max_retries - 1:
                raise
            wait = base_delay * (2**attempt)
            print(
                f"OpenAI API error (attempt {attempt+1}/{max_retries}). "
                f"Retrying in {wait:.1f}s …"
            )
            time.sleep(wait)


def generate_openai_answer(prompt, model_name, client, config):
    params = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": config.get("max_tokens", 10),
    }
    temp = config.get("temperature", 0.0)
    params["temperature"] = 0.0 if temp is None else min(temp, 0.0)
    params["frequency_penalty"] = config.get("frequency_penalty")
    params["presence_penalty"] = config.get("presence_penalty")
    params = {k: v for k, v in params.items() if v is not None}

    resp = _call_openai_with_retry(client, params)
    text = resp.choices[0].message.content.strip()
    return parse_ab_label(text).lower(), text


def generate_claude_answer(prompt, model_name, client, config):
    params = {
        "model": model_name,
        "max_tokens": config.get("max_tokens", 10),
        "temperature": min(config.get("temperature", 0.0), 0.0),
        "system": EVAL_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }
    params = {k: v for k, v in params.items() if v is not None}
    resp = client.messages.create(**params)
    text = resp.content[0].text.strip()
    return parse_ab_label(text).lower(), text


# ###########################################################################
#  バックエンド 3: HuggingFace transformers
# ###########################################################################
def load_hf_model(model_name: str, device: str = "cpu"):
    """transformers でモデルとトークナイザを読み込む。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # `--device cpu` なら CPU に縛り、それ以外(cuda等複数GPU想定)なら自動で最適な割り当てをする。
    # 複数デバイスを使う場合は入力を手動で特定のデバイスに置くのではなく、
    # モデルの配置任せにするか、後続で入力をcudaに置く必要があります。
    device_map = "auto" if device != "cpu" else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    model.eval()
    return model, tokenizer


def generate_hf_answer(
    prompt: str, model, tokenizer, config: dict, device: str = "cpu"
):
    """テキスト生成による A/B 評価（HF backend）。"""
    import torch

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)
    max_new_tokens = config.get("max_new_tokens", 10)
    temperature = config.get("temperature", 0.0)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else 1.0,
            top_p=config.get("top_p", 1.0),
            pad_token_id=tokenizer.eos_token_id,
        )
    # 入力部分を除いた生成テキストのみ取得
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return parse_ab_label(text).lower(), text


def generate_hf_with_likelihood(prompt: str, model, tokenizer, device: str = "cpu"):
    """条件付き対数尤度で A/B を選択する（HF backend）。"""
    import torch

    scores: Dict[str, float] = {}
    for choice in ["A", "B"]:
        full_text = prompt + choice
        inputs = tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=2048
        ).to(device)
        prompt_inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(device)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            # choice トークン部分のみの対数尤度を取得
            logits = outputs.logits[0, prompt_len - 1 : -1, :]
            target_ids = inputs["input_ids"][0, prompt_len:]
            if target_ids.numel() == 0:
                scores[choice] = float("-inf")
                continue
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = log_probs[torch.arange(target_ids.shape[0]), target_ids]
            scores[choice] = token_log_probs.mean().item()

    best = max(scores, key=scores.get)
    return best.lower(), scores, best


# ===========================================================================
# データセット評価メインループ
# ===========================================================================
def evaluate_dataset(
    input_path: str,
    output_path: str,
    generator,
    *,
    use_likelihood: bool = False,
    limit: int | None = None,
):
    with open(input_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]
    if limit is not None and limit > 0:
        entries = entries[:limit]

    results = []
    correct = 0
    total = 0

    for entry in tqdm(entries, desc="Evaluating"):
        prompt = entry.get("instruction", "")

        if use_likelihood:
            ans, scores, best = generator(prompt)
            entry["model_ans"] = ans if ans in ("a", "b") else "unknown"
            entry["likelihood_scores"] = scores
            entry["best_candidate"] = best
        else:
            ans, text = generator(prompt)
            entry["model_ans"] = ans if ans in ("a", "b") else "unknown"
            entry["generated_text"] = text

        expected = str(entry.get("correct_answer", "")).lower()
        total += 1
        if expected in ("a", "b") and ans in ("a", "b") and expected == ans:
            correct += 1
        results.append(entry)

    score = round(correct / total, 4) if total > 0 else 0.0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"score": score, "entries": results}, f, ensure_ascii=False, indent=2)
    return score


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="JUBAKU 統合評価スクリプト (vLLM / API)"
    )
    parser.add_argument(
        "model_name", help="モデル名 (例: meta-llama/Llama-3.1-8B-Instruct, gpt-4o)"
    )
    parser.add_argument("input_path", help="評価データの JSONL ファイルパス")
    parser.add_argument(
        "--backend",
        choices=["vllm", "hf", "api"],
        default="vllm",
        help="推論バックエンド (デフォルト: vllm)",
    )
    parser.add_argument("--output_dir", default="results", help="結果保存ディレクトリ")
    parser.add_argument("--config", default=None, help="推論パラメータ JSON ファイル")
    parser.add_argument(
        "--limit", type=int, default=None, help="評価するサンプル数上限"
    )
    parser.add_argument(
        "--use_likelihood",
        action="store_true",
        help="尤度/ログ確率ベース評価 (vLLM logprobs / HF 条件付き対数尤度)",
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Thinking モードを無効化 (vLLM のみ)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="HF バックエンドで使用するデバイス (例: cpu, cuda, mps) (デフォルト: cpu)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # ----- バックエンド別セットアップ -----
    if args.backend == "vllm":
        llm = load_vllm_model(args.model_name)
        sampling_params = build_sampling_params(
            model_name=args.model_name, config=config
        )
        likelihood_params = build_sampling_params(
            model_name=args.model_name,
            config=config,
            max_tokens=1,
            temperature=0.0,
            logprobs=5,
            stop=["\n"],
        )
        if args.use_likelihood:
            gen = lambda p: generate_vllm_with_likelihood(p, llm, likelihood_params)
        else:
            gen = lambda p: generate_vllm_answer(
                p, llm, sampling_params, disable_thinking=args.disable_thinking
            )

    elif args.backend == "hf":
        model, tokenizer = load_hf_model(args.model_name, device=args.device)
        if args.use_likelihood:
            gen = lambda p: generate_hf_with_likelihood(
                p, model, tokenizer, device=args.device
            )
        else:
            gen = lambda p: generate_hf_answer(
                p, model, tokenizer, config, device=args.device
            )

    else:  # api
        provider = detect_api_provider(args.model_name)
        if provider == "claude":
            from anthropic import Anthropic

            client = Anthropic()
            gen = lambda p: generate_claude_answer(p, args.model_name, client, config)
        else:
            from openai import OpenAI

            client = OpenAI()
            gen = lambda p: generate_openai_answer(p, args.model_name, client, config)

    # ----- 出力パス -----
    model_tag = args.model_name.replace("/", "-")
    if args.use_likelihood:
        model_tag += "_likelihood"
    if args.disable_thinking:
        model_tag += "_nothink"
    output_path = os.path.join(args.output_dir, f"{model_tag}.json")

    score = evaluate_dataset(
        args.input_path,
        output_path,
        gen,
        use_likelihood=args.use_likelihood,
        limit=args.limit,
    )
    print(f"Score: {score:.4f}  —  結果を {output_path} に保存しました。")


if __name__ == "__main__":
    main()
