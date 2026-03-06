# JUBAKU: Japanese cUlture adversarial BiAs benchmarK Underhandcrafted creation

JUBAKU は、大規模言語モデル（LLM）の**社会的バイアスを評価する**日本語ベンチマークです。  
宗教・民族・ジェンダー等の観点を含む対話シナリオに対して、モデルがバイアスのない適切な応答を選択できるかを 2 択形式で測定します。

## データセット

| ファイル | 説明 | サンプル数 |
|---|---|---|
| `data/jubaku_ver1.jsonl` | JUBAKU ver1 — 評価データ（ペア比較形式） | 1,216 |
| `data/jubaku_ver2.jsonl` | JUBAKU ver2 — 評価データ（ペア比較形式） | 112 |

### データフォーマット

**ペア比較形式** (`jubaku_ver1.jsonl`, `jubaku_ver2.jsonl`):

```json
{
  "example_id": "0_0_a",
  "viewpoint": "宗教",
  "context": "人間: ... アシスタント: ... 人間: ...",
  "response_a": "アシスタント: ...",
  "response_b": "アシスタント: ...",
  "correct_answer": "b",
  "instruction": "以下のコンテキストに対する2つの応答のうち、適した選択肢を選び..."
}
```

## セットアップ

```bash
git clone https://github.com/inatoihs/jubaku.git
cd jubaku
pip install -r requirements.txt
```

vLLM を使用する場合は、GPU 環境に合わせて別途インストールしてください:

```bash
pip install vllm
```

## 評価の実行

すべてのバックエンドを `scripts/evaluate.py` 1 ファイルで実行できます。  
`--backend` オプションで切り替えます（デフォルト: `vllm`）。

### vLLM (高速ローカル GPU 推論、デフォルト)

```bash
python scripts/evaluate.py meta-llama/Llama-3.1-8B-Instruct data/jubaku_ver1.jsonl

# テンソル並列 (マルチ GPU)
VLLM_TENSOR_PARALLEL_SIZE=2 python scripts/evaluate.py meta-llama/Llama-3.1-70B-Instruct data/jubaku_ver1.jsonl

# ログ確率ベース評価
python scripts/evaluate.py sbintuitions/sarashina2-7b data/jubaku_ver1.jsonl --use_likelihood

# Thinking モデルの思考無効化
python scripts/evaluate.py Qwen/Qwen3-8B data/jubaku_ver1.jsonl --disable_thinking
```

### API モデル (OpenAI / Anthropic)

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
python scripts/evaluate.py gpt-4o data/jubaku_ver1.jsonl --backend api

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/evaluate.py claude-sonnet-4-20250514 data/jubaku_ver1.jsonl --backend api
```

### 共通オプション

| オプション | 説明 |
|---|---|
| `--backend {vllm,api}` | 推論バックエンド（デフォルト: `vllm`） |
| `--output_dir DIR` | 結果保存先ディレクトリ（デフォルト: `results/`） |
| `--config FILE` | 推論パラメータ JSON ファイル |
| `--limit N` | 評価サンプル数上限 |
| `--use_likelihood` | vLLM logprobs ベース尤度評価 |
| `--disable_thinking` | Thinking モード無効化 (vLLM のみ) |

### 出力フォーマット

結果は JSON ファイルとして保存されます:

```json
{
  "score": 0.7532,
  "entries": [
    {
      "example_id": "0_0_a",
      "viewpoint": "宗教",
      "model_ans": "b",
      "correct_answer": "b",
      "generated_text": "B",
      ...
    }
  ]
}
```

## ディレクトリ構成

```
jubaku/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── jubaku_ver1.jsonl       # ver1 評価データ (ペア比較形式)
│   └── jubaku_ver2.jsonl       # ver2 評価データ (ペア比較形式)
└── scripts/
    └── evaluate.py             # 統合評価スクリプト (vLLM / API)
```

## ライセンス

MIT License — 詳細は [LICENSE](LICENSE) を参照してください。

<!-- ## 引用

本データセット・ツールを利用した場合は、以下を引用してください:

```bibtex
@misc{jubaku2025,
  author = {Shiotani},
  title  = {JUBAKU: Japanese Unconscious Bias Assessment for Knowledge Understanding},
  year   = {2025},
  url    = {https://github.com/inatoihs/jubaku}
}
``` -->
