# Inter-iit-nlp


## 🎯 Overview

This project implements a two-stage pipeline for analyzing workplace conversations:
1. **Event Extraction**: Identifies key events, speakers, utterances, actions, and sentiment from conversation transcripts
2. **Causal Analysis**: Generates structured explanations of why conflicts, disagreements, or issues arose

The system uses:
- **Local LLM**: Microsoft Phi-3.5-mini-instruct (no API keys required)
- **Vector Database**: FAISS for efficient semantic search
- **Embeddings**: sentence-transformers for document encoding
- **Framework**: LangChain for orchestration

## 🚀 Features

- ✅ Fully local execution (no external API dependencies)
- ✅ RAG-based context retrieval from conversation history
- ✅ Structured event extraction with sentiment analysis
- ✅ Multi-cause causal explanations with evidence
- ✅ Support for GPU acceleration (optional)
- ✅ Text-based output parsing (robust to LLM variations)

## 📋 Requirements

```bash
pip install torch transformers accelerate bitsandbytes
pip install langchain langchain_community langchain_core langchain_text_splitters
pip install sentence-transformers faiss-cpu
```

For GPU support, install PyTorch with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```


## 💻 Usage

### Basic Example

```python
from pipeline import causal_explanation_pipeline

# Query the system
result = causal_explanation_pipeline("Why did the conflict arise?")

# Access structured results
print(result["events"])      # List of extracted events
print(result["causal"])      # Causal explanation with causes
```

### Sample Output

**Query**: "Why did the client get annoyed?"

**Events Extracted**:
```json
{
  "events": [
    {
      "speaker": "A",
      "utterance": "They lost revenue.",
      "action_type": "Expressing concern about financial loss",
      "sentiment": "negative",
      "relevance": "high"
    },
    {
      "speaker": "B",
      "utterance": "That's not our fault.",
      "action_type": "Deflecting responsibility",
      "sentiment": "negative",
      "relevance": "high"
    }
  ]
}
```

**Causal Analysis**:
```json
{
  "causal_explanation": "The client's annoyance stemmed from financial loss and feeling dismissed during interactions.",
  "causes": [
    {
      "cause": "Loss of revenue",
      "mechanism": "Financial impact led to dissatisfaction",
      "evidence": ["They lost revenue."]
    },
    {
      "cause": "Dismissive attitude",
      "mechanism": "Client felt concerns were not valued",
      "evidence": ["I felt dismissed during the call."]
    }
  ]
}
```



## 🔧 Configuration

### Model Selection

The default model is `microsoft/Phi-3.5-mini-instruct`. To use a different model:

```python
model_name = "your-model-name"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
```

### Embedding Model

The system uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings. To change:

```python
embedding_model = HuggingFaceEmbeddings(
    model_name="your-embedding-model"
)
```

### RAG Parameters

Adjust retrieval settings in `vector_db.as_retriever()`:

```python
retriever = vector_db.as_retriever(
    search_kwargs={"k": 5}  # Number of documents to retrieve
)
```

## 🎓 Example Queries

The system can answer various analytical questions:

- "What escalations happened during the meet?"
- "Why did the conflict arise?"
- "Explain the cause of disagreement."
- "Why did the client get annoyed?"
- "What communication issues occurred?"

## 🛠️ Troubleshooting

### GPU Memory Issues

If you encounter OOM errors, reduce batch size or use CPU:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",
    torch_dtype=torch.float32
)
```

### Slow Inference

For faster inference:
1. Enable GPU acceleration
2. Use quantization (4-bit or 8-bit)
3. Reduce `max_new_tokens` in pipeline config

### Parsing Errors

The system uses regex-based text parsing instead of JSON parsing for robustness. If events/causes are missing:
1. Check the raw LLM output in debug logs
2. Adjust the prompt templates in `event_prompt` or `causal_prompt`




## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional conversation datasets
- Alternative LLM backends (Llama, Mistral, etc.)
- Enhanced prompt engineering
- Multi-language support
- Web interface

