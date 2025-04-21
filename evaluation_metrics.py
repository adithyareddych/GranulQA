import torch
from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel, RetrievalAugmentationConfig
from transformers import AutoTokenizer, pipeline
from raptor import RetrievalAugmentation 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import evaluate
from raptor.qa_pipeline import RAPTORLLM





# ────────────────────────────────────────────────────────────────
# 1) Load & quantize Llama‑2‑7b‑chat into 4‑bit NF4
# ────────────────────────────────────────────────────────────────
_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

_bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading Llama-2-7B-chat in 4-bit …")
_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, use_fast=False)
_model     = AutoModelForCausalLM.from_pretrained(
    _MODEL_NAME,
    quantization_config=_bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
)
print("Model loaded in 4-bit NF4")

# ────────────────────────────────────────────────────────────────
# 2) Summarization & QA wrappers
# ────────────────────────────────────────────────────────────────
class SummarizationModel(BaseSummarizationModel):
    def __init__(self):
        self.tokenizer = _tokenizer
        self._pipe     = pipeline(
            "text-generation",
            model=_model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        messages = [
            {"role": "user", "content": f"Write a concise, information-dense summary of the following:\n{context}"}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        out = self._pipe(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
        return out[0]["generated_text"][len(prompt):].strip()

class QAModel(BaseQAModel):
    def __init__(self):
        self.tokenizer = _tokenizer
        self._pipe     = pipeline(
            "text-generation",
            model=_model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )

    def answer_question(self, context: str, question: str) -> str:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer as thoroughly as possible:"
                ),
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        out = self._pipe(
            prompt,
            max_new_tokens=256,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
        return out[0]["generated_text"][len(prompt):].strip()

class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text: str):
        return self.model.encode(text)

# ────────────────────────────────────────────────────────────────
# 3) Build RetrievalAugmentation
# ────────────────────────────────────────────────────────────────
RAC = RetrievalAugmentationConfig(
    summarization_model=SummarizationModel(),
    qa_model=QAModel(),
    embedding_model=SBertEmbeddingModel()
)
RA = RetrievalAugmentation(config=RAC)

# ────────────────────────────────────────────────────────────────
# 4) Main evaluation loop (with torch.cuda.empty_cache() after each doc)
# ────────────────────────────────────────────────────────────────
def main():
    ds    = load_dataset("deepmind/narrativeqa", "default")
    split = ds["validation"].select(range(1500))

    preds, refs = [], []
    raptor = RAPTORLLM()

    for i, ex in enumerate(split):
        # unpack document → story string
        doc_dict = ex["document"]
        doc_text = doc_dict["summary"]["text"]

        # unpack question → text
        q_dict       = ex["question"]
        question_txt = q_dict.get("text") or str(q_dict)

        # unpack answers → list of strings
        answers   = ex["answers"][0]
        ref_texts = answers["text"]

        # add & answer
        raptor.index_corpus([doc_text]) 
        pred = raptor.answer(question=question_txt)

        preds.append(pred)
        refs.append(ref_texts)
        print(f"... processed {i+1}/{len(split)}")

        # free any leftover GPU memory before next iteration
        torch.cuda.empty_cache()

    # compute metrics
    bleu   = evaluate.load("bleu")
    rouge  = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    b1 = bleu.compute(predictions=preds, references=refs, max_order=1)["bleu"]
    b4 = bleu.compute(predictions=preds, references=refs)["bleu"]
    rL = rouge.compute(predictions=preds, references=refs, use_stemmer=True)["rougeL"]
    m  = meteor.compute(predictions=preds, references=refs)["meteor"]

    print("\n" + "-"*40)
    print(f"BLEU-1:  {b1*100:.2f}")
    print(f"BLEU-4:  {b4*100:.2f}")
    print(f"ROUGE-L: {rL*100:.2f}")
    print(f"METEOR:  {m*100:.2f}")

if __name__ == "__main__":
    main()
