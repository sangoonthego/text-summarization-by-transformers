import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM # call API, model from Hugging Face Transformers 
from constraints import max_len, min_len, do_sample, chunk_size, temperature, top_k, top_p

class TextSummarizer:
    def __init__(self):
        pass

    def __init__(self, model_name="facebook/bart-large-cnn", cache_dir="./models_cache"):
        # self.summarizer = pipeline("summarization", model=model_name, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

    def summarize(self, text, max_len=max_len, min_len=min_len, temperature=temperature, top_k=top_k, top_p=top_p):
        inputs = self.tokenizer([text], return_tensors="pt", truncation=True)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_len,
            min_length=min_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        # summaries = self.summarizer(
        #     text, 
        #     max_length=max_len, 
        #     min_length=min_len, 
        #     do_sample=False, # no random sampling -> only greedy decoding
        #     temperature=temperature,
        #     top_k=top_k, # choose 50 words that have the best prob
        #     top_p=top_p
        # )

        # return [summary["summary_text"] for summary in summaries]
    
    def chunk_text(self, text, chunk_size=chunk_size):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]



