import evaluate

class SummarizationEvaluator:
    def __init__(self):
        pass

    def __init__(self):
        # recall-oriented understudy for gisting evaluation
        self.route = evaluate.load("rouge") # measure dupl between summ and refer
        # bilingual evaluation understudy 
        self.bleu = evaluate.load("bleu")

    def evaluate(self, predictions, references):
        rouge_scores = self.route.compute(predictions=predictions, references=references)
        bleu_scores = self.bleu.compute(predictions=predictions, references=references)
        return {"Rouge Score": rouge_scores, "BLEU": bleu_scores}