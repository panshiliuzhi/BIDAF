import string
import collections


class SQuAD_Evaluate(object):

    def __init__(self):
        self.gold_answers = []
        self.pred_answers = []
        self.em = 0.0
        self.f1 = 0.0

    def add(self, gold_answer, pred_answer):

        self.gold_answers.append(gold_answer)
        self.pred_answers.append(pred_answer)
    def get_EM_F1(self):

        for a_gold, a_pred in zip(self.gold_answers, self.pred_answers):
            self.f1 += self.compute_f1(a_gold, a_pred)
            self.em += self.compute_exact(a_gold, a_pred)
        return 100.0*(self.f1/len(self.gold_answers)), 100.0*(self.em/len(self.gold_answers))

    def normalize_answer(self, s):
        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        return white_space_fix(remove_punc(s))

    def get_tokens(self, s):
        if not s: return []
        return self.normalize_answer(s).split()

    def compute_f1(self, a_gold, a_pred):
        gold_toks = self.get_tokens(a_gold)
        pred_toks = self.get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def compute_exact(self, a_gold, a_pred):
        return int(self.normalize_answer(a_gold) == self.normalize_answer(a_pred))