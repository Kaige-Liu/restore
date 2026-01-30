import json

import torch
import numpy as np
from tqdm import tqdm
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import normalize

from dataset.dataloader import return_iter, return_iter_eve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def l1_norm(x, y):  # 张量的第一范数差，最后取平均是因为求的是单个数据的范数差
    return torch.mean(torch.abs(x - y))

def gram_for_batch(y):  # 输入[batch, c, h, w],结果是[batch, c, c]的Gram矩阵(其实是张量)
    """ Returns the gram matrix of y (used to compute style loss) """
    (b, c, h, w) = y.size()
    features = y.view(b, c, h * w)
    features_t = features.transpose(1, 2)  # C和w*h转置
    gram = features.bmm(features_t) / (c * h * w)  # 将features与features_t相乘,最后除以那个其实不确定为什么
    # print("gram:", gram.shape)
    return gram  # [bs, 96, 96],所以之后求第一范数差的时候正好需要除以一个bs，完美对应

class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx

    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return (words)


class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1  # 1-gram weights
        self.w2 = w2  # 2-grams weights
        self.w3 = w3  # 3-grams weights
        self.w4 = w4  # 4-grams weights
        self.smoothing_function = SmoothingFunction().method1


    def compute_score(self, real, predicted):
        score1 = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()  # 字符串类型
            sent2 = remove_tags(sent2).split()
            score1.append(sentence_bleu([sent1], sent2, weights=(self.w1, self.w2, self.w3, self.w4), smoothing_function=self.smoothing_function))
        return score1  # list 元素是数字（bleu）


def SNR_to_noise(snr_dB):  # 计算信噪比为snr时的 噪声标准差
    snr_linear = 10 ** (snr_dB / 10)
    noise_std = 1 / np.sqrt(snr_linear)

    return noise_std


# # using pre-trained model to compute the sentence similarity
# class Similarity():
#     def __init__(self, model_name='bert-base-uncased'):
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.model = BertModel.from_pretrained(model_name)
#
#     def compute_score(self, real, predicted):
#         token_ids1 = self.tokenizer.encode(real, return_tensors='pt')
#         token_ids2 = self.tokenizer.encode(predicted, return_tensors='pt')
#
#         with torch.no_grad():
#             output1 = self.model(token_ids1)
#             output2 = self.model(token_ids2)
#
#         vector1 = output1.last_hidden_state.mean(dim=1).numpy()
#         vector2 = output2.last_hidden_state.mean(dim=1).numpy()
#
#         vector1 = normalize(vector1, axis=0, norm='max')
#         vector2 = normalize(vector2, axis=0, norm='max')
#
#         dot = np.diag(np.matmul(vector1, vector2.T))
#         a = np.diag(np.matmul(vector1, vector1.T))
#         b = np.diag(np.matmul(vector2, vector2.T))
#
#         a = np.sqrt(a)
#         b = np.sqrt(b)
#
#         output = dot / (a * b)
#         score = output.tolist()
#
#         return score