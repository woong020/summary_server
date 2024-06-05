# -*- coding: utf-8 -*-

import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration


# Model 2
class CustomSummarizer:
    def __init__(self):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
        self.model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

    def summarize(self, text):
        raw_input_ids = self.tokenizer.encode(text)
        input_ids = [self.tokenizer.bos_token_id] + raw_input_ids + [self.tokenizer.eos_token_id]

        summary_ids = self.model.generate(torch.tensor([input_ids]),  num_beams=4,  max_length=512,  eos_token_id=1)
        return self.tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)


# 텍스트 파일에서 원문 읽기
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def summarizing():
    # 파일 경로 수정 필요
    input_file_path = r'received_files/text.txt'
    # 원문 텍스트 파일에서 읽기
    original_text = read_text_file(input_file_path)
    # 요약 모델 인스턴스 생성
    custom_summarizer = CustomSummarizer()
    # 각 모델로부터 요약 결과 가져오기
    summary_custom = custom_summarizer.summarize(original_text)
    
    return summary_custom

#summarizing()
