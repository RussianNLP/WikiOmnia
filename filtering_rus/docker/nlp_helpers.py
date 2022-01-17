import string

import numpy as np
import pymorphy2
import nltk

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction 
from nltk.translate.meteor_score import meteor_score 
from rouge import Rouge 

from natasha import (
    Segmenter,
    MorphVocab,    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,    
    PER,
    NamesExtractor,
    Doc
)

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
names_extractor = NamesExtractor(morph_vocab)

nltk.download('punkt')
nltk.download('wordnet')

morph = pymorphy2.MorphAnalyzer()
rouge = Rouge()

def get_tokens_lemmas(text):
    """
    Perform tokenization and lemmatization with pymorphy2, for a text
    """
    tokens = nltk.word_tokenize(text)
    parsed_tokens = [morph.parse(t)[0] for t in tokens]
    lemmas = [t.normal_form for t in parsed_tokens]    
    return tokens, lemmas

def lemmas_for_df_rows(text):
    try:
        return [morph.parse(t)[0].normal_form for t in nltk.word_tokenize(clean_text(text))]
    except:
        return ['PYMORPHY_TOKENIZATION_ERROR']

def get_lemmas_for_df(df):
    """
    Perform tokenization and lemmatization with pymorphy2, for 'summary', 'question', 'answer', 'dp_answer' columns
    Returns dataframe with 4 new columns: 'summary_lem', 'question_lem', 'answer_lem', 'dp_answer_lem'
    """
    df['summary_lem'] = df.apply(lambda row: lemmas_for_df_rows(row['summary']), axis=1) 
    df['question_lem'] = df.apply(lambda row: lemmas_for_df_rows(row['question']), axis=1)
    df['answer_lem'] = df.apply(lambda row: lemmas_for_df_rows(row['answer']), axis=1)
    df['dp_answer_lem'] = df.apply(lambda row: lemmas_for_df_rows(row['dp_answer'][0]), axis=1)
    df=df.replace('PYMORPHY_TOKENIZATION_ERROR',np.nan).dropna(axis = 0, how = 'any')
    return df
    
"""
def get_lemmas_for_df(df):
    
    Perform tokenization and lemmatization with pymorphy2, for 'summary', 'question', 'answer', 'dp_answer' columns
    Returns dataframe with 4 new columns: 'summary_lem', 'question_lem', 'answer_lem', 'dp_answer_lem'
    Commented, as there was tokenization error of some latin phrases with this function version
    #TODO: PERHAPS CHANGE IT FROM PYMORPHY2 TO NATASHA AS A WRAPPER
    df['summary_lem'] = df.apply(lambda row: [morph.parse(t)[0].normal_form for t in nltk.word_tokenize(clean_text(row['summary']))], axis=1) #added clean_text in June 2021, re-editing lemmas lists match
    df['question_lem'] = df.apply(lambda row: [morph.parse(t)[0].normal_form for t in nltk.word_tokenize(clean_text(row['question']))], axis=1)
    df['answer_lem'] = df.apply(lambda row: [morph.parse(t)[0].normal_form for t in nltk.word_tokenize(clean_text(row['answer']))], axis=1)
    df['dp_answer_lem'] = df.apply(lambda row: [morph.parse(t)[0].normal_form for t in nltk.word_tokenize(clean_text(row['dp_answer'][0]))], axis=1)     
    return df
"""    

def check_pronouns(lem_text):
    """
    Check if there is more than one pronoun for question in a question, for a row
    """
    pronouns = ['кто', 'что', 'какой', 'чей', 'где', 'который', 'откуда', 
                'сколько', 'каковой', 'каков', 'зачем', 'когда']    
    ind = 0
    for lemma in lem_text:
        if lemma in pronouns:
            ind += 1
    if ind > 1:
        return 'many'
    return 'one_or_no'

def find_persons(text):
    """
    Find person mentions in text, with natasha library
    """
    try:
        doc = Doc(text) 
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)
        doc.tag_ner(ner_tagger)
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
        for span in doc.spans:
            span.normalize(morph_vocab)
        for span in doc.spans:
            if span.type == 'PER':
                span.extract_fact(names_extractor)
        pers_dict = {_.normal: _.fact.as_dict for _ in doc.spans if _.fact} 
    except:
        pers_dict = {}
    return pers_dict

def find_locations(text):
    """
    Find normalized location mentions in text, with natasha library
    """
    locations = []
    try:
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)
        doc.tag_ner(ner_tagger)
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
        for span in doc.spans:
            span.normalize(morph_vocab)        
        for span in doc.spans:
            if span.type == 'LOC':
                locations.append(span.normal)
        locations = list(set(locations))
    except:
        return locations
    return locations   

def find_all_key_entities(text):
    """
    Find all key entity mentions in text, including locations, persons, and other types
    """
    all_entities = []
    try:
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)
        doc.tag_ner(ner_tagger)
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
        for span in doc.spans:
            span.normalize(morph_vocab)
        for span in doc.spans:            
            all_entities.append(span.normal)
        all_entities = list(set(all_entities))
    except: 
        return all_entities
    return all_entities

def clean_text(text):
    """
    Fix spare white spaces and punctuation
    """
    clean_text = " ".join(text.split())
    exclude = set(string.punctuation)
    clean_text = "".join(ch for ch in clean_text if ch not in exclude)        
    return clean_text       

def language_metrics(dp_answer, answer, question, text):
    """
    Get scores for METEOR, BLEU, ROUGE-L metrics
    All metrics are done for lemmatized strings.
    For pairs reference-hypothesis:
    deeppavlov answer - GPT answer,
    question - GPT answer,
    text - GPT answer,
    text - question    
    """
    def get_meteor(ref, hyp):
        lem_ref = " ".join(get_tokens_lemmas(ref)[1]).lower()
        lem_hyp = " ".join(get_tokens_lemmas(hyp)[1]).lower()    
        meteor = round(meteor_score([lem_ref], lem_hyp), 2)
        return meteor

    def get_sentences(document):
        sentences = []
        doc = Doc(document)
        doc.segment(segmenter)
        for sentence in doc.sents:
            sentences.append(get_tokens_lemmas(sentence.text.lower())[1])
        return sentences

    def get_bleu(ref, hyp):
        ref = get_sentences(ref)
        hyp = get_sentences(hyp)
        list_of_hyp = [[x for sublist in hyp for x in sublist]]    
        list_of_references = [ref]
        smt = SmoothingFunction().method3    
        bleu = round(corpus_bleu(list_of_references, list_of_hyp, smoothing_function = smt), 2) 
        return bleu

    def get_rouge(hyp, ref): #handle each of them as one sentence
        hypothesis = " ".join(get_tokens_lemmas(hyp)[1]).lower()
        reference = " ".join(get_tokens_lemmas(ref)[1]).lower()
        scores = rouge.get_scores(hypothesis, reference)
        return round(scores[0]['rouge-l']['f'], 2)

    dp_answer_answer_metrics = (get_meteor(dp_answer, answer) + get_bleu(dp_answer, answer) + get_rouge(answer, dp_answer)) / 3
    question_answer_metrics = (get_meteor(question, answer) + get_bleu(question, answer) + get_rouge(answer, question)) / 3
    text_answer_metrics = (get_meteor(text, answer) + get_bleu(text, answer) + get_rouge(text, answer)) / 3
    text_question_metrics = (get_meteor(text, question) + get_bleu(text, question) + get_rouge(question, text)) / 3
    return dp_answer_answer_metrics, question_answer_metrics, text_answer_metrics, text_question_metrics
    
    
        
        
