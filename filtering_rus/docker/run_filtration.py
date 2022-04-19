import os
from tempfile import TemporaryDirectory
from typing import Union, List, Tuple, Iterable, Optional

import gensim
import pandas as pd
import tensorflow as tf

#In order to get scores instead of logits, changes as described here https://forum.deeppavlov.ai/t/how-to-generate-confidence-scores-uisng-configs-squad-squad-bert-infer-model/231/2 should be done; in this case, not in squad_ru_rubert_infer.json config file, but in squad_ru_rubert.json file 
from deeppavlov import build_model, configs
#!pip install fuzzywuzzy[speedup]
from fuzzywuzzy import fuzz
from gensim.similarities import WmdSimilarity

#from local files
import nlp_helpers as nhelpers
from language_checker import CheckRussian

MODEL_FILES = (
    'araneum_none_fasttextcbow_300_5_2018.model',
    'araneum_none_fasttextcbow_300_5_2018.model.vectors.npy',
    'araneum_none_fasttextcbow_300_5_2018.model.vectors_ngrams.npy',
    'araneum_none_fasttextcbow_300_5_2018.model.vectors_vocab.npy',
)
WORK_DIR = '/home/jovyan/'
S3_PREFIX = 'general_fasttext_2018'


class TextQA(dict):
    """summary (text of Wikipedia summary), generated question, generated answer, deeppavlov answer"""

    def __init__(self, summary: str, question: str, answer: str, dp_answer: str):
        dict.__init__(self, summary=summary, question=question, answer=answer, dp_answer=dp_answer)

    @property
    def summary(self):
        return self.get('summary')

    @property
    def question(self):
        return self.get('question')

    @property
    def answer(self):
        return self.get('answer')

    @property
    def dp_answer(self):
        return self.get('dp_answer')


class Filter:
    def __str__(self) -> str:
        return self.__class__.__name__ + '_filter'

    def do_filter(self, documents: Iterable[str]) -> List[TextQA]:
        """ Return only good quality QA results from a list of input data."""        
        return results


class RulebasedFilter(Filter):
    def __init__(self, language_checker: CheckRussian=None):
        """
        Base class for Rule-Based Wikipedia filter that needs language checking for the 1st text part, to avoid bad QA pairs
        """
        self.language_checker = language_checker

    def is_language(self, text: str) -> bool:
        if self.language_checker is None:
            raise AttributeError("Object must be instantiated with a LanguageChecker to call is_language().")
        return self.language_checker.is_language(text)


class HeuristicsFilter(RulebasedFilter):

    text_size = (50, 2000)
    lemmas_threshold = 70
    sim_threshold = (1.1, 1.5)
    dp_threshold = 0.99 #not 1; it may cause false positives, but the aim is to decrease number of false negatives
    
    def __init__(self, language_checker: CheckRussian=None, load_gensim=False):
        """
        Initializes HeuristicsFilter based on heuristics from filtering experiments
        """        
        super().__init__(language_checker)
        #download araneum_none_fasttextcbow_300_5_2018 fasttext model files from aicloud S3 bucket folder
        if load_gensim:
            with TemporaryDirectory(prefix=WORK_DIR) as model_dir: 
                for file in MODEL_FILES:
                    job = client_lib.S3CopyJob(f's3://{AWS_BUCKET}/{S3_PREFIX}/{file}', os.path.join(model_dir, file))
                    job.submit()
                    job.wait()
                # fasttext based gensim model init:
                model_path = os.path.join(model_dir, MODEL_FILES[0])
                self.gensim_model = gensim.models.KeyedVectors.load(model_path)
        else:
            self.gensim_model = None

        #deeppavlov squad ru ru_bert model init
        try:
            self.deeppavlov_model = build_model(configs.squad.squad_ru_rubert_infer, download=True)
        except ImportError:
            import subprocess as sp
            sp.run(('python', '-m', 'deeppavlov', 'install', 'squad_ru_rubert_infer'))
            self.deeppavlov_model = build_model(configs.squad.squad_ru_rubert_infer, download=True)#TODO: make download from the local folder so scores are got, for ru and en

    def is_filter(self) -> bool:
        return True

    def get_dp_answer(self, qa_df)-> pd.DataFrame:
        """
        Takes pandas dataframe with 'summary', 'question', 'answer' columns.
        Returns pandas dataframe with added 'dp_answer' and 'dp_score' columns for deeppavlov answer based on 'question' and deeppavlov score
        """        
        answers, scores = ([] for i in range(2))
        indices_to_remove = []
        for i, row in qa_df.iterrows():
            try:
                full_answer = self.deeppavlov_model([row['summary']], [row['question']])
                answers.append(full_answer[0])
                scores.append(full_answer[2])
            except:
                print(row['summary'], row['question'])
                indices_to_remove.append(i)
                answers.append('NO_DP_ANSWER')
                scores.append('NO_DP_ANSWER')        
        qa_df['dp_answer'] = answers
        qa_df['dp_score'] = scores 
        qa_df = qa_df[~qa_df.index.isin(indices_to_remove)]
        return qa_df  

    def filter_for_final_functions(self, qa_df)-> pd.DataFrame:
        """
        Handle possible errors that could arise while generating QA
        Takes pandas dataframe
        Returns pandas dataframe without rows with errors
        """
        qa_df = qa_df.dropna() #to avoid NaNs possibly got by question answering models
        bad_indices = []
        for i, row in qa_df.iterrows():
            gpt_answer = nhelpers.clean_text(row['answer']) # turn cases like ' ' to '' so it can be handled in this function
            #dp_answer = row['dp_answer'][2:-2]
            dp_answer = nhelpers.clean_text(row['dp_answer'][0])
            question = nhelpers.clean_text(row['question'])
            if len(gpt_answer) == 0 or type(gpt_answer) != str: # added for handling errors
                bad_indices.append(i)
            if len(dp_answer) == 0 or type(dp_answer) != str:  # added for handling errors     
                if i not in bad_indices:
                    bad_indices.append(i)
            if len(question) == 0 or type(question) != str: # added for handling errors
                if i not in bad_indices:
                    bad_indices.append(i)
        qa_df = qa_df[~qa_df.index.isin(bad_indices)]  
        return qa_df 

    def count_pronouns(self, qa_df)-> pd.DataFrame:
        """
        Takes df 
        Returns df after filtering: not more than one pronoun for question, in the question
        """
        pron_indices = []
        for i, row in qa_df.iterrows():
            pron_number = nhelpers.check_pronouns(row['question_lem'])
            if pron_number != 'many':
                pron_indices.append(i)
        qa_df = qa_df[qa_df.index.isin(pron_indices)]     
        return qa_df

    def avoid_spare_persons(self, qa_df)-> List[int]:
        """
        Takes df (with definite column names)
        Removes rows in df where persons in generated question are not presented in summary (text)
        or persons in generated answer are not presented in summary (text)
        Returns list of 'bad' indices
        """    
        pers_indices = []
        for i, row in qa_df.iterrows():
            text = row['summary']
            gpt_answer = row['answer']
            question = row['question']
            pers_text = nhelpers.find_persons(text)
            pers_question = nhelpers.find_persons(question)
            pers_answer = nhelpers.find_persons(gpt_answer)
            for k in pers_question:
                #only full intersections, to avoid cases like Albert Camus in Q&A and Albert Einstein in summary (text)
                if k not in pers_text:
                    pers_indices.append(i)
            for k in pers_answer:
                if k not in pers_text:
                    pers_indices.append(i)            
        bad_pers_indices = list(set(pers_indices))   
        return bad_pers_indices

    def avoid_spare_locations(self, qa_df)-> List[int]:
        """
        Takes df (with definite column names)
        Finds rows in df where locations in question are not presented in summary (text)
        or locations in generated answer are not presented in summary (text)
        Returns list of 'bad' indices
        """
        loc_indices = []
        for i, row in qa_df.iterrows():
            text = row['summary']
            gpt_answer = row['answer']
            question = row['question']
            loc_text = nhelpers.find_locations(text)
            loc_question = nhelpers.find_locations(question)
            loc_answer = nhelpers.find_locations(gpt_answer)
            for k in loc_question:
                #only full intersections
                if k not in loc_text:
                    loc_indices.append(i)
            for k in loc_answer:
                if k not in loc_text:
                    loc_indices.append(i)            
        bad_loc_indices = list(set(loc_indices))        
        return bad_loc_indices

    def avoid_different_entities(self, qa_df)-> List[int]:
        """
        Takes df (with definite column names)
        Finds rows in df where entities (of different types) in generated question are not presented in summary (text)
        or entities (of different types) in generated answer are not presented in summary (text)   
        Returns list of 'bad' indices
        """ 
        entity_indices = []
        for i, row in qa_df.iterrows():
            text = row['summary']
            gpt_answer = row['answer']
            question = row['question']
            entity_text = nhelpers.find_all_key_entities(text)
            entity_question = nhelpers.find_all_key_entities(question)
            entity_answer = nhelpers.find_all_key_entities(gpt_answer)
            for k in entity_question:
                #only full intersections
                if k not in entity_text:
                    entity_indices.append(i)
            for k in entity_answer:
                if k not in entity_text:
                    entity_indices.append(i)            
        bad_entity_indices = list(set(entity_indices))         
        return bad_entity_indices

    def check_similarity(self, qa_df, threshold: Tuple[float, float])-> List[int]:
        """
        Takes pandas dataframe,
        checks semantic similarity of answers using fasttext araneum model and similarity Word Mover's Distance measure    
        Returns list of indices where rows correspond to the given threshold
        Threshold chosen by default is not less than 1.1 and not more than 1.5
        """
        indices_sim = []    
        for i, row in qa_df.iterrows():
            gpt_answer = row['answer'].lower()            
            dp_answer = row['dp_answer'][0].lower()
            #TODO: better preprocessing (fasttext)
            wmd_similarity = self.gensim_model.wmdistance(gpt_answer, dp_answer) 
            if threshold[0] <= wmd_similarity <= threshold[1] and wmd_similarity != float("inf"): #inf if stated if words are out-of-vocab
                indices_sim.append((i, wmd_similarity))           
        return indices_sim

    def filter_dp_scores(self, qa_df, threshold: float)-> List[int]:
        """
        Takes dataframe with column 'dp_score'
        Return indices where its value is >= the given threshold
        """
        index = qa_df.index
        conf_indices = index[qa_df['dp_score'] >= threshold].tolist() 
        return conf_indices  

    def match_lemstrings(self, qa_df)-> List[int]:
        """
        Takes pandas dataframe with definite columns
        Looks for exact lemmatized string match between generated answer and deeppavlov answer
        Returns list of indices for such rows
        """
        indices_joined = []   
        for i, row in qa_df.iterrows():                 
            gpt_lemmas = row['answer_lem']
            gpt_lemmas_joined = " ".join(gpt_lemmas)
            #fix white spaces and punctuation
            gpt_lemmas_joined = nhelpers.clean_text(gpt_lemmas_joined)
            dp_lemmas = row['dp_answer_lem']
            dp_lemmas_joined = " ".join(dp_lemmas)  
            #fix white spaces and punctuation
            dp_lemmas_joined = nhelpers.clean_text(dp_lemmas_joined)         
            if gpt_lemmas_joined in dp_lemmas_joined or dp_lemmas_joined in gpt_lemmas_joined:
                indices_joined.append(i)
        return indices_joined

    def match_lemmas(self, qa_df, lemmas_threshold: int)-> List[int]:
        """
        Takes pandas dataframe with definite columns
        Looks for intersection of lemmas between generated answer and deeppavlov answer
        Returns list of indices for such rows
        """    
        indices_inters = []
        for i, row in qa_df.iterrows():
            gpt_lemmas = row['answer_lem']
            dp_lemmas = row['dp_answer_lem']
            setA = set(gpt_lemmas)
            setB = set(dp_lemmas)
            overlap = setA & setB
            universe = setA | setB
            try:
                result1 = float(len(overlap)) / len(setA) * 100
                result2 = float(len(overlap)) / len(setB) * 100
                intersect_first = float(len(overlap)) / len(universe) * 100            
                if intersect_first >= lemmas_threshold: #threshold chosen by default is 70
                    indices_inters.append(i)   
            except:
                continue     
        return indices_inters

    def get_language_metrics(self, qa_df)-> List[int]:
        """
        Takes pandas dataframe with definite columns
        For each row, ROUGE, METEOR and BLEU metrics are counted for pairs reference-hypothesis:
        deeppavlov answer - generated answer,
        question - generated answer,
        text - generated answer,
        text - generated question     
        Returns list of indices for rows where united metrics scores are higher than thresholds
        This function can be optionally used instead of combination of match lemstrings and match_lemmas.
        All metrics are done for lemmatized strings.
        """  
        good_indices = []
        for i, row in qa_df.iterrows():
            dp_answer = row['dp_answer'][0]               
            answer = row['answer']
            question = row['question']
            text = row['summary'] 
            dp_answer_answer_metrics, question_answer_metrics, text_answer_metrics, text_question_metrics = nhelpers.language_metrics(dp_answer, answer, question, text) 
            if dp_answer_answer_metrics >= 0.6 and question_answer_metrics >= 0.5 and text_answer_metrics >= 0.4 and text_question_metrics >= 0.4: #TODO: thresholds may be tuned during usage; add them to params as other thresholds
                good_indices.append(i)
        return good_indices

    def drop_bad_examples(self, qa_df)-> pd.DataFrame:
        """
        Takes pandas dataframe,
        Removes rows with disambiguation wiki page categories,
        Removes rows with age questions,
        Removes references from the end of the summary if they are there,   
        Returns pandas dataframe to give it to the preliminary filtering function
        """
        bad_cat = ['Страницы значений:Однофамильцы-тёзки', 'Категория:Страницы значений по алфавиту', 'Страницы значений:Однофамильцы']
        bad_indices = []               
        for i, row in qa_df.iterrows():
            document = row['summary']
            question = row['question']
            categories = row['category']           
            if 'сколько лет' in question or 'Сколько лет' in question:
                bad_indices.append(i)
            for elem in bad_cat:        
                if elem in categories: 
                    bad_indices.append(i)
        bad_indices = list(set(bad_indices))
        qa_df_strict = qa_df[~qa_df.index.isin(bad_indices)]   
        return qa_df_strict

    def prelim_filter(self, documents: Iterable[str], text_size: Optional[Tuple[int, int]] = None, lang_filter: bool = True)-> List[str]:
        """
        Takes list of documents,
        Preliminary filter: removes too short summaries (<50 characters),
        removes summaries with too many term translations in the beginning,
        Returns list of documents for asking generated QA
        """
        if not text_size:
            text_size = self.text_size
        prelim_filtered_docs = []        
        for document in documents:
            document_size = len(document)
            if text_size[0] < document_size < text_size[1]:
                if lang_filter and self.is_language(document):
                    prelim_filtered_docs.append(document)
        return prelim_filtered_docs

    def prelim_filter_df(self, qa_df, text_size: Optional[Tuple[int, int]] = None, lang_filter: bool = True)-> pd.DataFrame:
        """
        Takes pandas dataframe,
        Preliminary filter: removes rows with too short summaries (<50 characters),
        removes rows with summaries with too many term translations in the beginning,
        Returns pandas datafrale to give it to the existing do_filter (given that we already have df with generated answers or without them)
        """
        #TODO: REMOVE IT AFTER REFACTORING: PRELIM_FILTER (ABOVE) SHOULD BE INSTEAD OF PRELIM_FILTER_DF
        if not text_size:
            text_size = self.text_size 
        good_indices = []               
        for i, row in qa_df.iterrows():
            document = row['summary']
            document_size = len(document)
            if text_size[0] < document_size < text_size[1]:
                if lang_filter and self.is_language(document):
                    good_indices.append(i)
        qa_df_edited = qa_df[qa_df.index.isin(good_indices)]   
        return qa_df_edited   

    def remove_duplicated_qa(self, dframe)-> pd.DataFrame:
        """
        Takes a dataframe (after the final filtration stage)
        It there are almost similar questions and almost similar answers for the same summary,
        only one instance is left. There were such cases for Russian. 
        Returns cleaned pandas dataframe
        """
        df_dup = dframe[dframe.duplicated(['summary'], keep=False)].sort_values("summary")
        df_dup.reset_index(drop=True, inplace=True)
        indices_to_remove = []
        for i, row in df_dup.iterrows():
            if row['summary'] == df_dup.iloc[i-1]['summary']:
                question_r = fuzz.ratio(row['question'], df_dup.iloc[i-1]['question'])
                answer_r = fuzz.ratio(row['answer'], df_dup.iloc[i-1]['answer'])
                if question_r > 70 and answer_r > 70:
                    indices_to_remove.append(row['index'])
        new_dframe = dframe[~dframe.index.isin(indices_to_remove)]
        return new_dframe                            

    def do_filter(self, qa_df, lemmas_threshold: Optional[int] = None, sim_threshold: Optional[Tuple[float, float]] = None, dp_threshold: Optional[float] = None, sep_keywords: bool = False, metrics: bool = False, additional_checks: bool = False) -> List[TextQA]:
        """
        Main filtration function
        :param qa_df: dataframe with wikipedia 'summary' column, generated question ('question') column, generated answer ('answer') column
        :param lemmas_threshold: percent of intersections between lists of lemmas for generated and deeppavlov answers
        :param sim_threshold: Min and max values for WMD similarity measure between lists of lemmas for generated and deeppavlov answers
        :param dp_threshold: Min deeppavlov QA model score (is not useful, in general)           
        :param sep_keywords: To use only persons and locations, or to use all named entities as keywords (by default)
        :param metrics: To use get_language_metrics instead of combination of match_lemstrings and match_lemmas
        :param additional_checks: To check WMD similarity and deeppavlov QA model score or not
        :returns: results: dataframe with added columns (for every text: deeppavlov answer ('dp_answer'), lemmatized summary ('summary_lem'), 
        lemmatized generated question ('question_lem'), lemmatized generated answer ('answer_lem'), deeppavlov lemmatized answer ('dp_answer_lem')
        The returned dataframe is filtered and should be smaller than the initial dataframe.
        """      
        
        #if not text_size:
        #    text_size = self.text_size
        if not lemmas_threshold:
            lemmas_threshold = self.lemmas_threshold
        if not sim_threshold:
            sim_threshold = self.sim_threshold
        if not dp_threshold:
            dp_threshold = self.dp_threshold        
        #preliminary filter: removes too short summaries (<50 characters),
        #removes summaries with too many term translations in the beginning.
        #is not needed for Wikipedia: all texts are already in Russian and we know it.

        #prelim_filtered_docs = []        
        #for document in documents:
        #    document_size = len(document.split())
        #    if text_size[0] < document_size < text_size[1]:
        #        if lang_filter and self.is_language(document):
        #            prelim_filtered_docs.append(document)
  
        #qa_df = self.get_gpt_qa(prelim_filtered_docs)
        qa_df = self.get_dp_answer(qa_df) #here we also filter our rows where generated Q or A = nan, as nan filtering is next step: to do refactoring
        qa_df.reset_index(drop=True, inplace=True)   
        qa_df = self.filter_for_final_functions(qa_df) 
        qa_df.reset_index(drop=True, inplace=True)   
        print('after_first_filter: ', qa_df.shape)
        #choose 2 lemmas matching functions (by default) or choose metrics function for lemmas matching
        qa_df = nhelpers.get_lemmas_for_df(qa_df)
        qa_df.reset_index(drop=True, inplace=True)   
        qa_df = self.count_pronouns(qa_df)
        qa_df.reset_index(drop=True, inplace=True)   
        print('after_pronouns: ', qa_df.shape)
        if metrics:
            chosen_indices = self.get_language_metrics(qa_df)
        else:
            indices_lemstrings = self.match_lemstrings(qa_df) #function for strict intersection of non-lemmatized strings was not added, as this one is better
            indices_lemmas = self.match_lemmas(qa_df, lemmas_threshold)
            #Filter dataframe using both united indices lists
            chosen_indices = list(set(indices_lemstrings + indices_lemmas))        
        qa_df = qa_df[qa_df.index.isin(chosen_indices)]
        qa_df.reset_index(drop=True, inplace=True)  
        print('after_lemmas: ', qa_df.shape)         
        if sep_keywords:
            bad_indices = list(set(self.avoid_spare_persons(qa_df) + self.avoid_spare_locations(qa_df)))
        else:            
            bad_indices = self.avoid_different_entities(qa_df)
        print('before_entities: ', qa_df.shape)
        qa_df = qa_df[~qa_df.index.isin(bad_indices)]
        print('after_entities: ', qa_df.shape)
        qa_df.reset_index(drop=True, inplace=True)   
        if additional_checks:
            indices_sim = self.check_similarity(qa_df, sim_threshold)        
            conf_indices = self.filter_dp_scores(qa_df, dp_threshold)
            additional_indices = list(set(indices_sim + conf_indices))
            qa_df = qa_df[qa_df.index.isin(additional_indices)] 
            qa_df.reset_index(drop=True, inplace=True)   

        qa_df = self.remove_duplicated_qa(qa_df)
        print('after_removing_duplicates: ', qa_df.shape)
        return qa_df
