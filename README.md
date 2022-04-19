Generative QA corpus on the whole Russian Wikipedia

The full corpus is available on request (dinabpr@gmail.com, rybolos@gmail.com).

The provided docker for filtration can be used to filter one file with Wikipedia summaries and generated questions and answers for them, leaving only good quality examples. 
The automated heuristic approach is used.

After running docker for filtration (filtering_rus) it is possible to run filtration for a pandas dataframe that should contain columns 'summary', 'question', and 'answer'.
Its name should be like: {model_name}_batch{number}_all.csv
Usage example: t5_batch1_all.csv 
This file should be put in the following folder: data/in/

The file will be processed in fragments (each fragment = 1000 examples).
They will be saved in the following folder: data/out/

To run filtration: ./run.sh {batch_number}
Usage example: ./run.sh 1

In the current implementation, the following optimal filtration function run "from the box" and are debugged:
- remove triplets with more than one interrogative pronoun in the question;
- squad ru rubert infer BERT model for the Russian Language, trained on SberQuad, creates gold answers for the questions in the dataset; after that, only examples with exact match between the lemmatized answer and the lemmatized gold answer over than 70%, are left;
- remove examples in which named entities (of different types) in the question are not presented in the corresponding Wikipedia summary, and/or named entities (of different types) in the answer are not presented in the summary, using string match methods;
- remove duplicated examples for the same summaries where Levenshtein distance similarity ratio between questions and Levenshtein distance similarity ratio between answers is more than 70%.

Please pay attention that other parameters and checks in do_filter() function (i.e. additional_checks, metrics) were used only in experiments and yielded bad quality, that's why they are not fully debugged and may cause errors in some cases. 





