from langid.langid import LanguageIdentifier
from langid.langid import model as lang_ident


class CheckRussian:
    """
    Check if a text is in Russian.
    (Only for first 100 characters of text: 1) in order to make it faster 2) not Russian terms are usually in the beginning).
    (Because summaries with too many term translations in the beginning should be later removed).    
    """

    def __init__(self, language: str='ru'):
        self.language = language        
        self.language_identifier = LanguageIdentifier.from_modelstring(lang_ident, norm_probs=True)
        self.check_params()    

    def is_language(self, text: str) -> bool:
        """
        Check if a text is in Russian
        """
        lang, prob = self.language_identifier.classify(text[:100])
        return lang == self.language

    def check_params(self):
        pass  # FixMe
