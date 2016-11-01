

import pprint
import re
import string
import sys
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')
from gensim.parsing.preprocessing import STOPWORDS
STOPWORDS = list(set(STOPWORDS))

hand_picked_stop_words = ['rt' , "it's" , 'says' , "doesn't" , "shes" ,"hes" , "she's" ,"he's", u"don't" , "thanks" , "thank's", "like" ,
                         "today" , "time", "know" , "knows" , "help" , "check" , "good", 'must', 'back', 'service' ,
                         'trust', 'yesterday' , 'before' , 'away', 'products', "we're", "bad", "its" ,"it's" , "like" ,
                         'tell', 'talk' , 'wait', 'think','thinks','00pm',"jr's", 'truth','want','wants', 'give','gave',
                         'sure', 'edit', 'may', 'maybe', 'may not', 'might','might not', "we've", 'able', 'go', 'goes',
                         'went', "what's", 'list', 'lists', "can't", 'forever', 'ever', 'says', 'item', "we'd", '#deporthillary',
                         'woud', 'will', 'would', 'mmmmk', "t'was", "ira's", 'sehe', 'haaa', "l'art", 'spss', "bryan's"]

def translate_non_alphanumerics(to_translate, translate_to=u'_'):
    not_letters_or_digits = u'!"#%\'()&*+,-/:;<=>?[\]^_`{|}~'
    if isinstance(to_translate, unicode):
        translate_table = dict((ord(char), unicode(translate_to))
                               for char in not_letters_or_digits)
    else:
        assert isinstance(to_translate, str)
        translate_table = string.maketrans(not_letters_or_digits,
                                           translate_to
                                              *len(not_letters_or_digits))
    return to_translate.translate(translate_table)



def length_check(  word):
        if '_' in word:
            return word
        else:
            if len(word) >= 17:
                return None
            else:
                return word 

def pre_process_sentence (  sentence ):

    sentence = sentence.lower()
    try:
        sent = sentence.encode( 'ascii' , 'ignore')
    except:
        sent = sentence.decode( 'ascii', 'ignore')
    # sent = re.sub(ur'\b(\d+\s+)', '', sent)
    sent = translate_non_alphanumerics( sent, ' ')   
    sent = re.sub(ur"\s+"," ", sent)
    sent = re.sub(ur'\b(\s+\d+\s+)', '', sent)
    sent = re.split(ur'\.\s+', sent)
    text = []
    for text_ in sent:
        text.extend(text_.split())
    text = [word.strip() for word in text if len(word) >=2 and not word.isdigit() and length_check(word) and word not in STOPWORDS ]
    return text



if __name__ == "__main__":
	
	print "Start"
	print pre_process_sentence('Machine learning testong jsswasxa')