# -*- coding:utf-8 -*-

# l'annotateur entrainé avec performance basé sur le perceptron.
 
import os
import random
from collections import defaultdict
import pickle
import logging
 
from AP_algorithm import AveragedPerceptron
 
PICKLE = "data/tagger-0.1.0.pickle"   # le lieu à stocker le model généré
 
class PerceptronTagger():
 
    START = ['-START-', '-START2-']
    END = ['-END-', '-END2-']
    AP_MODEL_LOC = os.path.join(os.path.dirname('__file__'), PICKLE)  #le path à stocker le model 
 
    def __init__(self, load=True):
        self.model = AveragedPerceptron()
        self.tagdict = {}
        self.classes = set()
        if load:
            self.load(self.AP_MODEL_LOC)
            
            
    '''Etape 2
    Lire les mots du corpus. 
    le "." présente la fin d'une phrase, un certain nombre de mots qui leprécèdent sont transformés en une phrase, 
    qui est stockée dans un groupe binaire : par exemple (["Comment", "Allez"], [ adv', 'v'])

    la première liste est celle des mots de la phrase.

    La deuxième liste est celle des PoS correspondant aux mots. 
    Toutes les phrases du corpus sont stockées dans la liste training_data(phrases), 
    sous la formes comme [ ([ ], [ ]), ([ ], [ ]), ([ ], [ ]).
    '''
 
    def tag(self, corpus): 
# il nous renverra un list qui enregistre des mots et pos comme [(mot1, pos1), (mot2, pos2),(mot3,pos3)...], 
 
        s_split = lambda t: t.split('\n') #les phrases ont séparées par "\n", les mots ont séparés par" "
        w_split = lambda s: s.split()
 
        def split_sents(corpus):
            for s in s_split(corpus):
                yield w_split(s)  
# cette fonction contient la méthode yield, càd qu'il est un générateur itératable avec une valeur renvoyée
 
        prev, prev2 = self.START
        tokens = []  # stocker les pos
        for words in split_sents(corpus):
            context = self.START + [self._normalize(w) for w in words] + self.END  
        # context est une phrase avec des catactéristiques spécifiques ajoutés
            for i, word in enumerate(words): 
                tag = self.tagdict.get(word)  #acquérir les pos à partir des dics.
                if not tag:
                    features = self._get_features(i, word, context, prev, prev2)
                    tag = self.model.predict(features)
                tokens.append((word, tag))
                prev2 = prev
                prev = tag
        return tokens  # comme [(mot1, PoS1),(mot2, PoS2)...]
    
    
    
    ''' Etape3
    lors de l'entraînement du modèle, 
    les pondérations sont mises à jour en :
      +1 aux pondérations des caractéristiques correspondantes(caractéristiques correstes)
      -1 aux pondérations des caractéristiques correspondantes(caractéristiques fausses)

    Par conséquent : 
        non seulement les poids correspondant aux PoS corrects sont augmentés, 
        mais aussi les poids correspondant aux PoS fausses sont pénalisés.
    '''
    
 
    def train(self, sentences, save_loc=None, nr_iter=5):  # entrainer le model
        '''
        paramètre sentences : un liste contient le (mots, pos)
        paramètre save_loc : un lieu à stocker le model
        paramètre nr_iter : entrainer le nombre de l'itération
        '''
        self._make_tagdict(sentences)
        self.model.classes = self.classes  # classes comme set('adj','n','vb'...)
        for iter_ in range(nr_iter):
            c = 0
            n = 0
            for words, tags in sentences:
                prev, prev2 = self.START
                context = self.START + [self._normalize(w) for w in words] \
                          + self.END
                for i, word in enumerate(words):
                    guess = self.tagdict.get(word)  
#  acquérir des pos à partir des dics, sinon on utilise les pos prédits par des caractéristique
                    if not guess:
                        feats = self._get_features(i, word, context, prev, prev2)  
    # extraire des caractéristique tout d'abord
                        guess = self.model.predict(feats)  
    # prédire des POS par des caractéristiques extraits
                        self.model.update(tags[i], guess, feats)  # mis à jour les poids
                    prev2 = prev  # i-2 mot
                    prev = guess  # i-1 mot
                    c += guess == tags[i]  # si la prédiction est correcte : c+=1
                    n += 1
            random.shuffle(sentences)
            logging.info("Iter {0}: {1}/{2}={3}".format(iter_, c, n, _pc(c, n))) 
        self.model.average_weights()  # calculer la valeur moyenne
 
        if save_loc is not None:
            pickle.dump((self.model.weights, self.tagdict, self.classes),
                        open(save_loc, 'wb'), -1)  # paramètre - 1
        return None
 
    def load(self, loc):  # loader le model
 
        try:
            w_td_c = pickle.load(open(loc, 'rb'))
        except IOError:
            msg = ("Missing trontagger.pickle file.")
            raise IOError(msg)
        self.model.weights, self.tagdict, self.classes = w_td_c
        self.model.classes = self.classes
        return None
 

    ''' Etape 4

    (les données ont été prétraitées avant l'extraction des caractéristiques)

    Afin de former un modèle plus général : 
        a) Tous les mots ont été convertis en minuscules
        b) Les numéros à quatre chiffres et entre 1800 et 2100 ont été convertis en '!YEAR'
        c) Les autres chiffres ont convertis en '8DIGITS'
        d) Bien sûr qu'il existe d'autre cas, mais pour l'instant il ne sont pas étendue
    '''
    def _normalize(self, word):  
# pré-traitement des chaines de caractère, mettre des mots en minuscule, considérer les chiffres séparément
 
        if '-' in word and word[0] != '-':
            return '!HYPHEN' # signe '-'
        elif word.isdigit() and len(word) == 4:
            return '!YEAR' 
        elif word[0].isdigit():
            return '!DIGITS'
        else:
            return word.lower()

        
        
    ''' Etape5

    Extraction de caractéristiques pour le i ème mot :
        La première lettre du mot
        Le suffixe du mot
        Le préffixe du mot
        La PoS du (i-n) mot
        La PoS du (i+n) mot
        La suffixe du (i-n) ème mot
        La suffixe du (i+n) ème mot
            ...
            
    '''
    def _get_features(self, i, word, context, prev, prev2): 
# extraire les caractéristiques des mots, renvoyer un dic de features
        def add(name, *args):  # lier les chaine de caractères, pour nommer les features
            features[' '.join((name,) + tuple(args))] += 1
 
        i += len(self.START)
        features = defaultdict(int)
# personnaliser des features comme : préfixes, suffixes, mot précédent, mot suivant...
        add('bias')
        add('i suffix', word[-3:])  #le suffixe du mot
        add('i pref1', word[0])     #la première lettre du mot
        add('i-1 tag', prev)        #la nature de mot i-1
        add('i-2 tag', prev2)       #la nature de mot i-2
        add('i tag+i-2 tag', prev, prev2)
        add('i word', context[i])
        add('i-1 tag+i word', prev, context[i])
        add('i-1 word', context[i - 1])
        add('i-1 suffix', context[i - 1][-3:])
        add('i-2 word', context[i - 2])
        add('i+1 word', context[i + 1])
        add('i+1 suffix', context[i + 1][-3:])
        add('i+2 word', context[i + 2])
        return features
 
    def _make_tagdict(self, sentences):   
# faire un 'tagdic' comme {'remettre':'v','projet':'n'...}
        counts = defaultdict(lambda: defaultdict(int))
        for words, tags in sentences:  # mots comme ['bon','homme']; tags comme ['adj','n']
            for word, tag in zip(words, tags):  # mots comme'bon', tag comme 'adj'
                counts[word][tag] += 1
                self.classes.add(tag)
        freq_thresh = 20
        ambiguity_thresh = 0.97
        for word, tag_freqs in counts.items():
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            # paramétrer un seuil pour enregistrer des POS fréquents
            if n >= freq_thresh and (float(mode) / n) >= ambiguity_thresh:
                self.tagdict[word] = tag
                
def _pc(n, d):   # probabilité de la précision
    return (float(n) / d) * 100
    print((float(n) / d))
    
    
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) 
#implémenter la façon / le forme de l'exportation par la foncito logging.basicConfig
    tagger = PerceptronTagger(False)
    try:
        tagger.load(PICKLE)
        print(tagger.tag("Comment allez-vous ?"))
        logging.info('Start testing...')
        right = 0.0
        total = 0.0
        sentence = ([], [])
        for line in open('data/DSG_French/my_gsd_test.txt'):
            params = line.split()
            if len(params) != 2: continue
            sentence[0].append(params[0])  # mots
            sentence[1].append(params[1])  # POS
            if params[0] == '.': 
# le point indique la fin de phrase, combiner les mots en une phrase reliée par des espaces 
                text = ''
                words = sentence[0]
                tags = sentence[1]
                for i, word in enumerate(words):
                    text += word
                    if i < len(words): text += ' '
                outputs = tagger.tag(text)  # outputs comme[(mot1, pos1),(mot2, pos2)...]
                assert len(tags) == len(outputs)  
                # il y aurais une erreur tandis que des codes après assert sont fausses
                total += len(tags)
                for o, t in zip(outputs, tags):
                    if o[1].strip() == t: right += 1  # la prédiction est correcte : right+1
                sentence = ([], [])
        logging.info("Precision : %f", right / total)  # le taux de la prédiction
        
    except IOError:
        logging.info('Reading corpus...')
        training_data = []  #stocker les phrases du copus
        sentence = ([], []) 
        for line in open('data/DSG_French/my_gsd_train.txt'):
            params = line.split('\t')
            sentence[0].append(params[0])
            sentence[1].append(params[1])
            if params[0] == '.':   # le résultat d'une phrase
                training_data.append(sentence)
                sentence = ([], [])
        logging.info('la taille du training corpus : %d', len(training_data))  
        #nombre du phrase dans le texte train.txt
        logging.info('Start training...')
        tagger.train(training_data, save_loc=PICKLE)