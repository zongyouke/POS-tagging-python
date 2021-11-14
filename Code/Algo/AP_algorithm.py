# -*- coding:utf-8 -*-

#Le perceptron moyenné : 
#La valeur de poids moyen après la fin de l'entraînement 
#et la valeur de poids moyenne est utilisée comme valeur de poids finale

 
from collections import defaultdict
import pickle
 
class AveragedPerceptron(object):
 
    def __init__(self):
        #il y a un vecteur dans chaque lieu
        self.weights = {}
        self.classes = set()
        # les poids cumulés qu'ils sont utilisés pour calculer le poids moyen
        self._totals = defaultdict(int)  # un dictionnaire généré avec la valeur 0
        self._tstamps = defaultdict(int) # i lors de la dernière mise à jour des pondérations
        self.i = 0   # le nombre de cas d'enregistrement
 
    def predict(self, features):  # le vecteur de caractéristique * le vecteur de poids => l'étiquette lexicale
        scores = defaultdict(float)
        for feat, value in features.items():
            if feat not in self.weights or value == 0:
                continue
            weights = self.weights[feat]
            for label, weight in weights.items():
                scores[label] += value * weight
                
        # renvoyer les meilleurs scores
        return max(self.classes, key=lambda label: (scores[label], label)) # renvoyer le label ayant le socre le plus élevé, si des scores sont égals, on prend la lettre la plus grande
        print(max(self.classes, key=lambda label: (scores[label], label)))

    def update(self, truth, guess, features): # mis à jour les valeurs des pondérations 
 
        def upd_feat(c, f, w, v): # c: la nature correcte/prédite du mots; f:feature ; w:la valeur du poids correspondant; v:1(ou -1)
            param = (f, c)
            self._totals[param] += (self.i - self._tstamps[param]) * w  # l'accumulation :[i (à ce moment là) - i (mis à jour la prochaine fois)]*la valuer du poid
            self._tstamps[param] = self.i # i mis à jour à ce moment-là 
            self.weights[f][c] = w + v  # mis à jour de la valeur du poid
 
        self.i += 1
        if truth == guess:  # si le prédiction est correct, on n'en revouvelle pas
            return None
        for f in features:
            weights = self.weights.setdefault(f, {})  # renvoyer la valuer du f qui est dans le key du dic weights , si il n'y a pas la valeur, renvoit un dic vide:{}
            upd_feat(truth, f, weights.get(truth, 0.0), 1.0)  # plus 1 dans des pois de caractéristiques(feature, la nature correcte du mot)
            upd_feat(guess, f, weights.get(guess, 0.0), -1.0) # moins 1 dans des pois de caractéristiques(feature, la nature correcte du mot)
        return None
 
    def average_weights(self):  # calculer les poids moyen 
        for feat, weights in self.weights.items():
            new_feat_weights = {}
            for clas, weight in weights.items():
                param = (feat, clas)
                total = self._totals[param]
                total += (self.i - self._tstamps[param]) * weight
                averaged = round(total / float(self.i), 3)     # pour chaque poid : faire la moyenne par rapport ses itérations de i fois
                if averaged:
                    new_feat_weights[clas] = averaged
            self.weights[feat] = new_feat_weights
        return None
 
    def save(self, path):    # un dic pour enregistrer les poids
        return pickle.dump(dict(self.weights), open(path, 'w'))
 
    def load(self, path):    # parcourir le dic enregistré les poids
        self.weights = pickle.load(open(path))
        return None