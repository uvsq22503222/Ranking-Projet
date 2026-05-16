# Projet de Ranking — PageRank & Google Bombing

GE Shuning - GUO Xuxin

## Description

Ce projet implémente l’algorithme de PageRank sur des graphes web creux et étudie l’effet d’attaques de type Google Bombing sur le classement de pages cibles.

Les objectifs principaux sont :

- calculer le PageRank sur des graphes web réels,
- supporter plusieurs formats de graphes,
- simuler différentes structures d’attaque,
- mesurer l’impact des attaques sur le PageRank d’une page cible.

# Structure du projet

```text
.
├── pagerank.py     # Implémentation sparse de PageRank
├── attaque.py      # Expériences de Google Bombing
├── README.md
├── *.mtx           # Graphes au format Matrix Market
└── *.txt           # Graphes au format adjacency list
```

# Fonctionnalités

## PageRank

L’implémentation comprend :

- calcul sparse du PageRank,
- gestion des dangling nodes (pages sans lien sortant),
- critère de convergence avec norme L1,
- support des graphes pondérés et non pondérés,
- support de graphes de grande taille.

L’itération de PageRank utilisée est :

[
x^{(k+1)} = \alpha x^{(k)}P + \alpha \frac{x^{(k)}f^T}{N}e + (1-\alpha)\frac{1}{N}e
]

avec :

- (P) la matrice de transition,
- (f) le vecteur des dangling nodes,
- (\alpha) le facteur d’amortissement.

# Formats supportés

## Format Matrix Market (.mtx)

Exemple :

```text
500 500 2636
1 2
1 5
...
```


## Format adjacency (.txt)

Deux formats sont supportés.

### Format 1

```text
9914 36854
```

### Format 2

```text
9914
36854
```

Chaque ligne suivante décrit :

```text
sommet  degré_sortant  voisin1 poids1 voisin2 poids2 ...
```


# Simulation de Google Bombing

Le projet étudie comment l’ajout de pages attaquantes modifie le PageRank d’une page cible.

Trois types de cibles sont sélectionnés :

- une cible à PageRank fort,
- une cible à PageRank moyen,
- une cible à PageRank faible.


# Structures d’attaque

## 1. Graphe complet

Toutes les pages attaquantes pointent vers :

- tous les autres attaquants,
- la page cible.


## 2. Structure en anneau

La cible appartient à un cycle :

```text
cible -> a1 -> a2 -> ... -> an -> cible
```


## 3. Sommets isolés

Chaque attaquant pointe uniquement vers la cible :

```text
a1 -> cible
a2 -> cible
...
```


# Installation

## Pré-requis

- Python 3.10+
- NumPy

Installation de NumPy :

```bash
pip install numpy
```


# Utilisation

## Exécution de PageRank

```bash
python pagerank.py fichier.mtx ou .txt
```


## Exécution des expériences de Google Bombing

```bash
python attaque.py fichier.mtx ou .txt
```


# Exemple de résultat

```text
=== Cible forte (sommet 15) ===
  [isolés]
    n= 100  Δ=+0.03109292
```

Résumé final :

```text
============================================================
RÉSUMÉ : meilleure attaque par cible
============================================================
Cible      Structure      n         Δ max
------------------------------------------------------------
forte      isolés       100   +0.03109292
moyenne    isolés       100   +0.02118953
faible     isolés       100   +0.02120000
```


# Conclusions expérimentales

Les expériences montrent que :

- les sommets isolés constituent l’attaque la plus efficace,
- augmenter le nombre d’attaquants augmente fortement le PageRank de la cible,
- les graphes complets redistribuent une partie du PageRank entre attaquants et réduisent donc l’efficacité de l’attaque,
- les structures en anneau gardent une partie du PageRank dans le cycle,
- certaines structures deviennent moins efficaces sur des cibles déjà très fortes.

La stratégie la plus efficace observée est donc :

```text
un grand nombre de sommets isolés pointant directement vers la cible.
```

