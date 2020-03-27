
## Projet

Le projet est construit sous la forme d'un projet de programmation avec un mémoire résumant les expériences réalisées ainsi que les aspects théoriques d'Adam. Les codes sources sont commentés, intelligibles et modulaires.

Le dossier `gdsvm/` https://github.com/icannos/online-optimization/tree/master/gdsvm contient l'implémentation des différents algorithmes utilisés dans le projet et notamment un notebook résumant les expériences réalisées, `report/` contient les sources de mon rapport ainsi que le pdf compilé `mdarrin-adam.pdf`. https://github.com/icannos/online-optimization/blob/master/report/mdarrin-adam.pdf

Pour utiliser le code, il faut placer les jeux de données utilisés (ceux proposés en cours) en cours dans le dossier `gdsvm/data`.

### Structure du code

Dans le dossier `gdsvm`.

```
.
├── adam.ipynb					// Jupyter notebook avec des expériences
├── batch.py					// Script pour générer des graphs en faisant varier la taille des batchs
├── data					// Le jeu de données MNIST vu en cours
│   ├── mnist_test.csv
│   └── mnist_train.csv
├── exports					// Les résultats des scripts sont exportés ici
├── mnist.py					// Code de base pour apprendre sur mnist avec les algos usuels
├── mnist_svrg.py				// Une digression hors sujet avec l'algo svrg
├── optimizers					// Contient tous les optimizers implémentés 
│   ├── adabound.py				
│   ├── adagrad.py
│   ├── adam.py
│   ├── momentum.py
│   ├── projected_gradient_descent.py		// Une PGD que j'ai implémentée en pytorch (pour un autre projet)
│   ├── rmsprop.py
│   ├── sgd.py
│   └── svrg.py					// Une digression hors sujet: SVRG
├── regularization.py				// Génère les graphes en faisant varier la régularisation
├── svm.py					// Les fonctions de coûts et de gradient de la SVM vue en cours
├── toy_loop.py					// La boucle d'entraînement pour les exemples jouets
├── toys_function.py				// Les fonctions jouets à optimiser
└── utils.py					// Quelques outils pour générer graphs et animations
```
