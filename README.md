# Estructura del repositori de codi
Aquest repositori està dividit en tres parts:
- **Data:** En el directori "/data" estàn les dades que es processen en el treball. Totes les dades, conjuntament amb les dades que va donar el Michael estàn en aquest dataset de Kaggle https://www.kaggle.com/datasets/martpirla/dades-del-tfg-del-mart-pirla.
  - **data/barcelona_data2.csv**: Dades tal qual les va donar el Michael (Només està al Kaggle).
  - **data/data_cleared_barcelona.csv**: Dades simplement netegades i arreglats errors en les recompenses.
  - **data/parameters_fitting.csv**: Conjunt de paràmetres donats per l'ajustament d'hiperparametres.
- **Utils:** En el directori "/utils" està el codi de funcions i classes que es fan servir multiples vegades.
  - **utils/create_stimuli.py**: Té les funcions per la creació dels estímuls i per calcular d'un episòdi la performance.
  - **utils/gaussian_smoothed_performance.py**: Té les funcions per suavitzar i comparar les performances.
  - **utils/qlearningAgent.py**: Té la classe de l'agent de Q-learning.
  - **utils/column_game.py**: Classe que encapsula tota la llògica d'una etapa de l'exercici.
- **trim_data.ipynb**: Notebook que conté totes les modificacions de "barcelona_data2.csv" a l'hora de crear "data_cleared_barcelona.csv".
- **data_visualitzations.ipynb**: Notebook que conté totes el codi per crear tots els gràfics preliminars als resultats.
- **hyperparameter_fitting.ipynb**: Notebook amb tot el codi fet servir per obtenir els resultats.
- 
