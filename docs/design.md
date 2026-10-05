# Crop yield prediction — design

Objectif : benchmark rétrospectif de rendement du maïs au niveau comté, à partir de données disponibles jusqu'à la fin du dekad 21 (31 juillet). Source : Paudel, de Wit et Boogaard, Zenodo 7751191, county-data.zip, CC BY 4.0. Archive 50 051 357 octets, MD5 officiel b7cf000262da294caffc2fea39932246.

Périmètre : IA, IL, IN, OH, MN, WI, MI et MO, 2000–2018. Utiliser rendement USDA/NASS, météo Copernicus agrégée par dekad et capacité de rétention du sol WISE. Écarter CSSF (sorties de simulation), superficies récoltées et FAPAR lissé : ne pas ajouter de variable susceptible d'utiliser des informations après la date de prévision. La météo réanalysée rétrospective ne démontre pas une disponibilité opérationnelle au 31 juillet.

Modules : acquisition avec contrôle MD5 et extraction des trois fichiers nécessaires ; features mensuelles avril–juillet ; entraînement ; rapports ; prédiction ; API. La validation des clés et des 12 dekads évite des totaux incomplets. Pas d'imputation des rendements. Statistiques de preprocessing ajustées uniquement sur entraînement.

Train 2000–2013, validation 2014–2015, test final 2016–2018. Modèles fixés avant test : moyenne globale, tendance linéaire par comté, Random Forest, XGBoost. Sélection par RMSE de validation ; réentraînement sur train+validation ; test final de tous les candidats pour transparence, sans resélection. Métriques RMSE/MAE/R², erreurs par année/État, bootstrap des années (trois années seulement, incertitude fragile). SHAP sur le modèle XGBoost comme outil descriptif, sans causalité. Comparaison sans météo pour l'ablation seulement sur validation.

API FastAPI : santé et prédiction d'une ligne mensuelle validée ; modèle local de confiance ; 503 en absence de modèle ; domaine historique limité à 2000–2018 ; refuser États non supportés et entrées incohérentes. Interface simple de démonstration au sein de l'API. Aucun déploiement cloud revendiqué.

Vérification : tests des contrats/joins/cutoff/splits et de l'inférence ; lint ; notebook exécuté ; entraînement complet ; API réelle ; installation et pipeline depuis clone propre. Figures et résultats générés par code. Code MIT ; données CC BY 4.0 séparées ; raw et modèle ignorés par Git. Git conserve seulement de vrais commits du travail courant.

Alternatives écartées : USCrop complet (1 Go, plus lourd pour démarrer), rendement national FAOSTAT (peu d'observations indépendantes et agrégation excessive pour ce benchmark). Choix documenté pour apprendre à défendre l'étude.
