# Notes d'entretien — rendement du maïs

**Pitch de 45 secondes.** J'ai un benchmark reproductible de rendement du maïs au niveau comté, avec un dataset scientifique public, des features météo jusqu'au 31 juillet et une séparation chronologique. Le code compare deux baselines à Random Forest et XGBoost, produit des rapports, SHAP et une API locale. Le résultat principal est un échec de généralisation : XGBoost gagne la validation puis est battu par une tendance linéaire sur le test. Le projet démontre une méthodologie vérifiable, pas une promesse de prévision opérationnelle. Développement largement assisté par IA.

**Dataset.** Paudel, de Wit, Boogaard ; Zenodo 7751191 ; CC BY 4.0. Yields USDA/NASS, météo Copernicus, sol WISE. 9 524 lignes comté-année, 528 comtés, huit États, 2000–2018. 2 541 labels sélectionnés exclus faute de sol ; aucune imputation de cible. Pas de nouvelle collecte revendiquée.

**Pipeline.** Archive et checksum → validation clés → features avril–juillet → train 2000–2013 → validation 2014–2015 → sélection RMSE → réentraînement train+validation → test 2016–2018 → rapports et API. Préprocessing appris uniquement sur entraînement.

**Scores à connaître.** XGBoost validation RMSE 22,88 ; test RMSE 33,21, MAE 28,27, R² −0,650. Baseline tendance test RMSE 26,79. Les unités RMSE/MAE sont bu/acre. Les scores ne prouvent aucun transfert au Maroc. Le test final contient 1 391 lignes mais seulement trois années climatiques.

**Difficultés.** Zéros dans les labels source ; mois de durée différente ; couverture réduite par le sol ; dérive temporelle ; arbres qui n'extrapolent pas une tendance continue. La sélection RF/XGB est presque à égalité en validation. Les résultats, limites et bugs corrigés sont documentés ; pas d'optimisation après lecture du test.

**10 questions probables et réponses courtes**

1. **Quel problème résolvez-vous ?** J'étudie la prévision de rendement à l'échelle comté avant la fin de saison. Le modèle actuel ne démontre pas une qualité opérationnelle suffisante.
2. **D'où viennent les données ?** D'un dépôt scientifique Wageningen sur Zenodo, lui-même issu de NASS, Copernicus et WISE. Je cite la version et vérifie le checksum.
3. **Pourquoi ce split ?** Prédire une année future implique de ne pas entraîner sur elle. Un split aléatoire comté-année peut surestimer la performance temporelle.
4. **Comment évitez-vous la fuite ?** Aucune météo après juillet ; aucune superficie récoltée ni sortie de simulation proche du rendement ; preprocessing sur train ; choix du modèle sur validation. La réanalyse reste rétrospective, donc ce n'est pas une preuve de disponibilité en temps réel.
5. **Pourquoi RF et XGBoost ?** Données tabulaires de taille modérée et relations non linéaires ; ils sont adaptés à une première comparaison, mais leurs arbres extrapolent mal les tendances temporelles.
6. **Pourquoi une baseline de tendance ?** Les rendements changent avec les techniques et le contexte. Une méthode météo doit battre un historique simple, pas seulement une moyenne naïve.
7. **Comment interprétez-vous R² négatif ?** Les erreurs quadratiques dépassent celles de la moyenne du jeu évalué. C'est un signal d'échec, pas une accuracy négative ni un succès à cacher.
8. **Que prouve SHAP ?** Comment XGBoost attribue ses prédictions aux features. Cela ne prouve pas que modifier une variable provoquerait le changement de rendement annoncé.
9. **Peut-on l'utiliser au Maroc ?** Pas sans données locales et validation. Les comtés américains, cultures, climat, sols et pratiques ne correspondent pas automatiquement aux exploitations marocaines.
10. **Que feriez-vous ensuite et quelle part vient de l'IA ?** Validation roulante, tendances apprises sur train et résidus météo, puis nouvelle période intacte et test spatial. Le code/docs ont été largement assistés par IA ; je dois pouvoir relancer et expliquer le pipeline pour revendiquer une maîtrise personnelle.

**Ne pas dire :** « déployé en production », « meilleur modèle agricole », « validé au Maroc », « créé le dataset », « réalisé sans IA » ou « 95 % d'accuracy ». Aucun de ces éléments n'est établi.
