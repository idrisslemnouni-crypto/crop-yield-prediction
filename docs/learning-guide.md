# Comprendre et défendre le projet

## Le problème et la décision principale

On prédit le rendement annuel du maïs d'un comté américain à partir de météo d'avril à juillet et du sol. Le rendement est en bushels par acre, pas en tonnes par hectare. Il s'agit d'une étude rétrospective, pas d'un service agricole validé au Maroc. La date limite est fixée avant l'évaluation : 31 juillet. Une variable connue après récolte rendrait le problème artificiellement facile.

Le dataset appartient aux chercheurs cités dans data/README.md. La météo provient d'une réanalyse : connaître rétrospectivement une valeur avant le cutoff calendaire ne prouve pas qu'elle était disponible opérationnellement ce jour-là. Pour un vrai produit, il faudrait conserver les versions de données disponibles à la date de prévision.

## Pourquoi chaque étape existe

1. **Télécharger et vérifier le checksum** : garantir que le code travaille sur la source officielle attendue, plutôt qu'un CSV différent ou corrompu. Les SHA-256 documentent ensuite les fichiers réellement utilisés.
2. **Vérifier les clés** : un doublon comté-année peut multiplier des lignes pendant une jointure et fausser l'évaluation. Un doublon météo peut doubler une pluie mensuelle.
3. **Exiger les 12 dekads** : une somme de pluie sur deux tiers d'un mois n'est pas une observation mensuelle complète. Les lignes exclues sont comptées. Le sol manquant élimine une partie des comtés ; cela introduit un biais de couverture.
4. **Construire les features** : sommer pluie et ET0, calculer le bilan, pondérer la température moyenne par la durée, conserver les extrema. Les dekads n'ont pas tous dix jours. Le bilan PREC−ET0 n'est ni l'humidité mesurée du sol ni un besoin d'irrigation exact.
5. **Séparer par année** : train 2000–2013, validation 2014–2015, test 2016–2018. Un split aléatoire mélangerait les régimes temporels. Le même comté peut apparaître dans plusieurs périodes : on ne mesure donc pas le transfert vers de nouveaux territoires.
6. **Tester des baselines** : moyenne globale puis tendance linéaire du rendement par comté. La seconde représente des gains techniques et structurels sans météo. Un modèle sophistiqué doit justifier sa valeur face à elle.
7. **Entraîner RF et XGBoost** : ils apprennent des relations non linéaires sur un tableau de taille modérée. Le deep learning n'est pas nécessaire ici. Les hyperparamètres sont fixés, pas optimisés sur le test.
8. **Sélectionner, réentraîner, tester** : choisir le plus faible RMSE en validation ; ajuster le modèle choisi sur train+validation ; mesurer sur des années futures. Ne pas choisir à nouveau après avoir vu les scores du test.
9. **Examiner les erreurs et SHAP** : localiser les années/États difficiles et comprendre les variables utilisées. SHAP est une décomposition descriptive de la prédiction, pas une preuve de causalité agronomique.
10. **Servir le modèle** : appliquer exactement le même preprocessing et valider les entrées. L'API illustre l'engineering ; des réponses HTTP correctes ne prouvent pas la qualité scientifique du modèle.

## Ce que les résultats enseignent

XGBoost est retenu en validation : RMSE 22,88 bu/acre. Sur le test, RMSE 33,21, MAE 28,27 et R² −0,650. La tendance par comté fait mieux sur ce test : RMSE 26,79. On conserve la sélection initiale ; on n'annonce pas un succès de prévision.

R² négatif signifie que le modèle fait pire, au sens des erreurs quadratiques, qu'une prédiction constante égale à la moyenne du **jeu évalué**. Cette moyenne utilise les labels futurs : c'est une référence mathématique, pas une baseline disponible à la date de prévision. La vraie baseline moyenne est calculée sur les données d'entraînement et a ses propres scores.

L'année est la principale attribution SHAP. Les arbres partitionnent les années connues ; ils n'extrapolent pas naturellement une pente de rendement au-delà de la dernière année apprise. La dérive du rendement et la petite validation peuvent expliquer une partie de la dégradation ; l'étude ne démontre pas que ce soit l'unique cause. RF et XGB sont pratiquement ex æquo en validation (environ 0,02 bu/acre d'écart), donc cette sélection est fragile.

L'ablation XGBoost sans météo donne RMSE 24,88 en validation, contre 22,88 avec météo. Cela suggère une contribution sur ces deux années seulement. Aucun gain robuste à long terme ni bénéfice causal n'est établi.

## Erreurs rencontrées et corrections

Le premier contrôle exigeait des rendements strictement positifs. L'archive contient des zéros : cette hypothèse trop stricte a arrêté l'exécution. Après inspection, le contrôle accepte les rendements non négatifs et documente les zéros. Aucun zéro ne subsiste dans la table finale après la jointure au sol. Les zéros pourraient représenter une absence de production ou un codage particulier ; leur sens doit être confirmé avant extension du périmètre.

Le projet s'exécute, mais le modèle ne généralise pas suffisamment. Ce n'est pas un bug à masquer avec de nouvelles optimisations sur les mêmes années test. Après examen du test, un nouveau modèle nécessite un nouveau protocole et une période intacte. Le Dockerfile et le workflow CI sont fournis ; leur exécution distante n'est pas attestée.

## Alternatives à comprendre

- Régression régularisée : baseline interprétable intéressante, avec interactions/agronomie adaptées.
- Tendance historique + apprentissage des résidus météo : piste pour séparer progrès temporel et anomalies climatiques ; calculer toute tendance sur entraînement uniquement.
- Validation chronologique roulante : plusieurs années de développement pour stabiliser le choix ; garder une période finale non inspectée.
- Split spatial : nécessaire pour prétendre prédire un nouveau territoire.
- Sentinel-2 : pertinent pour le projet suivant, mais impose masques nuages, disponibilités temporelles et validation terrain.

## À faire soi-même avant entretien

Relancer les commandes README ; ouvrir `features.py` et vérifier un mois à la main ; recalculer le RMSE dans le notebook ; expliquer pourquoi la tendance bat XGBoost au test ; envoyer une entrée incorrecte à l'API et lire l'erreur ; modifier une donnée **après** le cutoff dans les tests et constater que les features restent identiques. Le test avec données artificielles vérifie un contrat logiciel, il ne constitue pas une preuve de performance agricole.

Le code et les décisions ont été largement assistés par IA. Les comprendre est une étape distincte de leur génération ; ne pas raconter un historique ou une expérience qui n'a pas eu lieu.
