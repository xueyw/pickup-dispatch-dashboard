# Ramassage petits volumes : planification des tournées, reporting quotidien et alertes de retour

[English](README.md) | Français | [中文](README.zh-CN.md)

Ce projet de simulation s'appuie sur le jeu de données public [LaDe](https://huggingface.co/datasets/Cainiao-AI/LaDe) de Cainiao (ramassages à Jilin, environ 262 000 commandes). Il modélise une activité de ramassage auprès de commerçants à petits volumes, avec les contraintes suivantes :

- Les véhicules sont planifiés la veille. **Chaque véhicule ne fait qu'une seule tournée par jour**, et le plan ne peut guère changer en cours de journée.
- Un véhicule supplémentaire demandé dans la journée met plus d'une heure à arriver.
- **Tous les véhicules doivent être rentrés au dépôt avant 18 h** pour décharger.

> La capacité des véhicules, le volume des colis et la vitesse ne figurent pas dans les données d'origine. Ils sont simulés, et l'en-tête de chaque script documente les hypothèses.

## Modules

| Module | Fichier | Rôle |
|---|---|---|
| Nettoyage des données | `src/clean.py` | Marquage à trois niveaux : sale `dq_*` (exclu), suspect `sus_*` (conservé), anomalie métier `biz_*` (analysée). Validation du schéma avec pandera. |
| Entrepôt de données | `sql/`, `src/load_db.py` | Couches PostgreSQL raw → stg → mart. Les résultats du nettoyage SQL et Python sont contrôlés règle par règle. |
| Optimisation des tournées | `src/capacity_sensitivity.py` | CVRPTW avec OR-Tools : capacité des véhicules, fenêtres horaires et retour au dépôt avant 18 h. |
| Carte interactive | `app.py` | Streamlit + folium. Trace la tournée de chaque véhicule et signale ceux qui ont moins de 60 min de marge avant le retour. |
| Simulation de marge | `src/buffer_sim.py` | Planifie la veille à partir de la prévision, puis exécute avec les commandes réelles du jour. Compare les véhicules de réserve à l'assouplissement des créneaux du matin. |
| Segmentation des commerçants | `src/segment.py` | Segmente les AOI par fréquence et par profil de créneau horaire. Estime le nombre de véhicules économisés en négociant des créneaux du matin plus tardifs avec les principaux commerçants. |
| Prévision de la demande | `src/forecast.py` | GBM quantile (P50/P80) avec variables de jours fériés. Repère les jours où les données s'effondrent. |
| Alertes de retour | `src/alert.py` | Rejoue des points de contrôle de 13 h à 17 h. Évalue la précision des règles d'alerte et si elles se déclenchent avant 16 h, ce qui laisse encore le temps d'ajouter un véhicule. |
| Reporting BI | Power BI | Connexion directe à la couche mart de PostgreSQL. |

## Technologies

| Usage | Bibliothèques / outils |
|---|---|
| Traitement des données | pandas, NumPy, PyArrow (stockage Parquet) |
| Validation des données | pandera (contrôle du schéma et des plages de valeurs) |
| Base de données | PostgreSQL 16, psycopg (chargement en masse via COPY), SQLAlchemy |
| Optimisation des tournées | Google OR-Tools (VRP avec capacité et fenêtres horaires) |
| Prévision | scikit-learn (HistGradientBoostingRegressor, perte quantile) |
| Visualisation | Streamlit, folium + streamlit-folium (cartes), Plotly, Power BI |
| Divers | tabulate (rapports Markdown), concurrent.futures (simulation parallèle) |

## Chaîne de traitement

```
CSV brut (262 000 lignes)
  │  clean.py : analyse des champs → dédoublonnage → 15 règles de marquage → validation pandera
  ▼
Données nettoyées (Parquet, toutes les lignes conservées, séparées par is_clean)
  ├─ capacity_sensitivity.py  taille de flotte minimale selon la capacité des véhicules
  ├─ buffer_sim.py            plan de la veille vs exécution du jour (multiprocessus, 8 jours × 2 zones × 6 scénarios)
  ├─ segment.py               segmentation + expérience de négociation des créneaux
  ├─ forecast.py              panel zone-jour → détection des anomalies → variables → prévision P50/P80
  └─ alert.py                 rejeu des points de contrôle → évaluation des règles d'alerte
  ▼
load_db.py : raw (brut) → stg (nettoyage SQL) → mart (rapport quotidien, coursier-jour, synthèse qualité)
  ▼
Rapport quotidien Power BI / carte Streamlit
```

## Problèmes résolus

**Nettoyage des données**
- **Horodatages sans année.** Le format brut est `MM-JJ HH:MM:SS`. J'ai ajouté l'année à partir du champ `ds` et vérifié que chaque heure de ramassage correspond bien à sa date.
- **Une règle de distance qui marquait à tort 9 % des lignes.** La première règle (plus de 100 km du centre-ville) classait tout un district rural comme sale. Je l'ai remplacée par « plus de 30 km du point médian de sa propre zone », qui ne touche plus que 2 enregistrements réellement aberrants.
- **Traitement à trois niveaux.** On ne peut pas simplement supprimer les GPS manquants (16 %) ou en dérive sans perdre beaucoup de commandes valides. Les règles sont donc classées en sales (exclues), suspectes (conservées et marquées) et anomalies métier (analysées), et chaque règle indique son nombre de lignes touchées et sa part.
- **Résultats SQL et Python contrôlés l'un par l'autre.** Les nombres de lignes touchées par les deux chaînes sont comparés pour les 13 règles. Ils concordent tous, ce qui garantit que la base et l'analyse utilisent les mêmes définitions.
- **Problèmes de collecte détectés.** Début mai correspond à la montée en charge de la collecte. Du 20 au 25 juin, 11 zones urbaines sont tombées presque à zéro alors que le district restait normal : ce sont des données manquantes, pas un arrêt réel de l'activité. Toute zone-jour inférieure à 30 % de sa médiane des 28 jours précédents est marquée automatiquement et exclue de l'entraînement et de l'évaluation.

**Modélisation et analyse**
- **Points abandonnés à cause de créneaux à cheval sur deux jours.** Certaines commandes ont un créneau qui tombe le lendemain. Coupé à 18 h, ce créneau devenait impossible à planifier ; ces commandes sont désormais flexibles (n'importe quand dans la journée).
- **Identifier ce qui détermine vraiment la taille de flotte.** J'ai d'abord pensé à un manque de marge de temps, mais les véhicules rentraient à moitié vides. L'analyse a montré que 73 % des points planifiés doivent être ramassés entre 9 h et 11 h : ce sont les créneaux du matin qui fixent le nombre de véhicules. J'ai donc reconçu la simulation en « véhicules de réserve × assouplissement des créneaux du matin ».
- **Une première segmentation où presque tous les commerçants étaient « mixtes ».** Une AOI regroupe plusieurs commerçants, donc des règles fondées sur la part de commandes ne les distinguent pas. En segmentant plutôt selon la « part des jours avec une commande à ramasser obligatoirement le matin », on obtient des groupes nets.
- **Trop d'alertes de pic.** Avec un seuil sur le P80, on obtenait 574 alertes sur 875 zones-jours, soit presque tous les jours. Avec un seuil sur le P50, on passe à 199 alertes, avec 33 % de précision et 40 % de rappel.
- **Délai des alertes redéfini.** Une alerte à 17 h était comptée « à temps », alors qu'il est déjà trop tard pour ajouter un véhicule. Seules les alertes émises avant 16 h comptent désormais, et un mode « alerte précoce » est évalué à part.
- **Stabilité du plan.** Lors d'une réoptimisation en cours de journée, réaffecter un point à un autre véhicule est pénalisé, ce qui respecte la contrainte « le plan ne change guère ».

## Résultats clés

- **Le plan de la veille est le principal levier.** Dans la grande zone urbaine, avec 6,75 véhicules planifiés, le travail n'est pas terminé 38 % des jours. Un véhicule de réserve ramène ce taux à 0 %.
- **Les créneaux du matin (9 h–11 h) fixent la taille de flotte.** Négocier un créneau décalé d'une heure avec les 20 premiers commerçants en volume du matin économise environ un véhicule (1,13 en zone urbaine, 1 dans le district).
- **Prévision.** Le WAPE global est au niveau de la référence « médiane sur 28 jours » (12,8 % contre 12,6 %). Le gain porte sur les jours de pic : la sous-estimation passe de 20,3 % à 13,0 %, et la couverture du P80 est de 80,5 %.
- **Les alertes du jour sont un filet de sécurité.** La règle la plus simple, « au moins 2 points encore en attente à 16 h », détecte 67 % des retours tardifs à un moment où l'on peut encore ajouter un véhicule. Elle fait mieux que les règles fondées sur l'ETA.
- **Les problèmes de données passent en premier.** La montée en charge de début mai et l'arrêt urbain de fin juin doivent être détectés et retirés, sinon la prévision et la référence sont faussées.

Les résultats détaillés se trouvent dans `reports/` (en chinois).

## Démarrage

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p data/raw && curl -L -o data/raw/pickup_jl.csv \
  https://huggingface.co/datasets/Cainiao-AI/LaDe/resolve/main/pickup/pickup_jl.csv

.venv/bin/python src/clean.py
.venv/bin/python src/capacity_sensitivity.py
.venv/bin/python src/buffer_sim.py
.venv/bin/python src/segment.py
.venv/bin/python src/forecast.py
.venv/bin/python src/alert.py
.venv/bin/python src/load_db.py      # nécessite un PostgreSQL local avec une base nommée lade
.venv/bin/streamlit run app.py
```

## Limites

- Les coursiers de LaDe font aussi des livraisons, d'où un taux de retard de base élevé dans les données brutes.
- La simulation suppose que toutes les commandes du jour sont connues dès le matin.
- Chaque zone n'est testée que sur 8 jours.
