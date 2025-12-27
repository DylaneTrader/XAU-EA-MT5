# Guide d'Intégration MT5 - XAU-EA-MT5

## 📋 Guide Complet d'Intégration de l'Expert Advisor dans MetaTrader 5

Ce guide vous accompagne pas à pas pour intégrer et utiliser l'Expert Advisor (EA) Transformer pour le trading automatisé de XAUUSD dans MetaTrader 5.

---

## 🎯 Vue d'Ensemble

Cet EA utilise un réseau de neurones Transformer (PyTorch) pour analyser les mouvements de prix de l'or (XAUUSD) et générer automatiquement des signaux de trading. L'intégration complète comprend:

1. **Préparation de l'environnement** (Python + MT5)
2. **Entraînement du modèle** avec des données historiques
3. **Configuration de l'EA** avec vos paramètres
4. **Déploiement et exécution** sur MT5
5. **Surveillance et optimisation** des performances

---

## 📦 Prérequis

### 1. Logiciels Nécessaires

#### MetaTrader 5
- **Télécharger**: [Site officiel MetaQuotes](https://www.metatrader5.com/fr/download)
- **Version requise**: MT5 version 5.0.45 ou supérieure
- **Installation**: Suivre l'assistant d'installation standard
- **Compte**: Compte démo ou réel chez un courtier supportant MT5

#### Python
- **Version requise**: Python 3.8 ou supérieur
- **Télécharger**: [python.org](https://www.python.org/downloads/)
- **Important**: Cocher "Add Python to PATH" lors de l'installation Windows

### 2. Vérification des Installations

Ouvrez un terminal/invite de commande et vérifiez:

```bash
# Vérifier Python
python --version
# Devrait afficher: Python 3.8.x ou supérieur

# Vérifier pip
pip --version
```

### 3. Configuration MT5

1. **Ouvrir MetaTrader 5**
2. **Activer le trading algorithmique**:
   - Menu: `Outils` → `Options` → `Expert Advisors`
   - ✅ Cocher "Autoriser le trading algorithmique"
   - ✅ Cocher "Autoriser l'importation de DLL"
   - ✅ Cocher "Autoriser les signaux en temps réel"
   - Cliquer sur `OK`

3. **Vérifier le symbole XAUUSD**:
   - Menu: `Affichage` → `Symboles` (Ctrl+U)
   - Rechercher "XAUUSD" ou "XAUUSDm"
   - Clic droit → `Afficher` (s'il est masqué)
   - Note: Le nom exact peut varier selon le courtier (XAUUSD, XAUUSDm, GOLD)

---

## 🚀 Installation du Projet

### Étape 1: Cloner le Dépôt

```bash
# Ouvrir le terminal dans le dossier de votre choix
cd C:\Users\VotreNom\Documents\Trading  # Windows
# ou
cd ~/Documents/Trading  # macOS/Linux

# Cloner le projet
git clone https://github.com/DylaneTrader/XAU-EA-MT5.git
cd XAU-EA-MT5
```

Si vous n'avez pas Git, téléchargez le ZIP depuis GitHub et extrayez-le.

### Étape 2: Installer les Dépendances Python

```bash
# Créer un environnement virtuel (recommandé)
python -m venv venv

# Activer l'environnement virtuel
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Installer les packages requis
pip install -r requirements.txt
```

**Packages installés**:
- `MetaTrader5`: API Python pour MT5
- `torch`: Framework PyTorch pour le modèle Transformer
- `numpy`, `pandas`: Manipulation de données
- `scikit-learn`: Prétraitement et évaluation
- `ta`: Bibliothèque d'indicateurs techniques
- `streamlit`: Interface graphique pour l'entraînement

### Étape 3: Vérifier l'Installation

```bash
# Tester la connexion MT5
python test_ea.py
```

**Résultat attendu**:
```
[INFO] Testing MT5 connection...
[INFO] MT5 initialized successfully
[INFO] Terminal: MetaTrader 5 version 5.0.xx
✅ MT5 Connection: PASSED
```

Si des erreurs apparaissent, consultez la section [Dépannage](#-dépannage).

---

## 🎓 Entraînement du Modèle

Avant d'utiliser l'EA en production, vous **devez** entraîner le modèle avec des données historiques. Deux méthodes sont disponibles.

### Méthode 1: Interface Graphique (Recommandée)

#### Lancer le Dashboard Streamlit

```bash
streamlit run streamlit_dashboard.py
```

Votre navigateur ouvrira automatiquement l'interface à `http://localhost:8501`

#### Processus d'Entraînement Visuel

**1. Chargement des Données** (Onglet "Data Overview")
   - **Option A**: Sélectionner "Use Default Data"
     - Utilise les données XAUUSD incluses (2015-2025, 1M+ barres)
   - **Option B**: Sélectionner "Upload Custom File"
     - Format accepté: CSV ou XLSX
     - Colonnes requises: `open`, `high`, `low`, `close`, `volume`
   - Cliquer sur **"Calculate Technical Indicators"**
   - Vérifier les indicateurs ajoutés: RSI, MACD, Bollinger Bands, ATR

**2. Configuration de l'Entraînement** (Barre Latérale)
   - **Training Epochs**: 20 (recommandé pour début)
   - **Test Set Size**: 0.2 (20% pour validation)
   - **Sequence Length**: 60 (barres à analyser)
   - **Forward Bars**: 5 (prédiction 5 barres à l'avance)
   - **Price Threshold**: 0.001 (0.1% mouvement minimum)
   - **Hidden Dimension**: 128 (complexité du modèle)
   - **Transformer Layers**: 4

**3. Sélection des Features** (Onglet "Training")
   - Cocher les indicateurs à utiliser:
     - ✅ open, high, low, close, volume
     - ✅ rsi, macd, macd_signal
     - ✅ bb_upper, bb_lower, atr
   - Cliquer sur **"Start Training"**
   - Attendre la fin de l'entraînement (peut prendre 5-30 minutes)

**4. Évaluation** (Onglet "Evaluation")
   - **Accuracy**: Devrait être > 50% (aléatoire = 33%)
   - **Confusion Matrix**: Vérifier la répartition des prédictions
   - **Classification Report**: Analyser précision par classe
   - **Objectif**: Accuracy > 55% pour un bon modèle

**5. Sauvegarde** (Onglet "Model Management")
   - Cliquer sur **"Save Model to Disk"**
   - Fichier créé: `transformer_ea_model.pth`
   - ✅ Message de confirmation: "Model saved successfully"

### Méthode 2: Script en Ligne de Commande

```bash
python train_model.py
```

Ce script:
1. Se connecte à MT5
2. Télécharge 5000 barres historiques de XAUUSD
3. Calcule les indicateurs techniques
4. Crée les labels (BUY/HOLD/SELL)
5. Entraîne le modèle Transformer
6. Sauvegarde automatiquement dans `transformer_ea_model.pth`

**Sortie Console**:
```
[INFO] Loading historical data...
[INFO] Data shape: (5000, 11)
[INFO] Creating labels...
[INFO] Training model...
Epoch 1/20: Loss=1.0234, Accuracy=45.2%
Epoch 2/20: Loss=0.8765, Accuracy=52.1%
...
Epoch 20/20: Loss=0.4321, Accuracy=58.7%
[INFO] Test Accuracy: 56.3%
[INFO] Model saved to transformer_ea_model.pth
✅ Training completed successfully
```

### Vérification du Modèle

```bash
# Vérifier que le fichier existe
# Windows:
dir transformer_ea_model.pth
# Linux/macOS:
ls -lh transformer_ea_model.pth
```

**Fichier attendu**: `transformer_ea_model.pth` (environ 1-5 MB)

---

## ⚙️ Configuration de l'EA

### Éditer le Fichier de Configuration

Ouvrez `config.py` avec un éditeur de texte (Notepad++, VSCode, etc.)

#### 1. Paramètres de Connexion MT5

```python
# MT5 Connection Settings
MT5_LOGIN = 297581462        # Votre numéro de compte MT5
MT5_PASSWORD = "#Trader001"  # Votre mot de passe MT5
MT5_SERVER = "Exness-MT5Trial9"  # Serveur de votre courtier
```

**Notes**:
- Pour un **compte démo**, ces paramètres sont souvent optionnels
- Pour un **compte réel**, ils sont **obligatoires**
- Le serveur dépend de votre courtier (ex: "ICMarkets-Demo", "XM-Real")

#### 2. Paramètres de Trading

```python
# Trading Parameters
SYMBOL = "XAUUSDm"           # Symbole exact dans votre MT5
TIMEFRAME = "M5"             # M1, M5, M15, M30, H1, H4, D1
LOT_SIZE = 0.01              # Taille de position (0.01 = micro lot)
MAGIC_NUMBER = 234000        # Identifiant unique (ne pas changer)
```

**Adapter le symbole**:
- Exness: `XAUUSDm`
- IC Markets: `XAUUSD`
- XM: `GOLD`
- Vérifier dans MT5: Menu → Affichage → Symboles

#### 3. Gestion du Risque

```python
# Risk Management
STOP_LOSS_PIPS = 500         # Stop loss en pips (50.0 pips)
TAKE_PROFIT_PIPS = 1000      # Take profit en pips (100.0 pips)
MAX_TRADES = 2               # Nombre maximum de positions simultanées
RISK_PERCENT = 5.0           # Risque par trade (% du capital)
```

**Recommandations pour débutants**:
- `LOT_SIZE = 0.01` (minimum)
- `STOP_LOSS_PIPS = 500` (protection 50 pips)
- `MAX_TRADES = 1` (une position à la fois)
- `RISK_PERCENT = 1.0` (1% max par trade)

#### 4. Paramètres du Modèle

```python
# Model Parameters
SEQUENCE_LENGTH = 60          # Historique analysé (60 barres)
PREDICTION_THRESHOLD = 0.6    # Confiance minimum (60%)
MODEL_HIDDEN_DIM = 128        # Doit correspondre à l'entraînement
MODEL_NUM_LAYERS = 4          # Doit correspondre à l'entraînement
MODEL_NUM_HEADS = 8           # Doit correspondre à l'entraînement
```

**Important**: Ces valeurs doivent correspondre à celles utilisées lors de l'entraînement.

#### 5. Intervalle de Prédiction

```python
# Data Parameters
PREDICTION_INTERVAL = 60      # Secondes entre chaque prédiction
```

**Optimisation**:
- `60` secondes = prudent, moins de CPU
- `30` secondes = réactif, plus de CPU
- `300` secondes (5 min) = pour timeframes > M15

### Sauvegarder la Configuration

Enregistrez le fichier `config.py` après vos modifications.

---

## 🎬 Déploiement et Exécution

### Phase 1: Test sur Compte Démo (OBLIGATOIRE)

**⚠️ NE JAMAIS SAUTER CETTE ÉTAPE ⚠️**

#### Préparer l'Environnement de Test

1. **Ouvrir MetaTrader 5** avec votre compte **DÉMO**
2. **Vérifier** que l'algorithmic trading est activé
3. **Ouvrir** un graphique XAUUSD (celui configuré dans `config.py`)

#### Lancer l'EA en Mode Test

```bash
# S'assurer que l'environnement virtuel est activé
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Lancer l'EA
python main.py
```

#### Sortie Console Normale

```
2024-12-27 10:30:00 - INFO - Initializing Transformer EA...
2024-12-27 10:30:01 - INFO - MT5 initialized: (5, 0, 45, 1234)
2024-12-27 10:30:01 - INFO - MT5 login successful
2024-12-27 10:30:02 - INFO - Loaded existing model
2024-12-27 10:30:02 - INFO - Trade manager initialized successfully
2024-12-27 10:30:03 - INFO - ========================================
2024-12-27 10:30:03 - INFO - Transformer EA Started Successfully
2024-12-27 10:30:03 - INFO - Symbol: XAUUSDm | Timeframe: M5
2024-12-27 10:30:03 - INFO - Current balance: $10000.00
2024-12-27 10:30:03 - INFO - ========================================
2024-12-27 10:30:03 - INFO - Starting trading loop...

--- Trading Cycle ---
2024-12-27 10:31:00 - INFO - Signal: HOLD, Confidence: 0.4523
2024-12-27 10:31:00 - INFO - Confidence below threshold (0.60), skipping trade
2024-12-27 10:31:00 - INFO - Open positions: 0 | Balance: $10000.00

--- Trading Cycle ---
2024-12-27 10:32:00 - INFO - Signal: BUY, Confidence: 0.7234
2024-12-27 10:32:00 - INFO - Opening BUY position (confidence: 0.7234)
2024-12-27 10:32:01 - INFO - Buy order successful: ticket=12345678, price=2045.32
2024-12-27 10:32:01 - INFO - SL: 2040.32 | TP: 2055.32
2024-12-27 10:32:01 - INFO - Open positions: 1 | Balance: $10000.00
```

#### Surveillance Initiale (24-48 heures)

**À surveiller**:
- ✅ **Connexion MT5**: Pas d'erreurs de déconnexion
- ✅ **Signaux générés**: HOLD/BUY/SELL avec niveaux de confiance
- ✅ **Exécution des ordres**: Tickets créés dans MT5
- ✅ **Stop Loss / Take Profit**: Correctement placés
- ✅ **Positions fermées**: Automatiquement à SL ou TP

**Vérifier dans MT5**:
1. Onglet **"Boîte à outils"** → **"Historique"**
2. Voir les trades exécutés par l'EA (Magic Number: 234000)
3. Vérifier les prix d'entrée, SL, TP

**Analyser les Résultats**:
- Nombre de trades: 5-20 par jour (M5)
- Win rate: > 50% souhaitable
- Drawdown maximum: < 20% du compte
- Trades fermés correctement (pas d'erreurs)

### Phase 2: Ajustements et Optimisation

Si les résultats du test ne sont pas satisfaisants:

#### Problème: Trop peu de trades
**Solution**: Réduire `PREDICTION_THRESHOLD`
```python
PREDICTION_THRESHOLD = 0.5  # Au lieu de 0.6
```

#### Problème: Trop de pertes
**Solutions**:
1. Augmenter le seuil de confiance:
```python
PREDICTION_THRESHOLD = 0.7  # Plus sélectif
```
2. Élargir le Stop Loss:
```python
STOP_LOSS_PIPS = 700  # Au lieu de 500
```
3. Réentraîner le modèle avec plus de données

#### Problème: Erreurs de connexion MT5
**Solutions**:
- Vérifier que MT5 est ouvert et connecté
- Redémarrer MT5 et l'EA
- Vérifier les identifiants dans `config.py`

### Phase 3: Déploiement en Production (Compte Réel)

**⚠️ ATTENTION: ARGENT RÉEL ⚠️**

**Prérequis avant production**:
- [ ] Tests démo réussis pendant 1-2 semaines minimum
- [ ] Win rate > 50% sur compte démo
- [ ] Aucune erreur système
- [ ] Drawdown acceptable (< 20%)
- [ ] Compréhension totale du fonctionnement

#### Checklist de Déploiement

**1. Configuration Compte Réel**

Modifier `config.py`:
```python
# Utiliser identifiants RÉELS
MT5_LOGIN = 123456789        # Votre compte RÉEL
MT5_PASSWORD = "MotDePasse"  # Mot de passe RÉEL
MT5_SERVER = "VotreBroker-Real"

# Paramètres conservateurs
LOT_SIZE = 0.01              # Commencer petit
PREDICTION_THRESHOLD = 0.65  # Plus sélectif
MAX_TRADES = 1               # Une position max
STOP_LOSS_PIPS = 500         # Protection stricte
```

**2. Vérifications Finales**

- [ ] Capital suffisant (min $500 recommandé)
- [ ] MT5 ouvert avec compte réel
- [ ] Modèle entraîné récemment (< 3 mois)
- [ ] Connexion internet stable
- [ ] PC/serveur allumé H24

**3. Lancement Production**

```bash
# Activer environnement virtuel
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Lancer en production
python main.py
```

**4. Surveillance Continue (Première Semaine)**

**Quotidien**:
- Vérifier équité du compte
- Analyser les trades exécutés
- Contrôler le drawdown
- Vérifier logs d'erreurs

**Hebdomadaire**:
- Calculer le win rate
- Analyser le profit factor (gains/pertes)
- Évaluer les performances vs objectifs
- Ajuster paramètres si nécessaire

#### Arrêt de l'EA

**Arrêt normal**:
```
# Dans le terminal où l'EA tourne
Ctrl+C
```

L'EA effectue un arrêt propre:
```
[INFO] Shutting down EA...
[INFO] Saving model...
[INFO] Closing open positions (optional)
[INFO] MT5 connection closed
[INFO] EA stopped successfully
```

**Arrêt d'urgence**:
1. Fermer le terminal Python
2. Ouvrir MT5
3. Fermer manuellement les positions ouvertes

---

## 📊 Surveillance et Maintenance

### Logs et Historique

#### Fichiers de Logs

Les logs sont affichés dans la console. Pour les sauvegarder:

```bash
# Rediriger vers un fichier
python main.py > ea_log_$(date +%Y%m%d).txt 2>&1

# Ou utiliser nohup (Linux/macOS)
nohup python main.py > ea.log 2>&1 &
```

#### Analyser les Logs

**Rechercher les erreurs**:
```bash
grep "ERROR" ea.log
grep "WARNING" ea.log
```

**Compter les trades**:
```bash
grep "order successful" ea.log | wc -l
```

### Métriques de Performance

#### Indicateurs Clés

1. **Win Rate** = (Trades gagnants / Total trades) × 100
   - Objectif: > 50%

2. **Profit Factor** = Total gains / Total pertes
   - Objectif: > 1.2

3. **Maximum Drawdown** = Plus grande perte depuis le pic
   - Limite: < 20% du capital

4. **Average Trade Duration**
   - Variable selon timeframe

5. **Sharpe Ratio** (si calcul des rendements quotidiens)
   - Objectif: > 1.0

#### Suivi dans MT5

1. **Onglet "Historique"**: Tous les trades
2. **Onglet "Positions"**: Trades ouverts
3. **Graphique de la balance**: Évolution du compte
4. **Rapport détaillé**: Clic droit sur historique → "Rapport"

### Maintenance Régulière

#### Hebdomadaire

- [ ] Vérifier la balance et l'équité
- [ ] Analyser les trades de la semaine
- [ ] Contrôler les erreurs dans les logs
- [ ] Vérifier la connexion MT5

#### Mensuelle

- [ ] Calculer les métriques de performance
- [ ] Évaluer si le modèle est toujours efficace
- [ ] Comparer avec les objectifs fixés
- [ ] Ajuster les paramètres si nécessaire

#### Trimestrielle

- [ ] **Réentraîner le modèle** avec données récentes
- [ ] Backtester sur les 3 derniers mois
- [ ] Optimiser les hyperparamètres
- [ ] Mettre à jour les dépendances Python

### Réentraînement du Modèle

**Quand réentraîner ?**
- Performances dégradées (win rate < 45%)
- Changement de conditions de marché
- Tous les 3-6 mois (recommandé)

**Processus**:
1. Télécharger données récentes (6-12 derniers mois)
2. Utiliser le dashboard Streamlit
3. Entraîner avec nouveaux paramètres si nécessaire
4. Évaluer sur données de validation
5. Si accuracy > modèle actuel → remplacer
6. Tester sur démo avant production

---

## 🔧 Dépannage

### Problèmes Courants et Solutions

#### 1. Erreur: "MT5 initialization failed"

**Causes possibles**:
- MT5 n'est pas ouvert
- MT5 n'est pas installé correctement
- Problème de permissions

**Solutions**:
```bash
# Vérifier si MT5 est installé
# Windows: Chercher dans C:\Program Files\MetaTrader 5\
# Lancer MT5 manuellement d'abord
```

Dans `config.py`, essayer sans credentials:
```python
MT5_LOGIN = None
MT5_PASSWORD = None
MT5_SERVER = None
```

#### 2. Erreur: "Symbol not found" ou "symbol_info returned None"

**Cause**: Le symbole n'est pas disponible dans votre MT5

**Solutions**:
1. Ouvrir MT5 → `Affichage` → `Symboles` (Ctrl+U)
2. Rechercher votre symbole (XAUUSD, XAUUSDm, GOLD)
3. Clic droit → `Afficher le symbole`
4. Mettre à jour `config.py`:
```python
SYMBOL = "XAUUSD"  # Nom exact de votre courtier
```

#### 3. Erreur: "Insufficient margin" ou "Not enough money"

**Cause**: Capital insuffisant pour ouvrir une position

**Solutions**:
- Réduire la taille de lot:
```python
LOT_SIZE = 0.01  # Minimum
```
- Vérifier la marge requise dans MT5
- Augmenter le capital du compte

#### 4. Erreur: "ModuleNotFoundError: No module named 'torch'"

**Cause**: Dépendances non installées

**Solution**:
```bash
# Réinstaller les dépendances
pip install -r requirements.txt

# Ou installer manuellement
pip install torch numpy pandas scikit-learn MetaTrader5 ta
```

#### 5. Erreur: "Model file not found"

**Cause**: Le modèle n'a pas été entraîné

**Solution**:
```bash
# Entraîner le modèle
streamlit run streamlit_dashboard.py
# OU
python train_model.py
```

#### 6. Aucun Trade Exécuté

**Causes et Solutions**:

a. **Confiance trop faible**:
```python
PREDICTION_THRESHOLD = 0.5  # Réduire le seuil
```

b. **MAX_TRADES atteint**:
```python
MAX_TRADES = 3  # Augmenter si nécessaire
```

c. **Modèle non entraîné**: Entraîner le modèle d'abord

d. **Marché fermé**: XAUUSD trade 24/5, vérifier les heures

#### 7. Trades se Ferment Immédiatement

**Causes possibles**:
- Stop Loss trop serré
- Spread trop élevé
- Problème de calcul des niveaux

**Solutions**:
```python
STOP_LOSS_PIPS = 1000  # Élargir le SL
TAKE_PROFIT_PIPS = 2000  # Élargir le TP
```

Vérifier le spread dans MT5 (onglet Observation du marché)

#### 8. Erreur: "Trading is disabled" ou "Trade not allowed"

**Cause**: Trading algorithmique désactivé

**Solution**:
1. MT5: `Outils` → `Options` → `Expert Advisors`
2. ✅ Cocher "Autoriser le trading algorithmique"
3. Redémarrer l'EA

#### 9. Performance GPU/CPU

**Si l'EA est lent**:

```python
# Dans transformer_model.py, forcer CPU
device = torch.device('cpu')  # Au lieu de 'cuda'
```

Ou réduire la complexité:
```python
SEQUENCE_LENGTH = 30  # Au lieu de 60
MODEL_HIDDEN_DIM = 64  # Au lieu de 128
```

#### 10. Crash ou Freeze

**Solutions**:
- Vérifier la RAM disponible (minimum 4GB)
- Fermer autres applications
- Redémarrer MT5 et l'EA
- Vérifier les logs pour l'erreur exacte

### Support et Aide

**Ressources**:
- 📖 **README.md**: Documentation générale
- 🏗️ **ARCHITECTURE.md**: Détails techniques
- 📊 **DASHBOARD_README.md**: Guide du dashboard
- 🧪 **test_ea.py**: Scripts de test

**En cas de problème persistant**:
1. Vérifier les issues GitHub: [github.com/DylaneTrader/XAU-EA-MT5/issues](https://github.com/DylaneTrader/XAU-EA-MT5/issues)
2. Créer une nouvelle issue avec:
   - Description du problème
   - Message d'erreur complet
   - Configuration utilisée (sans mots de passe)
   - Version de Python et MT5

---

## ⚠️ Avertissements Importants

### Risques Financiers

> **Le trading comporte des risques financiers importants. Vous pouvez perdre tout votre capital.**

- ✋ **Jamais** trader avec de l'argent que vous ne pouvez pas vous permettre de perdre
- 📚 **Toujours** tester sur compte démo pendant plusieurs semaines minimum
- 💰 **Commencer** avec le capital minimum et des micro-lots (0.01)
- 📉 **Accepter** que des pertes sont possibles et normales
- 🎯 **Définir** un stop loss global (ex: -20% du compte = arrêt)

### Limitations Techniques

- Le modèle Transformer n'est **pas infaillible**
- Les performances passées **ne garantissent pas** les résultats futurs
- Les conditions de marché changent constamment
- Un réentraînement régulier est **nécessaire**
- L'EA nécessite une connexion internet **stable**
- L'ordinateur doit rester **allumé 24/7** (ou utiliser un VPS)

### Bonnes Pratiques

1. **Test Rigoureux**: Minimum 2 semaines sur démo avec résultats positifs
2. **Démarrage Progressif**: Commencer avec 0.01 lot et 1 trade max
3. **Surveillance Active**: Vérifier quotidiennement pendant le premier mois
4. **Stop Loss Obligatoire**: Ne jamais désactiver le SL
5. **Diversification**: Ne pas investir tout votre capital sur un seul EA
6. **Formation Continue**: Comprendre le trading et l'apprentissage machine
7. **Sauvegarde**: Sauvegarder régulièrement le modèle entraîné

### Responsabilité

- Les auteurs ne sont **pas responsables** des pertes financières
- Cet EA est fourni à des **fins éducatives**
- Vous êtes **seul responsable** de vos décisions de trading
- Consultez un conseiller financier si nécessaire

---

## 📈 Optimisation Avancée

### Backtesting (Tests Historiques)

L'EA n'a pas de backtester intégré, mais vous pouvez:

**Méthode Manuelle**:
1. Créer un script Python qui simule les trades sur données historiques
2. Utiliser le modèle entraîné pour générer des signaux
3. Simuler l'exécution avec slippage et spread
4. Calculer les métriques de performance

**Exemple de Structure**:
```python
# Charger données historiques
df = pd.read_csv('XAUUSDm_M5_20150101_20251226.csv')

# Pour chaque barre
for i in range(len(df)):
    # Obtenir features
    features = prepare_features(df.iloc[max(0, i-60):i])
    
    # Prédire signal
    signal, confidence = model.predict(features)
    
    # Simuler trade si confiance > seuil
    if confidence > 0.6:
        # Enregistrer entry, SL, TP
        # Calculer profit/perte à la fermeture
```

### Multi-Timeframe Analysis

**Concept**: Combiner plusieurs périodes pour confirmation

```python
# Dans config.py
TIMEFRAMES = ["M5", "M15", "H1"]  # Multi-timeframe

# Logique: Trade seulement si signaux alignés
# M5: BUY + M15: BUY + H1: BUY → Exécuter
```

### Gestion Dynamique du Risque

**Position Sizing Basée sur Volatilité**:
```python
# Ajuster lot_size selon l'ATR
current_atr = get_current_atr()
if current_atr > threshold:
    lot_size = LOT_SIZE * 0.5  # Réduire en forte volatilité
else:
    lot_size = LOT_SIZE
```

### Trailing Stop

**Stop Loss Suiveur** (non implémenté par défaut):
```python
# Si profit > X pips, déplacer SL au BE (Break Even)
if current_profit > 50:
    modify_sl_to_be()

# Si profit > Y pips, activer trailing
if current_profit > 100:
    set_trailing_stop(50)  # Trail de 50 pips
```

### Notifications

**Intégrer Telegram** pour alertes:
```bash
pip install python-telegram-bot
```

```python
# Envoyer notification à l'ouverture d'un trade
send_telegram_message(f"🔔 Trade ouvert: {signal} @ {price}")
```

---

## 🎯 Checklist Finale Avant Trading Réel

### Phase Préparation
- [ ] MetaTrader 5 installé et fonctionnel
- [ ] Python 3.8+ installé avec toutes les dépendances
- [ ] Projet cloné et dépendances installées (`pip install -r requirements.txt`)
- [ ] Test de connexion MT5 réussi (`python test_ea.py`)

### Phase Entraînement
- [ ] Modèle entraîné avec données historiques (5000+ barres)
- [ ] Accuracy de test > 55%
- [ ] Fichier `transformer_ea_model.pth` créé et vérifié
- [ ] Compréhension des métriques (confusion matrix, classification report)

### Phase Configuration
- [ ] `config.py` édité avec symbole correct (XAUUSD, XAUUSDm, etc.)
- [ ] Paramètres de risque définis (LOT_SIZE, STOP_LOSS_PIPS, MAX_TRADES)
- [ ] Seuil de confiance ajusté (PREDICTION_THRESHOLD)
- [ ] Identifiants MT5 corrects (pour compte réel)

### Phase Test Démo
- [ ] Tests sur compte démo pendant 1-2 semaines minimum
- [ ] Trades exécutés correctement (vérifiés dans MT5)
- [ ] Stop Loss et Take Profit placés correctement
- [ ] Win rate > 50% sur période de test
- [ ] Aucune erreur système dans les logs
- [ ] Drawdown acceptable (< 20%)

### Phase Production
- [ ] Compte réel avec capital suffisant (min $500)
- [ ] Configuration ajustée pour être plus conservatrice
- [ ] Plan de surveillance quotidienne établi
- [ ] Limite de perte maximale définie (ex: -20% = arrêt)
- [ ] Compréhension totale des risques

### Phase Maintenance
- [ ] Processus de sauvegarde des logs mis en place
- [ ] Calcul hebdomadaire des métriques de performance
- [ ] Plan de réentraînement trimestriel du modèle
- [ ] Veille sur les conditions de marché

---

## 📚 Ressources Complémentaires

### Documentation Technique
- [Documentation MetaTrader 5 Python API](https://www.mql5.com/en/docs/python_metatrader5)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Technical Analysis Library (TA)](https://technical-analysis-library-in-python.readthedocs.io/)

### Formation Trading
- Comprendre les bases du trading Forex/Or
- Apprendre l'analyse technique (indicateurs, patterns)
- Étudier la gestion du risque et du capital
- Se former sur les modèles d'apprentissage machine

### Communauté
- [Forum MQL5](https://www.mql5.com/en/forum)
- [Reddit - r/algotrading](https://www.reddit.com/r/algotrading/)
- [QuantConnect Community](https://www.quantconnect.com/forum)

### Outils Utiles
- **VPS Trading**: Pour exécution 24/7 (Amazon AWS, Google Cloud, VPS Forex spécialisés)
- **TradingView**: Analyse graphique complémentaire
- **Myfxbook**: Suivi de performance publique
- **GitHub**: Versioning de vos modifications

---

## 📞 Contact et Support

Pour toute question ou problème:

1. **Issues GitHub**: [github.com/DylaneTrader/XAU-EA-MT5/issues](https://github.com/DylaneTrader/XAU-EA-MT5/issues)
2. **Documentation**: Consulter README.md, ARCHITECTURE.md
3. **Tests**: Exécuter `python test_ea.py` pour diagnostics

---

## ✅ Récapitulatif en 10 Étapes

1. **Installer** MetaTrader 5 et Python 3.8+
2. **Cloner** le projet et installer dépendances (`pip install -r requirements.txt`)
3. **Configurer** MT5 (activer trading algorithmique)
4. **Entraîner** le modèle (`streamlit run streamlit_dashboard.py`)
5. **Vérifier** que `transformer_ea_model.pth` existe
6. **Éditer** `config.py` avec vos paramètres
7. **Tester** sur compte **DÉMO** (`python main.py`)
8. **Surveiller** pendant 1-2 semaines minimum
9. **Optimiser** paramètres si nécessaire
10. **Déployer** en production (compte réel) avec prudence

---

## 🚀 Bon Trading !

Vous êtes maintenant prêt à intégrer et utiliser l'Expert Advisor Transformer pour le trading automatisé de XAUUSD sur MetaTrader 5.

**Rappels Finaux**:
- 🎯 Toujours commencer par le **compte démo**
- 💡 Le modèle doit être **entraîné** avant utilisation
- 📊 **Surveiller** activement les performances
- ⚠️ **Ne trader** qu'avec de l'argent que vous pouvez perdre
- 🔄 **Réentraîner** le modèle régulièrement

**Bonne chance et tradez prudemment ! 📈💰**

---

*Guide d'intégration MT5 - XAU-EA-MT5 - Version 1.0 - Décembre 2024*
