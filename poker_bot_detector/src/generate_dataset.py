"""
Génération d'un dataset simulé basé sur des décisions préflop de poker.

Objectif de cette version : être plus proche d'une vraie analyse poker.
Pour chaque main, on simule :
- deux cartes privées ;
- la position du joueur, notamment SB et BB ;
- le contexte préflop : pot, grosse blinde, mise à payer, action précédente ;
- une stratégie théorique simplifiée sous forme de fréquences fold/call/raise ;
- l'action réellement jouée par un humain ou un bot ;
- l'écart entre l'action jouée et la stratégie théorique.

Une ligne dans simulated_hands.csv = une décision préflop.
Une ligne dans simulated_players.csv = un joueur résumé par ses statistiques agrégées.
"""

import numpy as np # sert à générer nb aléatoire, calculs num, gérer proba et manipuler des tab
import pandas as pd #crée manipuler tab, crée dataframe

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"] # cartes dispo dans  le jeu
SUITS = ["s", "h", "d", "c"] # s= pique, h = cœur, d= carreau, c = trèfle
DECK = np.array([rank + suit for rank in RANKS for suit in SUITS]) #ici on Construit le paquet de 52 cartes et on combine chaque rang avec chaque couleur

RANK_VALUE = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
} # on associe chaque cartes à un rang

POSITIONS = ["UTG", "MP", "CO", "BTN", "SB", "BB"] # positions à une table de 6 joueurs avec ici UTG = premier à parler préflop, position défavorable, BTN = bouton, meilleure position, SB = petite blinde, BB = grosse blinde.
ACTIONS = ["fold", "call", "raise"] # actions possible
SMALL_BLIND_BB = 0.5 # montant petite blinde
BIG_BLIND_BB = 1.0 #montant big blinde

HUMAN_PROFILES = [ # profil humain
    "recreational_loose",# joueur qui joue trop de mains
    "tight_regular",# joueur régulier
    "aggressive_regular", # joueur régulier agressif
    "beginner", # débutant
    "solid_regular",# joueur solide, plus proche de la théorie
]

BOT_PROFILES = [ # profil bot
    "gto_strict",# bot très proche de la stratégie théorique
    "humanized_gto", # bot qui ajoute un peu de bruit pour paraître humain
    "tight_grinder_bot",# bot serré et régulier
]


def draw_two_cards(rng: np.random.Generator) -> tuple[str, str]:
    """Tire deux cartes différentes dans un paquet de 52 cartes."""
    cards = rng.choice(DECK, size=2, replace=False) # rng.choice choisit deux cartes dans DECK; replace=False : on ne peut pas tirer deux fois la même carte
    return str(cards[0]), str(cards[1]) #On retourne les deux cartes


def get_card_rank(card: str) -> str:
    """Récupère le rang d'une carte : As -> A, Kh -> K."""
    return card[0] # Une carte est codée sur deux caractères : rang + couleur donc ici on prend le rang


def get_card_suit(card: str) -> str:
    """Récupère la couleur d'une carte : As -> s, Kh -> h."""
    return card[1]# Une carte est codée sur deux caractères : rang + couleur donc ici on prend la couleur


def get_hand_class(card_1: str, card_2: str) -> str:
    """
    Convertit deux cartes en classe de main préflop.

    Exemples :
    - As Kh -> AKo ;
    - As Ks -> AKs ;
    - Qh Qd -> QQ.
    """
    rank_1 = get_card_rank(card_1) 
    rank_2 = get_card_rank(card_2) # on récup le rang des deux cartes
    suit_1 = get_card_suit(card_1) 
    suit_2 = get_card_suit(card_2) # on récup couleurs

    value_1 = RANK_VALUE[rank_1]
    value_2 = RANK_VALUE[rank_2] #On transfo les rangs en valeurs numériques pour savoir quelle carte est la plus forte

    if rank_1 == rank_2: 
        return rank_1 + rank_2 # Si les deux cartes ont le même rang, c'est une paire, on renvoie donc ça

    if value_2 > value_1: # Si la deuxième carte est plus forte, on inverse donc l'ordre, cela permet d'avoir toujours la plus grosse carte en premier
        rank_1, rank_2 = rank_2, rank_1
        suit_1, suit_2 = suit_2, suit_1

    suited_marker = "s" if suit_1 == suit_2 else "o" # On ajoute s si les deux cartes sont de même couleur, sinon o
    return rank_1 + rank_2 + suited_marker


def compute_hand_strength(hand_class: str) -> float:
    """
    Calcule une force préflop simplifiée.

    Ce n'est pas un solver GTO complet. C'est une approximation pédagogique basée sur :
    - les paires ;
    - la hauteur des cartes ;
    - les cartes suited ;
    - la connectivité ;
    - les mains dominées et très faibles.
    """
    if len(hand_class) == 2: #Cas particulier des paires
        value = RANK_VALUE[hand_class[0]]
        
        return float(45 + value * 4) # Les paires sont fortes préflop, plus la paire est haute, plus le score augmente

    high_rank = hand_class[0]
    low_rank = hand_class[1]
    suited_marker = hand_class[2] #Si ce n'est pas une paire, la main a trois caractères

    high_value = RANK_VALUE[high_rank]
    low_value = RANK_VALUE[low_rank] # On récupère la valeur numérique des deux cartes

    score = high_value * 3 + low_value * 2 #Score de base fondé sur la hauteur des cartes, la carte haute compte davantage que la carte basse

    if high_rank == "A":
        score += 10
    elif high_rank == "K":
        score += 7
    elif high_rank == "Q":
        score += 5# Bonus pour les grosses carte

    if suited_marker == "s":
        score += 6 #Bonus suited : deux cartes de même couleur ont plus de potentiel

    gap = abs(high_value - low_value) # Connectivité : deux cartes proches peuvent faire plus facilement des quintes
    if gap == 1:
        score += 7
    elif gap == 2:
        score += 4
    elif gap == 3:
        score += 2
    
    if suited_marker == "o" and high_value < 12 and gap >= 5:
        score -= 9# Malus pour mains faibles, offsuit et déconnectées

   
    if high_value >= 10 and low_value >= 10:
        score += 4 # Petit bonus pour broadways

    return float(np.clip(score, 5, 100))

# Ici on classe les mains en fonction de leur force
def classify_hand_family(hand_class: str, hand_strength: float) -> str: 
    """Classe une main dans une famille poker lisible."""
    if len(hand_class) == 2: # Si la main est une paire 
        value = RANK_VALUE[hand_class[0]] 
        if value >= 12: # QQ, KK, AA pari haute
            return "premium_pair"
        if value >= 7: # 77 à JJ 
            return "medium_pair"
        return "low_pair" # 22 à 66

    high_rank = hand_class[0]
    low_rank = hand_class[1]
    suited = hand_class[2] == "s" 
    high_value = RANK_VALUE[high_rank]
    low_value = RANK_VALUE[low_rank]
    gap = abs(high_value - low_value)# Pour les mains non paires, on récup les carac

    if high_rank == "A" and low_value >= 10: #as avec grosse cartes 
        return "premium_broadway"
    if high_value >= 10 and low_value >= 10: # deux grosse cartes
        return "broadway"
    if high_rank == "A" and suited: # As avec autres carte
        return "suited_ace"
    if suited and gap <= 2 and low_value >= 5: # suite avec écrat max de 2 carte
        return "suited_connector"
    if hand_strength < 35: # mauvaise main
        return "trash"
    if 35 <= hand_strength < 55: # main difficile/bof à jouer
        return "marginal"
    return "standard" #  # Main standard si aucune catégorie précédente ne correspond

# La position joue un rôle important dans la force de notre jeu, on la prends donc en compte, les valeurs des bonus sont à débattre/expliquer
def position_bonus(position: str) -> float:
    """Bonus de jouabilité selon la position."""
    return {
        "UTG": -9.0,
        "MP": -5.0,
        "CO": 1.0,
        "BTN": 7.0,
        "SB": -1.0,
        "BB": 4.0,
    }[position] 

# Fonction pour prendre en compte le prix de la blinde et réequilibrer les stats en fonction
def posted_blind(position: str) -> float:
    """Retourne la blinde déjà postée par la position."""
    if position == "SB": #SB a déjà posté 0.5 BB
        return SMALL_BLIND_BB
    if position == "BB": #BB a déjà posté 1 BB
        return BIG_BLIND_BB
    return 0.0 # les autres positions n'ont rien posté

# Fonction qui va servir de base pour le comportement des joueurs, on définit le pot, les stacks (la profondeur) et les pots odds
def simulate_preflop_context(position: str, rng: np.random.Generator) -> dict:
    """
    Simule un contexte préflop minimal : action précédente, pot, mise à payer,
    stack effectif et pot odds.
    """
    table_size = 6 ## On travaille sur une table 6-max pour simplifier
    stack_bb = float(np.clip(rng.normal(loc=100, scale=28), 20, 220)) #Stack du joueur en grosses blindes, on simule autour de 100 BB, avec des varia
    effective_stack_bb = float(np.clip(stack_bb + rng.normal(loc=0, scale=12), 15, 220)) #Stack effectif : ici on a une profondeur réelle contre l'adversaire, on ne peut gagner/perdre que jusqu'au plus petit stack

    if position in {"UTG", "MP"}:
        previous_action = str(rng.choice(["unopened", "limped"], p=[0.86, 0.14]))
    elif position in {"CO", "BTN"}:
        previous_action = str(rng.choice(["unopened", "limped", "open_raise"], p=[0.62, 0.13, 0.25]))
    elif position == "SB":
        previous_action = str(rng.choice(["unopened", "limped", "open_raise"], p=[0.58, 0.12, 0.30])) #Simulation de l'action précédente selon la position, plus on parle tard, plus il y a de chances qu'une action ait déjà eu lieu
    else:  #En BB, il est fréquent de faire face à une relance
        previous_action = str(rng.choice(["unopened", "limped", "open_raise", "three_bet"], p=[0.24, 0.16, 0.52, 0.08]))

    facing_raise = previous_action in {"open_raise", "three_bet"} # Indique si le joueur fait face à une relance
    facing_3bet = previous_action == "three_bet"# Indique si le joueur fait face à un 3-bet

    base_pot = SMALL_BLIND_BB + BIG_BLIND_BB # Pot initial composé de la petite blinde et de la grosse blinde
    current_bet_bb = 0.0 # montant de la mise actuelle à égaler

    if previous_action == "limped": #Si quelqu'un a limpé, la mise à égaler est la grosse blinde
        current_bet_bb = BIG_BLIND_BB
        pot_size_bb = base_pot + rng.integers(1, 4) * BIG_BLIND_BB
    elif previous_action == "open_raise": #Si quelqu'un a open raise, on simule un sizing autour de 2.5 BB
        current_bet_bb = float(np.clip(rng.normal(loc=2.5, scale=0.35), 2.0, 3.5))
        pot_size_bb = base_pot + current_bet_bb + rng.uniform(0.0, 1.0)
    elif previous_action == "three_bet": #Si quelqu'un a 3-bet, la mise à payer est beaucoup plus élevée
        current_bet_bb = float(np.clip(rng.normal(loc=8.5, scale=1.2), 6.5, 11.5))
        pot_size_bb = base_pot + current_bet_bb + rng.uniform(2.0, 5.0)
    else: #Si personne n'a relancé
        current_bet_bb = BIG_BLIND_BB if position in {"SB", "BB"} else 0.0
        pot_size_bb = base_pot

    amount_to_call_bb = max(current_bet_bb - posted_blind(position), 0.0) # montant réel à payer après prise en compte de la blinde déjà postée
    pot_after_call = pot_size_bb + amount_to_call_bb #Pot après paiement
    pot_odds = amount_to_call_bb / pot_after_call if pot_after_call > 0 else 0.0 #Pot odds = montant à payer / pot après call, plus les pot odds sont faibles, plus call est attractif

    return {
        "table_size": table_size,
        "stack_bb": stack_bb,
        "effective_stack_bb": effective_stack_bb,
        "small_blind_bb": SMALL_BLIND_BB,
        "big_blind_bb": BIG_BLIND_BB,
        "posted_blind_bb": posted_blind(position),
        "previous_action": previous_action,
        "facing_raise": int(facing_raise),
        "facing_3bet": int(facing_3bet),
        "pot_size_bb": float(pot_size_bb),
        "current_bet_bb": float(current_bet_bb),
        "amount_to_call_bb": float(amount_to_call_bb),
        "pot_odds": float(np.clip(pot_odds, 0.0, 0.8)),
    } # on retourne tout


def normalize_probabilities(fold: float, call: float, raise_: float) -> dict:
    """Force des fréquences fold/call/raise valides."""
    values = np.array([fold, call, raise_], dtype=float) # On crée un tableau avec les trois probabilités
    values = np.clip(values, 0.0, None) # on empeche val néga
    total = values.sum()# calcul somme des trois proba
    if total == 0: # si toutres las val nulles on met fold =1
        values = np.array([1.0, 0.0, 0.0])
    else: #Sinon on normalise pour que la somme fasse 1
        values = values / total
    return {"fold": float(values[0]), "call": float(values[1]), "raise": float(values[2])}

## On utilise ici une sigmoide pour transformer un score (relatif à la force de la main et à la position) en une valeur comprise entre 0 et 1. Plus le sore de jeu est élevé par rapport au seil défini, plus la sigmoide se rapproche de 1, qui est la probabilité de faire l'action
def get_gto_frequencies(hand_strength: float, position: str, context: dict) -> dict:
    """
    Retourne une stratégie préflop simplifiée sous forme de fréquences.

    Le modèle prend en compte :
    - la force de la main ;
    - la position ;
    - l'action précédente ;
    - la grosse blinde / petite blinde ;
    - les pot odds ;
    - la profondeur de stack.
    """
    playability = hand_strength + position_bonus(position) #Jouabilité = force de la main + bonus/malus de position
    pot_odds = context["pot_odds"] #On récup les pot odds
    effective_stack = context["effective_stack_bb"] #On récup la profondeur de stack
    previous_action = context["previous_action"] #On récupère last action

    # Les mains à potentiel gagnent en jouabilité quand les stacks sont profonds.
    if effective_stack > 120 and hand_strength >= 45:
        playability += 2.0
    if effective_stack < 35: #Quand les stacks sont courts les mains fortes gagnent en valeur, les mains moyennes/faibles perdent en jouabilité
        playability += 2.0 if hand_strength >= 72 else -2.0

    # BB/SB : la blinde déjà postée rend certains calls plus défendables, on a déjà investi 1 BB, donc certains calls deviennent plus défendables
    if position == "BB":
        playability += 3.0 + 10.0 * pot_odds
    elif position == "SB": #En SB on est hors position postflop donc un léger malus
        playability -= 1.0

    if previous_action in {"unopened", "limped"}:
        # Spot d'ouverture ou d'isolation : on raise plus souvent les mains fortes.
        raise_score = (playability - 56) / 11 #Score de raise : plus var playability dépasse 56, plus raise devient probable
        call_score = (playability - 43) / 12 #Score de call : plus var playability dépasse 43, plus call devient probable
        raise_prob = 1 / (1 + np.exp(-raise_score)) #ici on transforme un score en probabilité entre 0 et 1
        call_prob = (1 / (1 + np.exp(-call_score))) * (1 - 0.55 * raise_prob) #ici onn limite le call quand le raise devient déjà très probable
        fold_prob = 1 - raise_prob - call_prob #Le reste passe fold
        return normalize_probabilities(fold_prob, call_prob, raise_prob)

    if previous_action == "open_raise":
        # Face à une relance : call/3-bet/fold selon force, pot odds et position.
        threebet_score = (playability - 73) / 10 #face à une relance, raise correspond à un 3-bet
        call_score = (playability + 20 * pot_odds - 53) / 10 #call dépend de la main aussi des pot odds
        raise_prob = 1 / (1 + np.exp(-threebet_score))
        call_prob = (1 / (1 + np.exp(-call_score))) * (1 - 0.45 * raise_prob)
        fold_prob = 1 - raise_prob - call_prob
        return normalize_probabilities(fold_prob, call_prob, raise_prob)

    # Face à un 3-bet : ranges très resserrées.
    raise_score = (playability - 84) / 8
    call_score = (playability + 12 * pot_odds - 68) / 9
    raise_prob = 1 / (1 + np.exp(-raise_score))
    call_prob = (1 / (1 + np.exp(-call_score))) * (1 - 0.5 * raise_prob)
    fold_prob = 1 - raise_prob - call_prob
    return normalize_probabilities(fold_prob, call_prob, raise_prob)


def recommended_action(gto_freqs: dict) -> str:
    """Retourne l'action ayant la fréquence théorique la plus forte."""
    return max(gto_freqs, key=gto_freqs.get)#si fold = 0.10, call = 0.20, raise = 0.70 alors l'action recommandée= raise


def sample_action_from_frequencies(freqs: dict, rng: np.random.Generator) -> str:
    """Échantillonne fold/call/raise à partir d'une distribution."""
    return str(rng.choice(ACTIONS, p=[freqs["fold"], freqs["call"], freqs["raise"]]))    # fonction disposi si on veut tirer directement une action depuis une distrib de fréquences


def choose_human_action(gto_freqs: dict, hand_strength: float, context: dict, hand_index: int, total_hands: int, profile: str, rng: np.random.Generator) -> str:
    """
    Simule l'action d'un humain.

    L'humain ne suit pas une stratégie optimale pure. Il peut être loose, tight,
    agressif, débutant ou régulier. La fatigue augmente aussi légèrement les erreurs.
    """
    progress = hand_index / max(total_hands - 1, 1) # Progression dans l'échantillon de mains plus on avance et plus la fatigue peut créer du bruit
    freqs = gto_freqs.copy() #copie des fréquences théo pour les modif 

    # Plus la session avance, plus les fréquences humaines se bruitent.
    noise_level = 0.10 + 0.10 * progress

    if profile == "recreational_loose":
        freqs["fold"] *= 0.72
        freqs["call"] *= 1.28
        freqs["raise"] *= 1.08
        noise_level += 0.05 
    elif profile == "tight_regular":
        freqs["fold"] *= 1.22
        freqs["call"] *= 0.86
        freqs["raise"] *= 0.92
    elif profile == "aggressive_regular":
        freqs["fold"] *= 0.92
        freqs["call"] *= 0.78
        freqs["raise"] *= 1.35
    elif profile == "beginner":
        freqs["fold"] *= 0.88
        freqs["call"] *= 1.40
        freqs["raise"] *= 0.82
        noise_level += 0.10
    elif profile == "solid_regular":
        noise_level -= 0.03 # profil de joueur diverse

    # Les mains marginales sont plus difficiles pour les humains.
    if 42 <= hand_strength <= 62:
        noise_level += 0.07

    freqs = normalize_probabilities(freqs["fold"], freqs["call"], freqs["raise"]) # On renormalise les probabilités modifiées

    # Mélange avec une distribution plus bruitée.
    random_style = rng.dirichlet([1.4, 1.3, 1.2]) #distribution aléatoire représentant l'imprévisibilité d'un humain
    final_probs = np.array([freqs["fold"], freqs["call"], freqs["raise"]])
    final_probs = (1 - noise_level) * final_probs + noise_level * random_style
    final_probs = final_probs / final_probs.sum()

    return str(rng.choice(ACTIONS, p=final_probs)) # On tire l'action humaine


def choose_bot_action(gto_freqs: dict, hand_strength: float, context: dict, profile: str, rng: np.random.Generator) -> str:
    """
    Simule l'action d'un bot.

    Les bots suivent très fortement les fréquences théoriques. Certains sont stricts,
    d'autres ajoutent du bruit pour paraître plus humains.
    """
    freqs = gto_freqs.copy()

    if profile == "gto_strict": #On copie les fréquences théoriques
        noise_level = 0.015
    elif profile == "humanized_gto": #Bot très strict
        noise_level = 0.07
    else:  # tight_grinder_bot
        noise_level = 0.04
        freqs["fold"] *= 1.08
        freqs["call"] *= 0.92
        freqs["raise"] *= 1.03

    if 42 <= hand_strength <= 62: # Les mains marginales restent complexe même pour un bot
        noise_level += 0.02

    freqs = normalize_probabilities(freqs["fold"], freqs["call"], freqs["raise"]) #On renormalise
    base_probs = np.array([freqs["fold"], freqs["call"], freqs["raise"]])
    random_style = rng.dirichlet([1.1, 1.1, 1.1]) # Distribution très légère de bruit
    final_probs = (1 - noise_level) * base_probs + noise_level * random_style
    final_probs = final_probs / final_probs.sum() # Le bot reste proche de la théorie

    return str(rng.choice(ACTIONS, p=final_probs))


def compute_l1_distance(action: str, gto_freqs: dict) -> float:
    """Distance L1 entre l'action jouée en one-hot et la distribution théorique."""
    player_distribution = {candidate: 1.0 if candidate == action else 0.0 for candidate in ACTIONS} #Distribution one-hot de l'action jouée
    return float(sum(abs(player_distribution[candidate] - gto_freqs[candidate]) for candidate in ACTIONS)) #somme des écarts absolus entre distribution joueur et distribution théo


def simulate_decision_time(is_bot: int, action: str, hand_strength: float, context: dict, rng: np.random.Generator) -> float:
    """Simule un temps de décision en secondes."""
    if is_bot:
        base_time = rng.normal(loc=2.25, scale=0.38) #Temps de base selon type joueur
    else:
        base_time = rng.normal(loc=5.35, scale=1.55)

    if action == "raise": #relance demande réflexion
        base_time += rng.normal(loc=0.38, scale=0.18)

    if 42 <= hand_strength <= 62 or context["facing_raise"]: #mains marginales ou spots face à une relance plus complexes
        base_time += rng.normal(loc=0.65 if not is_bot else 0.12, scale=0.20)

    return float(np.clip(base_time, 0.5, 22.0)) #borne le temps pour éviter val absurdes


def simulate_bet_size_bb(is_bot: int, action: str, context: dict, rng: np.random.Generator) -> float:
    """Simule un sizing en grosses blindes."""
    if action == "fold":#si le joueur fold, il ne mise rien
        return 0.0

    amount_to_call = context["amount_to_call_bb"] #Montant pour suivre
    previous_action = context["previous_action"] #Action précédente

    if action == "call":  # Si le joueur call, il paie le montant demandé
        return float(max(amount_to_call, BIG_BLIND_BB if previous_action in {"unopened", "limped"} else amount_to_call))

    # Raise / open raise / 3-bet / 4-bet simplifié.
    if previous_action in {"unopened", "limped"}: # Cas d'un open raise ou d'un raise après limp
        loc = 2.45 if is_bot else 2.55 # il s'agit du premier raise, le bot relance de 2.45 en moyenne sinon 2.55 pour l'humain
        scale = 0.15 if is_bot else 0.38 # c'est ici que la différence est flagrante
        return float(np.clip(rng.normal(loc=loc, scale=scale), 2.0, 4.5))

    if previous_action == "open_raise": #Cas d'un 3-bet face à un open rais
        loc = 8.2 if is_bot else 8.5
        scale = 0.45 if is_bot else 1.25 # pour un trois bet, l'écart type est plus grand ce qui coincide avec la difficulté de sizer ce coup
        return float(np.clip(rng.normal(loc=loc, scale=scale), 5.5, 13.0))
    # ici on est dans la situation du 4 bet face à 3_bet, plus rare et bcp plus imprécis ( dans la réalité, ça part souvent à tapis)
    loc = 19.0 if is_bot else 20.0 
    scale = 1.0 if is_bot else 3.0
    return float(np.clip(rng.normal(loc=loc, scale=scale), 12.0, 35.0))

# indicateur d'entropie, qu'on va utiliser pour le random forest ensuite
def action_entropy(actions: pd.Series) -> float:
    """Calcule une entropie normalisée entre 0 et 1 sur fold/call/raise."""
    probabilities = actions.value_counts(normalize=True) #Cette ligne regarde l'historique des actions du joueur (ex: sur 100 mains) et calcule la fréquence de chaque action
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0) # calcule l'entropie de Shannon
    return float(entropy / np.log2(len(ACTIONS))) # Normalise l'entropie afin d'avoir une valeur entre 0 et 1


def simulate_player_hands(player_id: str, is_bot: int, n_hands: int, rng: np.random.Generator) -> pd.DataFrame:
    """Simule toutes les décisions préflop d'un joueur."""
    rows = [] #Liste qui va recevoir toutes les mains simulées du joueur
    profile = str(rng.choice(BOT_PROFILES if is_bot else HUMAN_PROFILES))  #on choisit un profil humain ou bot

    for hand_index in range(n_hands): #On simule n_hands mains pour ce joueur
        card_1, card_2 = draw_two_cards(rng) #Tirage de deux cartes privées
        hand_class = get_hand_class(card_1, card_2) # Classe de main : AA, AKo, AKs, ...
        hand_strength = compute_hand_strength(hand_class) #Force théo simplifiée de la main
        hand_family = classify_hand_family(hand_class, hand_strength) # famille de main : trash, marginal, suited_connector, premium_pair ...
        position = str(rng.choice(POSITIONS)) #Position du joueur à table
        context = simulate_preflop_context(position, rng) #Contexte préflop : pot, mise à payer, blindes, action précédente
        gto_freqs = get_gto_frequencies(hand_strength, position, context) #gto_freqs (pour GTO Frequencies) est un dictionnaire qui contient les probabilités optimales pour chaque action possible.
        gto_action = recommended_action(gto_freqs) #gto_action est une chaîne de caractères ("fold", "call" ou "raise") qui représente l'action la plus logique ou la plus fréquente selon la théorie

        if is_bot: #Choix de l'action réelle selon le type de joueur
            player_action = choose_bot_action(gto_freqs, hand_strength, context, profile, rng)
        else:
            player_action = choose_human_action(gto_freqs, hand_strength, context, hand_index, n_hands, profile, rng) #ici on ajoute n_hands et hands_index pour ajouetr le facteur de fatigue de l'humain

        chosen_action_probability = gto_freqs[player_action] #probabilité théo de l'action réellement choisie
        gto_l1_distance = compute_l1_distance(player_action, gto_freqs) # On calcul la distance entre le play (action joueur) et la théorie pour chaque main préflop
        decision_time = simulate_decision_time(is_bot, player_action, hand_strength, context, rng) # Temps de décision simulé
        bet_size_bb = simulate_bet_size_bb(is_bot, player_action, context, rng) # Sizing simulé en grosses blindes
 
        rows.append(
            {
                "player_id": player_id,
                "hand_id": hand_index + 1,
                "is_bot": is_bot,
                "player_profile": profile,
                "table_size": context["table_size"],
                "position": position,
                "small_blind_bb": context["small_blind_bb"],
                "big_blind_bb": context["big_blind_bb"],
                "posted_blind_bb": context["posted_blind_bb"],
                "stack_bb": context["stack_bb"],
                "effective_stack_bb": context["effective_stack_bb"],
                "previous_action": context["previous_action"],
                "facing_raise": context["facing_raise"],
                "facing_3bet": context["facing_3bet"],
                "pot_size_bb": context["pot_size_bb"],
                "current_bet_bb": context["current_bet_bb"],
                "amount_to_call_bb": context["amount_to_call_bb"],
                "pot_odds": context["pot_odds"],
                "card_1": card_1,
                "card_2": card_2,
                "hand_class": hand_class,
                "hand_family": hand_family,
                "hand_strength": hand_strength,
                "gto_fold_probability": gto_freqs["fold"],
                "gto_call_probability": gto_freqs["call"],
                "gto_raise_probability": gto_freqs["raise"],
                "gto_action": gto_action,
                "player_action": player_action,
                "chosen_action_probability": chosen_action_probability,
                "gto_l1_distance": gto_l1_distance,
                "is_gto_correct": int(player_action == gto_action),
                "decision_time": decision_time,
                "bet_size_bb": bet_size_bb,
                "bet_size": bet_size_bb,
            }
        ) #On ajoute ligne dans dataset main par main

    return pd.DataFrame(rows) # On convertit toutes les lignes en Df

# Ces fonctions servent à s'assurer que le programme ne va pas planter, par exemple si un joueur n'a jamais rencontré de situation de 4 Bet, il peut y avoir un Nan dans sa colonne, ce qui peut entrainer des divisions par zéro et faire planter l'algo
def _safe_mean(series: pd.Series) -> float:
    """Retourne une moyenne sûre, égale à 0 si la série est vide."""
    if len(series) == 0:
        return 0.0
    return float(series.mean())


def _safe_rate(df: pd.DataFrame, condition_column: str = "is_gto_correct") -> float:
    """Retourne une moyenne sûre, égale à 0 si le DataFrame est vide."""
    if len(df) == 0:
        return 0.0
    return float(df[condition_column].mean())


def aggregate_player_features(player_hands: pd.DataFrame) -> dict:
    """Agrège les décisions préflop d'un joueur en variables de modèle."""
    actions = player_hands["player_action"]
    hands_played = len(player_hands)

    vpip = float((actions != "fold").mean()) #ici on calcul les métrics pour le random forest, fréquence de mise d'argent dans le pot; VPIP : fréquence à laquelle le joueur met volontairement de l'argent dans le pot
    pfr = float((actions == "raise").mean()) #Fréquence de relance

    calls = int((actions == "call").sum())
    raises = int((actions == "raise").sum())
    af = float(raises / max(calls, 1)) # "facteur d'agression" simplifié

    played_hands = player_hands[player_hands["player_action"] != "fold"] #Mains jouées volontairement
    bet_size_mean = _safe_mean(played_hands["bet_size_bb"]) 
    bet_size_std = float(played_hands["bet_size_bb"].std(ddof=0)) if len(played_hands) else 0.0 #Moyenne et écart-type des sizings
    
    # ici on filtre les mains selon leur force relative afin d'analyser ensuite les plays des joueurs en fonction de la force de leur main
    weak_hands = player_hands[player_hands["hand_strength"] < 45]
    trash_hands = player_hands[player_hands["hand_family"] == "trash"]
    premium_hands = player_hands[player_hands["hand_strength"] >= 78]
    strong_hands = player_hands[player_hands["hand_strength"] >= 70]
    marginal_hands = player_hands[(player_hands["hand_strength"] >= 42) & (player_hands["hand_strength"] <= 62)]
    
    # On calcule les différents ratio pour le random forest ensuite 
    weak_hand_play_rate = float((weak_hands["player_action"] != "fold").mean()) if len(weak_hands) else 0.0 # On regarde lorsque le joueur a joué des mains faible, Taux de jeu des mains faibles
    trash_hand_vpip = float((trash_hands["player_action"] != "fold").mean()) if len(trash_hands) else 0.0 # On regarde lorsque le joueur a joué des mains dites "poubelles", ici c'est un bon indicateur de bot ou non, si son ratio est proche de 0 et que le joueur est gagnant, il se peut que ce soit un bot
    premium_hand_play_rate = float((premium_hands["player_action"] != "fold").mean()) if len(premium_hands) else 0.0 # Taux de jeu des mains premium
    strong_hand_aggression_rate = float((strong_hands["player_action"] == "raise").mean()) if len(strong_hands) else 0.0 #Taux de raise avec les mains fortes
    marginal_hand_error_rate = float(1 - marginal_hands["is_gto_correct"].mean()) if len(marginal_hands) else 0.0 # Cet indicateur est pertinent car c'est ici que l'humain fait le plus d'erreur, = Taux d'erreur sur les mains marginales

    fold_spots = player_hands[player_hands["gto_action"] == "fold"]
    call_spots = player_hands[player_hands["gto_action"] == "call"]
    raise_spots = player_hands[player_hands["gto_action"] == "raise"] #Spots par action théo
    open_spots = player_hands[player_hands["previous_action"].isin(["unopened", "limped"])]
    defense_spots = player_hands[player_hands["previous_action"] == "open_raise"]
    threebet_spots = player_hands[player_hands["previous_action"] == "three_bet"]
    blind_defense_spots = player_hands[(player_hands["position"].isin(["SB", "BB"])) & (player_hands["facing_raise"] == 1)]
    sb_steal_spots = player_hands[(player_hands["position"] == "SB") & (player_hands["previous_action"] == "unopened")]
    pot_odds_call_spots = player_hands[(player_hands["amount_to_call_bb"] > 0) & (player_hands["pot_odds"] <= 0.30)] #Spots préflop spécifiques

    #Ici on va comparer la pente de fatigue des joueurs entre la première moitié de la session et la deuxième, normalement pour un humain, il devrait y avoir un écart significatif et pour un bot 0
    first_half = player_hands.iloc[: hands_played // 2]
    second_half = player_hands.iloc[hands_played // 2 :] # ici ccomparaison entre première moitié et deuxième moitié
    fatigue_slope = float(second_half["is_gto_correct"].mean() - first_half["is_gto_correct"].mean()) #ici si quali baisse dans la deuxième moitié la pente est néga

    gto_similarity = float(player_hands["is_gto_correct"].mean()) #Similarité globale avec l'action recommandée

    return {
        "vpip": vpip,
        "pfr": pfr,
        "af": af,
        "action_entropy": action_entropy(actions),
        "decision_time_mean": float(player_hands["decision_time"].mean()),
        "decision_time_std": float(player_hands["decision_time"].std(ddof=0)),
        "bet_size_mean": bet_size_mean,
        "bet_size_std": bet_size_std,
        "gto_similarity": gto_similarity,
        "gto_deviation_rate": float(1 - gto_similarity),
        "mean_gto_action_probability": float(player_hands["chosen_action_probability"].mean()),
        "mean_l1_gto_distance": float(player_hands["gto_l1_distance"].mean()),
        "std_l1_gto_distance": float(player_hands["gto_l1_distance"].std(ddof=0)),
        "avg_hand_strength": float(player_hands["hand_strength"].mean()),
        "weak_hand_play_rate": weak_hand_play_rate,
        "trash_hand_vpip": trash_hand_vpip,
        "premium_hand_play_rate": premium_hand_play_rate,
        "strong_hand_aggression_rate": strong_hand_aggression_rate,
        "marginal_hand_error_rate": marginal_hand_error_rate,
        "gto_fold_follow_rate": _safe_rate(fold_spots),
        "gto_call_follow_rate": _safe_rate(call_spots),
        "gto_raise_follow_rate": _safe_rate(raise_spots),
        "open_raise_accuracy": _safe_rate(open_spots),
        "defense_accuracy": _safe_rate(defense_spots),
        "threebet_accuracy": _safe_rate(threebet_spots),
        "blind_defense_rate": float((blind_defense_spots["player_action"] != "fold").mean()) if len(blind_defense_spots) else 0.0,
        "blind_defense_accuracy": _safe_rate(blind_defense_spots),
        "sb_steal_attempt_rate": float((sb_steal_spots["player_action"] == "raise").mean()) if len(sb_steal_spots) else 0.0,
        "pot_odds_call_accuracy": _safe_rate(pot_odds_call_spots),
        "hands_played": int(hands_played),
        "sessions_played": int(np.ceil(hands_played / 60)),
        "fatigue_slope": fatigue_slope,
    } # dictio final: une clé = une colonne du dataset joueur


def generate_dataset(n_players: int, bot_ratio: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Génère le dataset joueur agrégé et le dataset détaillé main par main."""
    rng = np.random.default_rng(random_state) # random_state permet la reproductabilité de la simulation, il s'agit de la graine pour l'aléa, par exemple lors du 1 er tirage, le joueur 1 recoit AKs, lors d'un relancement, il reçoit de nouveau AKs au premier tirage
 
    n_bots = int(n_players * bot_ratio)    # Ici on détermine le nombre d'humains et de bots
    n_humans = n_players - n_bots

    labels = np.array([0] * n_humans + [1] * n_bots) #Création des labels : 0 = humain, 1 = bot
    rng.shuffle(labels) # On mélange les joueurs bot et humains

    player_rows = [] # liste jouer
    hand_rows = [] # liste main

    for player_index, is_bot in enumerate(labels):
        player_id = f"P{str(player_index + 1).zfill(5)}" #Identifiant propre du joueur : P00001, P00002,

        # Même ordre de grandeur de volume entre humains et bots : le modèle doit
        # surtout apprendre les décisions, pas seulement le nombre de mains.
        n_hands = int(rng.integers(low=50, high=91))

        player_hands = simulate_player_hands(player_id, int(is_bot), n_hands, rng) # Simulation de toutes les mains du joueur
        features = aggregate_player_features(player_hands) #Agrégation des mains en var joueur

        player_rows.append({"player_id": player_id, "is_bot": int(is_bot), **features}) #Ajout de la ligne joueur
        hand_rows.extend(player_hands.to_dict("records")) #Ajout des lignes main par main

        # Affichage léger de progression : utile car la génération simule beaucoup de mains.
        if (player_index + 1) % 250 == 0 or (player_index + 1) == n_players:
            print(f"Progression génération : {player_index + 1}/{n_players} joueurs simulés")

    players_dataset = pd.DataFrame(player_rows) #C'est ce dataset qui est envoyé au Random forest
    hands_dataset = pd.DataFrame(hand_rows)  # Des milliers de lignes. Utile pour l'analyse comportementale profonde.

    return players_dataset, hands_dataset


def save_dataset(players_dataset: pd.DataFrame, output_path) -> None:
    """Sauvegarde le dataset agrégé par joueur."""
    output_path.parent.mkdir(parents=True, exist_ok=True) ## Crée le dossier parent si nécessaire
    players_dataset.to_csv(output_path, index=False)


def save_hands_dataset(hands_dataset: pd.DataFrame, output_path) -> None: 
    """Sauvegarde le dataset détaillé main par main."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hands_dataset.to_csv(output_path, index=False)
