import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)  # Permite accesul de la diferite origini

base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, 'questions.json')

with open(json_path, 'r', encoding='utf-8') as f:
    questions_data = json.load(f)

# Definirea profilurilor pentru fiecare facultate/domeniu
domain_profiles = {
    "informatica": {
        "name": "Informatică",
        "description": "Potrivit pentru persoane pasionate de fundamentele informaticii, algoritmi și dezvoltare software, care preferă un mediu academic cu accent pe cercetare.",
        "strengths": ["algoritmi avansați", "fundamentele informaticii", "programare", "cercetare științifică"],
        "career_paths": ["cercetător", "dezvoltator software", "data scientist", "profesor"],
        "learning_style": "teoretic cu aplicații practice în domenii fundamentale",
        "keywords": ["theoretical", "software", "research", "algorithms", "math", "academic", "abstract", "ai"],
        "universities": [
            {
                "name": "Universitatea București",
                "faculty": "Facultatea de Matematică și Informatică",
                "program": "Informatică",
                "strengths": "Pregătire solidă în bazele teoretice ale informaticii, mediu academic stimulativ, profesori cu experiență în cercetare"
            },
            {
                "name": "Universitatea Babeș-Bolyai",
                "faculty": "Facultatea de Matematică și Informatică",
                "program": "Informatică",
                "strengths": "Recunoscută internațional, oportunități excelente de cercetare, comunitate academică diversă"
            }
        ]
    },
    "informatica_economica": {
        "name": "Informatică Economică",
        "description": "Potrivit pentru persoane interesate de aplicarea tehnologiei în domeniul economic și de afaceri, preferând un mediu dinamic orientat spre business.",
        "strengths": ["economie digitală", "analiză de date", "sisteme informatice pentru afaceri"],
        "career_paths": ["analist de business", "data scientist", "manager IT", "consultant"],
        "learning_style": "echilibrat între teorie și aplicații practice în domeniul economic",
        "keywords": ["business", "economic", "financial", "erp", "business_analytics", "applications", "case_study",
                     "fintech"],
        "universities": [
            {
                "name": "ASE București",
                "faculty": "Facultatea de Cibernetică, Statistică și Informatică Economică",
                "program": "Informatică Economică",
                "strengths": "Combinație excelentă între pregătirea in domeniul IT și pregatirea in domeniul economic, colaborări strânse cu mediul de afaceri, oportunități de practică în companii de top"
            }
        ]
    },
    "automatica": {
        "name": "Automatică și Calculatoare",
        "description": "Potrivit pentru persoane interesate de inginerie, robotică și sisteme de calcul, care preferă un mediu tehnic axat pe practică.",
        "strengths": ["programare avansată", "electronică", "robotică", "sisteme embedded"],
        "career_paths": ["inginer software", "dezvoltator embedded", "specialist robotică", "inginer de sistem"],
        "learning_style": "orientat spre practică și inovație tehnică",
        "keywords": ["hardware", "systems", "engineering", "embedded", "low_level", "robotics", "industrial", "iot",
                     "technical"],
        "universities": [
            {
                "name": "Universitatea Politehnica București",
                "faculty": "Facultatea de Automatică și Calculatoare",
                "program": "Ingineria Sistemelor / Calculatoare și Tehnologia Informației",
                "strengths": "Educație tehnică avansată, laboratoare moderne, pregătire inginerească complexă, conexiuni puternice cu industria"
            }
        ]
    }
}

# Definirea profilurilor standard pentru utilizatori
# Definirea profilurilor standard pentru utilizatori
user_profiles = {
    "informatica": {
        "about": "Ești o persoană analitică cu o puternică înclinație spre gândirea teoretică și abstractă. Te pasionează algoritmii, matematica și fundamentele informaticii. Îți place să rezolvi probleme complexe și să explici fenomenele din spatele soluțiilor.",
        "strengths": [
            "Gândire analitică și algoritmică",
            "Abilități avansate de programare",
            "Capacitate de abstractizare",
            "Pasiune pentru cercetare și inovație",
            "Abilitate de a rezolva probleme complexe"
        ],
        "learning_style": "Preferi un stil de învățare teoretic, cu fundamente solide și aprofundare în concepte. Apreciezi mediul academic și libertatea de a explora subiecte complexe.",
        "career_paths": [
            "Cercetător în informatică",
            "Dezvoltator de algoritmi",
            "Specialist în inteligență artificială",
            "Arhitect software",
            "Profesor universitar"
        ]
    },
    "informatica_economica": {
        "about": "Ești o persoană analitică cu o înclinație naturală spre aplicarea tehnologiei în contextul afacerilor. Îți place să rezolvi probleme reale din domeniul economic și să vezi impactul direct al soluțiilor tale. Preferi un mediu dinamic, orientat spre rezultate practice și aplicații de business.",
        "strengths": [
            "Abilități de analiză a datelor economice",
            "Înțelegerea proceselor de business",
            "Capacitate de a traduce nevoi de afaceri în soluții tehnice",
            "Abilități de comunicare și prezentare",
            "Gândire strategică orientată spre rezultate"
        ],
        "learning_style": "Preferi un stil echilibrat între teorie și practică, cu accent pe studii de caz din lumea reală și aplicații pentru mediul de business. Înveți cel mai bine prin proiecte practice care abordează probleme economice concrete.",
        "career_paths": [
            "Analist de Business",
            "Data Scientist în domeniul financiar",
            "Manager de Proiecte IT",
            "Consultant ERP",
            "Specialist în Business Intelligence"
        ]
    },
    "automatica": {
        "about": "Ești o persoană practică și tehnică, interesată de modul în care funcționează sistemele și dispozitivele. Îți place să construiești lucruri concrete și să vezi rezultatele tangibile ale muncii tale. Ești orientat spre detalii tehnice și soluții inginerești.",
        "strengths": [
            "Gândire tehnică și practică",
            "Abilități în proiectarea sistemelor",
            "Cunoștințe hardware și software",
            "Rezolvare practică a problemelor",
            "Capacitate de implementare și testare"
        ],
        "learning_style": "Preferi învățarea prin practică și experimente concrete. Te dezvolți cel mai bine în laboratoare și proiecte hands-on, unde poți testa și experimenta cu tehnologii reale.",
        "career_paths": [
            "Inginer de sistem",
            "Specialist în sisteme embedded",
            "Dezvoltator IoT",
            "Inginer în robotică",
            "Specialist în automatizări"
        ]
    }
}

# Definirea mapărilor între ID-urile de răspunsuri și domeniile
# Fiecare opțiune de răspuns este asociată cu unul sau mai multe domenii
response_domain_mapping = {
    # Întrebarea 1: Ce tip de probleme preferi să rezolvi?
    "business": ["informatica_economica"],
    "theoretical": ["informatica"],
    "hardware": ["automatica"],

    # Întrebarea 2: Care aspect al informaticii te pasionează cel mai mult?
    "software": ["informatica"],
    # "business" deja definit mai sus
    "systems": ["automatica"],

    # Întrebarea 3: Ce tip de proiecte ai vrea să realizezi în facultate?
    "research": ["informatica"],
    # "business" deja definit mai sus
    "engineering": ["automatica"],

    # Întrebarea 4: Ce materie din liceu ți-a plăcut cel mai mult?
    "math": ["informatica"],
    "economics": ["informatica_economica"],
    "physics": ["automatica"],

    # Întrebarea 5: În ce tip de companie ai prefera să lucrezi după absolvire?
    "tech": ["informatica"],
    "financial": ["informatica_economica"],
    "industrial": ["automatica"],

    # Întrebarea 6: Ce preferi să faci în timpul liber?
    "code": ["informatica"],
    # "business" deja definit mai sus
    # "hardware" deja definit mai sus

    # Întrebarea 7: Ce tip de curs te-ar interesa cel mai mult?
    "algorithms": ["informatica"],
    "erp": ["informatica_economica"],
    "embedded": ["automatica"],

    # Întrebarea 8: Care aspect al programării te interesează mai mult?
    "theory": ["informatica"],
    "applications": ["informatica_economica"],
    "low_level": ["automatica"],

    # Întrebarea 9: Ce tip de mediu de învățare preferi?
    "academic": ["informatica"],
    # "business" deja definit mai sus
    "technical": ["automatica"],

    # Întrebarea 10: Ce rol ți-ar plăcea într-un proiect de grup?
    "architect": ["informatica"],
    "manager": ["informatica_economica"],
    "implementer": ["automatica"],

    # Întrebarea 11: Ce tip de probleme matematice preferi?
    "abstract": ["informatica"],
    "applied": ["informatica_economica"],
    "engineering": ["automatica"],

    # Întrebarea 12: Care dintre următoarele discipline conexe te interesează mai mult?
    "ai": ["informatica"],
    "finance": ["informatica_economica"],
    "robotics": ["automatica"],

    # Întrebarea 13: Ce tip de internship ai prefera?
    # "research" deja definit mai sus
    # "business" deja definit mai sus
    # "engineering" deja definit mai sus

    # Întrebarea 14: Ce tip de proiect personal ai vrea să dezvolți?
    "app": ["informatica"],
    "business_analytics": ["informatica_economica"],
    "iot": ["automatica"],

    # Întrebarea 15: Ce aspect te entuziasmează cel mai mult în tehnologie?
    "innovation": ["informatica"],
    "digital_economy": ["informatica_economica"],
    "automation": ["automatica"],

    # Întrebarea 16: Ce abilități vrei să-ți dezvolți în facultate?
    "deep_cs": ["informatica"],
    "business_tech": ["informatica_economica"],
    # "engineering" deja definit mai sus

    # Întrebarea 17: Care domeniu crezi că va avea cel mai mare impact în viitor?
    # "algorithms" deja definit mai sus
    "fintech": ["informatica_economica"],
    # "robotics" deja definit mai sus

    # Întrebarea 18: În ce mod preferi să înveți concepte noi?
    "theoretical": ["informatica"],
    "case_study": ["informatica_economica"],
    "hands_on": ["automatica"],

    # Întrebarea 19: Ce tip de provocare te motivează mai mult?
    "intellectual": ["informatica"],
    # "business" deja definit mai sus
    # "technical" deja definit mai sus

    # Întrebarea 20: Ce platformă preferată utilizezi cel mai des?
    "github": ["informatica"],
    "analytics": ["informatica_economica"],
    "arduino": ["automatica"],

    # Întrebarea 21: Ce limbaj de programare preferi?
    "theory_langs": ["informatica"],
    "business_langs": ["informatica_economica"],
    "low_level_langs": ["automatica"],

    # Întrebarea 22: Ce scenariu de carieră ți se potrivește cel mai bine?
    "researcher": ["informatica"],
    "business_analyst": ["informatica_economica"],
    "system_engineer": ["automatica"],

    # Întrebarea 23: Ce tip de conferință te-ar interesa cel mai mult?
    # "academic" deja definit mai sus
    "business_tech": ["informatica_economica"],
    # "industrial" deja definit mai sus

    # Întrebarea 24: Ce tip de aplicații preferi să dezvolți?
    "algorithm_based": ["informatica"],
    "business_apps": ["informatica_economica"],
    "embedded_apps": ["automatica"],

    # Întrebarea 25: Ce proiect de final de facultate ți-ar plăcea să realizezi?
    "algorithm": ["informatica"],
    "business_platform": ["informatica_economica"],
    "hardware_solution": ["automatica"]
}


# Funcția pentru procesarea răspunsurilor și calcularea scorurilor pentru fiecare domeniu
def process_user_responses(responses):
    domain_scores = {
        "informatica": 0,
        "informatica_economica": 0,
        "automatica": 0
    }

    total_questions = 0

    # Parcurgem fiecare răspuns și actualizăm scorurile
    for question_id, response_id in responses.items():
        # Verificăm dacă există un mapping pentru acest răspuns
        if response_id in response_domain_mapping:
            # Incrementăm scorul pentru toate domeniile asociate cu acest răspuns
            for domain in response_domain_mapping[response_id]:
                domain_scores[domain] += 1
            total_questions += 1

    # Normalizăm scorurile (între 0 și 1)
    if total_questions > 0:
        for domain in domain_scores:
            domain_scores[domain] /= total_questions

    return domain_scores


# Update the user_profiles section with conditional about text based on score

def get_dynamic_about(domain, score_percent):
    """
    Returns a dynamic 'about' description based on the domain and score percentage.

    Args:
        domain (str): The domain profile key ('informatica', 'informatica_economica', or 'automatica')
        score_percent (int): The user's score percentage (0-100)

    Returns:
        str: A personalized 'about' description
    """

    base_descriptions = {
        "informatica": {
            "high": (
                "Demonstrezi o gândire analitică excepțională și o pasiune profundă pentru algoritmi și structuri de date. "
                "Abilitatea ta de a rezolva probleme complexe și de a transforma concepte abstracte în soluții inovatoare "
                "te plasează printre viitorii lideri ai industriei IT. Facultatea de Informatică îți va oferi cadrul perfect "
                "pentru a-ți rafina abilitățile și a-ți atinge potențialul maxim."
            ),
            "medium": (
                "Ai o bază solidă în gândirea logică și înțelegerea conceptelor fundamentale ale informaticii. "
                "Cu dedicare și practică, îți poți perfecționa abilitățile și poți deveni un specialist apreciat. "
                "Facultatea de Informatică îți va oferi oportunitățile necesare pentru a-ți dezvolta creativitatea "
                "și capacitatea de a construi soluții tehnologice eficiente."
            ),
            "low": (
                "Interesul tău pentru informatică este un început promițător. "
                "Chiar dacă acum ești la început de drum, curiozitatea și dorința de a învăța sunt elementele cheie "
                "care te vor ajuta să avansezi. Facultatea de Informatică îți oferă un mediu stimulativ în care să îți "
                "dezvolți abilitățile pas cu pas."
            )
        },
        "informatica_economica": {
            "high": (
                "Demonstrezi o înțelegere profundă a modului în care tehnologia poate optimiza și revoluționa mediul de afaceri. "
                "Capacitatea ta de a analiza date, identifica tendințe și implementa soluții inovatoare te poziționează "
                "excelent pentru o carieră de succes în informatică economică. Facultatea îți va oferi resursele necesare "
                "pentru a deveni un profesionist valoros în acest domeniu dinamic."
            ),
            "medium": (
                "Ai un interes clar pentru integrarea tehnologiei în mediul de afaceri și o capacitate solidă de analiză. "
                "Continuând să îți perfecționezi abilitățile, vei putea deveni un specialist apreciat în acest domeniu. "
                "Facultatea de Informatică Economică îți va oferi instrumentele necesare pentru a-ți valorifica potențialul."
            ),
            "low": (
                "Ești la început în explorarea conexiunii dintre tehnologie și economie, dar interesul tău în acest domeniu "
                "indică un potențial semnificativ. Facultatea de Informatică Economică îți va oferi îndrumarea și suportul "
                "necesar pentru a dezvolta competențele esențiale și a transforma curiozitatea în expertiză."
            )
        },
        "automatica": {
            "high": (
                "Ai o înclinație naturală către rezolvarea problemelor tehnice și o pasiune pentru sisteme inteligente și automatizare. "
                "Abilitatea ta de a înțelege și inova în domeniul ingineriei informatice te plasează pe un drum promițător. "
                "Facultatea de Automatică și Calculatoare îți va oferi instrumentele necesare pentru a dezvolta soluții tehnologice de impact."
            ),
            "medium": (
                "Demonstrezi o capacitate solidă de a înțelege și aplica principii inginerești, iar curiozitatea ta pentru tehnologie "
                "te poate propulsa spre realizări deosebite. Facultatea de Automatică și Calculatoare îți va oferi un mediu de învățare "
                "ideal pentru a-ți îmbunătăți cunoștințele și abilitățile tehnice."
            ),
            "low": (
                "Interesul tău pentru domeniul tehnic este un punct de plecare excelent. "
                "Chiar dacă acum abia începi să explorezi acest domeniu, cu determinare și învățare constantă, "
                "poți deveni un profesionist de succes. Facultatea de Automatică și Calculatoare te poate ghida în această călătorie."
            )
        }
    }

    if score_percent >= 80:
        return base_descriptions[domain]["high"]
    elif score_percent >= 60:
        return base_descriptions[domain]["medium"]
    else:
        return base_descriptions[domain]["low"]


# Modify the generate_recommendation function to include dynamic about text
def generate_recommendation(domain_scores):
    # Sortăm domeniile după scor (descrescător)
    sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)

    # Obținem domeniul cu cel mai mare scor
    best_match_domain = sorted_domains[0][0]
    best_match_score = sorted_domains[0][1]

    # Convertim scorul în procente pentru afișare
    best_match_score_percent = int(best_match_score * 100)

    # Generate dynamic about text based on score
    dynamic_about = get_dynamic_about(best_match_domain, best_match_score_percent)

    # Pregătim rezultatul cu profilul de utilizator standard pentru domeniul potrivit
    recommendation = {
        "best_match": {
            "domain": best_match_domain,
            "name": domain_profiles[best_match_domain]["name"],
            "score": best_match_score,
            "score_percent": best_match_score_percent,
            "description": domain_profiles[best_match_domain]["description"],
            "about": dynamic_about,  # Use dynamic about text instead of static
            "strengths": user_profiles[best_match_domain]["strengths"],
            "career_paths": user_profiles[best_match_domain]["career_paths"],
            "learning_style": user_profiles[best_match_domain]["learning_style"],
            "universities": domain_profiles[best_match_domain]["universities"]
        },
        "all_matches": []
    }

    # Adăugăm toate potrivirile în ordine descrescătoare a scorului
    for domain, score in sorted_domains:
        score_percent = int(score * 100)
        dynamic_domain_about = get_dynamic_about(domain, score_percent)

        match_info = {
            "domain": domain,
            "name": domain_profiles[domain]["name"],
            "score": score,
            "score_percent": score_percent,
            "description": domain_profiles[domain]["description"],
            "about": dynamic_domain_about,  # Include dynamic about for all matches
            "universities": domain_profiles[domain]["universities"]
        }
        recommendation["all_matches"].append(match_info)

    # Adăugăm scorurile detaliate pentru debugging/dezvoltare
    recommendation["detailed_scores"] = domain_scores

    return recommendation


# Funcție pentru vectorizarea răspunsurilor pentru KMeans
def create_domain_vectors():
    # Creăm vectori pentru fiecare domeniu, bazați pe cuvintele cheie asociate
    all_keywords = set()
    for profile in domain_profiles.values():
        all_keywords.update(profile["keywords"])

    keyword_list = sorted(list(all_keywords))
    keyword_to_index = {keyword: i for i, keyword in enumerate(keyword_list)}

    domain_vectors = {}
    for domain, profile in domain_profiles.items():
        vector = np.zeros(len(keyword_list))
        for keyword in profile["keywords"]:
            vector[keyword_to_index[keyword]] = 1
        domain_vectors[domain] = vector

    return domain_vectors, keyword_list


# Convertim răspunsurile în vectori pentru analiza cu KMeans
def responses_to_vector(responses, keyword_list):
    response_to_keywords = {}

    # Mapăm fiecare răspuns posibil la cuvinte cheie
    for response_id, domains in response_domain_mapping.items():
        response_to_keywords[response_id] = []
        for domain in domains:
            response_to_keywords[response_id].extend(domain_profiles[domain]["keywords"])

    # Creăm un dicționar pentru a converti cuvintele cheie la indici
    keyword_to_index = {keyword: i for i, keyword in enumerate(keyword_list)}

    # Inițializăm vectorul cu zero
    vector = np.zeros(len(keyword_list))

    # Pentru fiecare răspuns, adăugăm cuvintele cheie asociate
    for question_id, response_id in responses.items():
        if response_id in response_to_keywords:
            for keyword in response_to_keywords[response_id]:
                if keyword in keyword_to_index:
                    vector[keyword_to_index[keyword]] += 1

    # Normalizăm vectorul
    if np.sum(vector) > 0:
        vector = vector / np.sum(vector)

    return vector


def calculate_vector_similarity(user_vector, domain_vectors):
    """
    Calculează similaritatea cosinus între vectorul utilizatorului și vectorii domeniilor
    """
    similarities = {}
    for domain, vector in domain_vectors.items():
        # Calcul similitudine cosinus
        dot_product = np.dot(user_vector, vector)
        norm_user = np.linalg.norm(user_vector)
        norm_domain = np.linalg.norm(vector)

        if norm_user > 0 and norm_domain > 0:
            similarity = dot_product / (norm_user * norm_domain)
        else:
            similarity = 0

        similarities[domain] = similarity

    return similarities


def calculate_confidence_interval(domain_scores):
    """
    Calculează un interval de încredere pentru recomandare bazat pe distribuția scorurilor,
    folosind o metodologie îmbunătățită care reflectă mai bine certitudinea recomandării.
    """
    scores = list(domain_scores.values())

    # Sortăm scorurile în ordine descrescătoare
    sorted_scores = sorted(scores, reverse=True)
    top_score = sorted_scores[0]

    # Factor 1: Diferența relativă între primul și al doilea scor
    # Cu cât diferența e mai mare, cu atât suntem mai siguri de recomandare
    score_difference = 0
    if len(sorted_scores) > 1 and sorted_scores[1] > 0:
        second_score = sorted_scores[1]
        score_difference = (top_score - second_score) / top_score

    # Factor 2: Valoarea absolută a scorului principal
    # Un scor mai mare indică o potrivire mai clară cu domeniul
    absolute_score_factor = top_score

    # Factor 3: Concentrarea scorurilor - dacă primul scor este mult mai mare decât media celorlalte
    mean_other_scores = np.mean(sorted_scores[1:]) if len(sorted_scores) > 1 else 0
    concentration_factor = 0
    if mean_other_scores > 0:
        concentration_factor = (top_score - mean_other_scores) / top_score

    # Calculăm încrederea combinând cei trei factori
    # Dăm importanță mai mare valorii absolute a scorului și diferenței dintre scoruri
    confidence = (
            0.4 * absolute_score_factor +  # 40% importanță pentru scorul absolut
            0.4 * score_difference +  # 40% importanță pentru diferența între scoruri
            0.2 * concentration_factor  # 20% importanță pentru concentrarea scorurilor
    )

    # Aplicăm o transformare pentru a amplifica încrederea pentru scoruri foarte mari
    # Această funcție sigmoid modificată va da valori aproape de 1 pentru scoruri foarte mari
    if top_score > 0.8:  # Pentru scoruri peste 80%
        confidence = min(1.0, confidence + (top_score - 0.8) * 2)

    # Asigurăm că încrederea este între 0 și 1
    confidence = min(1.0, max(0.0, confidence))

    # Convertim în procent
    confidence_percent = int(confidence * 100)

    return {
        "confidence_score": confidence,
        "confidence_percent": confidence_percent,
        "interpretation": get_confidence_interpretation(confidence_percent)
    }


def calculate_integrated_confidence(domain_scores, kmeans_result, rf_result):
    """
    Calculează un scor de încredere integrat care ia în considerare atât scorurile directe,
    cât și rezultatele modelelor de ML.
    """
    # Obținem încrederea de bază din scorurile directe
    base_confidence = calculate_confidence_interval(domain_scores)
    base_confidence_score = base_confidence["confidence_score"]

    # Determinăm domeniul cu cel mai mare scor
    best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]

    # Factor 1: Consensul între algoritmi
    # Verificăm dacă KMeans și RandomForest sunt de acord cu scorul direct
    algorithm_consensus = 0.0

    # Verificăm KMeans
    kmeans_boost = 0.0
    if "cluster_domains" in kmeans_result and best_domain in kmeans_result["cluster_domains"]:
        kmeans_boost = 0.15  # Adăugăm un boost de încredere dacă KMeans confirmă rezultatul

    # Verificăm RandomForest
    rf_boost = 0.0
    if "predicted_domain" in rf_result and rf_result["predicted_domain"] == best_domain:
        # Folosim și probabilitatea predisă de RandomForest pentru a scala boost-ul
        rf_probability = rf_result.get("probabilities", {}).get(best_domain, 0)
        rf_boost = 0.15 * min(1.0, rf_probability)

    # Verificăm dacă ambele modele sunt de acord
    if kmeans_boost > 0 and rf_boost > 0:
        algorithm_consensus = 0.1  # Bonus suplimentar dacă ambele modele confirmă

    # Factor 2: Distanța față de centroid în KMeans
    # Cu cât distanța e mai mică, cu atât suntem mai încrezători
    distance_factor = 0.0
    if "distance_to_centroid" in kmeans_result:
        # Transformăm distanța într-un scor invers (distanță mai mică = scor mai mare)
        # Presupunem că distanțele normale sunt între 0 și 5 (ajustează conform datelor)
        distance = kmeans_result["distance_to_centroid"]
        distance_factor = max(0, 0.1 * (1 - min(1, distance / 5)))

    # Factor 3: Probabilitatea din RandomForest
    probability_factor = 0.0
    if "probabilities" in rf_result and best_domain in rf_result["probabilities"]:
        probability = rf_result["probabilities"][best_domain]
        probability_factor = 0.2 * probability

    # Integrăm toți factorii
    integrated_confidence = min(1.0,
                                base_confidence_score + kmeans_boost + rf_boost + algorithm_consensus + distance_factor + probability_factor)

    # Convertim în procent
    integrated_confidence_percent = int(integrated_confidence * 100)

    return {
        "confidence_score": integrated_confidence,
        "confidence_percent": integrated_confidence_percent,
        "base_confidence": base_confidence["confidence_percent"],
        "kmeans_contribution": int((kmeans_boost + distance_factor) * 100),
        "random_forest_contribution": int((rf_boost + probability_factor) * 100),
        "algorithm_consensus_boost": int(algorithm_consensus * 100),
        "interpretation": get_improved_confidence_interpretation(integrated_confidence_percent,
                                                                 base_confidence["confidence_percent"])
    }


def get_improved_confidence_interpretation(confidence_percent, base_confidence_percent):
    """
    Oferă o interpretare textuală detaliată a nivelului de încredere,
    luând în considerare atât scorul final, cât și contribuțiile diferitelor metode.
    """
    if confidence_percent >= 85:
        return f"Nivel foarte ridicat de încredere în recomandare ({confidence_percent}%). Analiza avansată prin multiple algoritmi de inteligență artificială confirmă că profilul tău se potrivește excepțional cu acest domeniu."
    elif confidence_percent >= 70:
        return f"Nivel ridicat de încredere ({confidence_percent}%). Algoritmii noștri de machine learning indică o potrivire clară între profilul tău și domeniul recomandat."
    elif confidence_percent >= 50:
        return f"Nivel bun de încredere ({confidence_percent}%). Analiza arată că acest domeniu este potrivit pentru tine, deși ai și caracteristici care te-ar face compatibil cu alte domenii conexe."
    elif confidence_percent >= 30:
        return f"Nivel moderat de încredere ({confidence_percent}%). Profilul tău conține elemente compatibile cu mai multe domenii, ceea ce sugerează versatilitate profesională."
    else:
        return f"Nivel de încredere rezervat ({confidence_percent}%). Profilul tău este foarte divers și îți recomandăm să explorezi mai multe opțiuni înainte de a lua o decizie."


def get_confidence_interpretation(confidence_percent):
    """
    Oferă o interpretare textuală a nivelului de încredere.
    """
    if confidence_percent >= 80:
        return "Nivel foarte ridicat de încredere în recomandare. Profilul tău se potrivește extrem de bine cu acest domeniu."
    elif confidence_percent >= 60:
        return "Nivel ridicat de încredere. Există o potrivire clară cu acest domeniu, deși ai și caracteristici comune cu alte domenii."
    elif confidence_percent >= 40:
        return "Nivel moderat de încredere. Deși acest domeniu este cel mai potrivit, ai un profil echilibrat cu trăsături din mai multe domenii."
    else:
        return "Nivel scăzut de încredere. Profilul tău este foarte divers și ar putea fi compatibil cu mai multe domenii. Recomandăm să explorezi și celelalte opțiuni."


def train_random_forest():
    """
    Antrenează un model Random Forest pentru clasificarea studenților
    pe baza răspunsurilor lor.
    """
    # Generăm date sintetice pentru antrenament
    # (în practică, ar trebui să ai un set de date real)
    domain_vectors, keyword_list = create_domain_vectors()

    X = []
    y = []

    # Generăm exemple pozitive pentru fiecare domeniu
    for domain, vector in domain_vectors.items():
        # Adăugăm vectorul original
        X.append(vector)
        y.append(domain)

        # Generăm variații (adăugăm "zgomot")
        for _ in range(20):  # Generăm 20 de exemple pentru fiecare domeniu
            noise = np.random.normal(0, 0.2, size=vector.shape)
            noisy_vector = vector + noise
            # Normalizăm pentru a menține valori între 0 și 1
            noisy_vector = np.clip(noisy_vector, 0, 1)
            X.append(noisy_vector)
            y.append(domain)

    X = np.array(X)

    # Împărțim datele în set de antrenament și validare
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Antrenăm modelul
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluăm modelul pe setul de validare
    accuracy = rf_model.score(X_val, y_val)
    print(f"Random Forest validation accuracy: {accuracy:.4f}")

    # Salvăm modelul
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    joblib.dump(keyword_list, 'models/rf_keyword_list.pkl')

    return rf_model, keyword_list, accuracy


def predict_with_random_forest(responses):
    """
    Utilizează modelul Random Forest pentru a prezice domeniul potrivit.
    """
    try:
        # Verificăm dacă modelul există sau îl antrenăm
        if not os.path.exists('models/random_forest_model.pkl') or not os.path.exists('models/rf_keyword_list.pkl'):
            rf_model, keyword_list, _ = train_random_forest()
        else:
            rf_model = joblib.load('models/random_forest_model.pkl')
            keyword_list = joblib.load('models/rf_keyword_list.pkl')

        # Convertim răspunsurile în vector
        user_vector = responses_to_vector(responses, keyword_list)

        # Facem predicția
        prediction = rf_model.predict([user_vector])[0]

        # Obținem probabilitățile de predicție pentru fiecare clasă
        probabilities = rf_model.predict_proba([user_vector])[0]

        # Mapăm probabilitățile la domenii
        domain_probs = {}
        for i, domain in enumerate(rf_model.classes_):
            domain_probs[domain] = float(probabilities[i])

        return {
            "predicted_domain": prediction,
            "domain_name": domain_profiles[prediction]["name"],
            "probabilities": domain_probs,
            "feature_importances": get_feature_importances(rf_model, keyword_list, responses)
        }
    except Exception as e:
        print(f"Error in Random Forest prediction: {str(e)}")
        return {"error": str(e)}


def get_feature_importances(rf_model, keyword_list, responses):
    """
    Identifică caracteristicile importante care au influențat decizia modelului.
    """
    importances = rf_model.feature_importances_

    # Obținem răspunsurile utilizatorului ca vector de caracteristici
    user_vector = responses_to_vector(responses, keyword_list)

    feature_importance = []
    for i, keyword in enumerate(keyword_list):
        if user_vector[i] > 0:  # Doar caracteristicile prezente în răspunsurile utilizatorului
            feature_importance.append({
                "keyword": keyword,
                "importance": float(importances[i]),
                "user_value": float(user_vector[i])
            })

    # Sortăm după importanță
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)

    # Luăm primele 5 caracteristici importante
    top_features = feature_importance[:5]

    return top_features


def generate_ml_insights(kmeans_result, rf_result, confidence_data, domain_scores):
    """
    Generează insights personalizate bazate pe rezultatele modelelor ML.
    """
    best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
    best_domain_name = domain_profiles[best_domain]["name"]

    # Verifică consistența între algoritmi
    rf_prediction = rf_result.get("predicted_domain")
    kmeans_domains = kmeans_result.get("cluster_domains", [])

    algorithms_agree = rf_prediction == best_domain or best_domain in kmeans_domains

    # Generăm dominant traits specific fiecărui domeniu
    domain_dominant_traits = {
        "informatica": [
            "Abordare analitică profundă a problemelor și pasiune pentru programare",
            "Gândire logică și creativă, esențială în dezvoltarea de algoritmi",
            "Aptitudine remarcabilă în utilizarea tehnologiilor de ultimă oră"
        ],
        "informatica_economica": [
            "Integrarea eficientă a tehnologiei cu viziunea economică",
            "Abilități analitice solide și orientare spre strategii de afaceri",
            "Capacitate de a transforma datele în informații de valoare"
        ],
        "automatica": [
            "Pasiune pentru robotică și soluții tehnologice inovatoare",
            "Abilități practice în inginerie și automatizare",
            "Creativitate în aplicarea tehnologiilor de control și procesare"
        ]
    }
    dominant_traits = domain_dominant_traits.get(best_domain, [])

    # Generăm insights personalizate
    insights = {
        "summary": f"Analiza noastră bazată pe inteligență artificială indică o potrivire de {domain_scores[best_domain] * 100:.1f}% cu domeniul {best_domain_name}.",
        "confidence": confidence_data["interpretation"],
        "algorithm_consensus": (
            "Ambele algoritme (clustering și clasificare) indică același domeniu ca fiind potrivit pentru tine."
            if algorithms_agree else
            "Algoritmii noștri au identificat perspective diferite asupra profilului tău, ceea ce sugerează un profil echilibrat."
        ),
        "dominant_traits": dominant_traits,
        "advice": generate_personalized_advice(best_domain, domain_scores, confidence_data["confidence_percent"])
    }

    return insights


def generate_personalized_advice(best_domain, domain_scores, confidence):
    """
    Generează sfaturi personalizate bazate pe profilul utilizatorului,
    cu mesaje adaptate pentru fiecare domeniu: Informatică, Informatică Economică
    și Automatică și Calculatoare.
    """
    advice = []

    # Sfaturi pentru domeniul principal
    if best_domain == "informatica":
        advice.append(
            "Felicitări! Profilul tău indică o pasiune naturală pentru logică și creativitate. "
            "În domeniul Informaticii vei avea ocazia să construiești soluții inovatoare, să rezolvi "
            "probleme complexe și să modelezi viitorul digital. Continuă să aprofundezi studiul algoritmilor "
            "și structurilor de date și să te implici în proiecte care te provoacă intelectual."
        )
    elif best_domain == "informatica_economica":
        advice.append(
            "Bravo! Ai o combinație unică de aptitudini tehnice și viziune economică. "
            "În Informatică Economică vei învăța să aplici tehnologia pentru a optimiza procesele de business "
            "și pentru a analiza date, contribuind astfel la inovația în mediul economic. Explorează cursuri care "
            "îmbină tehnologia cu aspectele de management și dezvoltă-ți gândirea strategică."
        )
    elif best_domain == "automatica":
        advice.append(
            "Felicitări! Dacă ești pasionat de tehnologie aplicată și de lumea roboticii, "
            "Automatică și Calculatoare este domeniul ideal pentru tine. Vei dobândi abilități practice în inginerie, "
            "vei lucra pe proiecte care aduc la viață idei inovatoare și vei contribui la transformarea mediului tehnologic. "
            "Nu ezita să te implici în laboratoare moderne și să explorezi proiecte practice care să-ți stimuleze creativitatea."
        )

    # Sfaturi suplimentare în cazul unui nivel de încredere redus
    if confidence < 50:
        advice.append(
            "Observăm că profilul tău prezintă diversitate în aptitudini, ceea ce îți oferă flexibilitate. "
            "Este o oportunitate excelentă să explorezi mai multe direcții înainte de a te specializa, astfel încât să-ți găsești "
            "cu adevărat chemarea în carieră."
        )

    # Sfaturi pentru domeniile secundare (dacă există o a doua opțiune relevantă)
    second_best = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[1]
    if second_best[1] > 0.3:  # Dacă al doilea domeniu are un scor semnificativ
        advice.append(
            f"De asemenea, ai aptitudini remarcabile și pentru {domain_profiles[second_best[0]]['name']}. "
            "Explorarea unor cursuri sau proiecte interdisciplinare care să îmbine cele două domenii te-ar putea ajuta să-ți diversifici "
            "opțiunile și să devii și mai competitiv pe piața muncii."
        )

    return advice


def enhance_domain_scores_with_ml(domain_scores, kmeans_result, rf_result):
    """
    Îmbunătățește scorurile domeniilor folosind rezultatele modelelor ML
    pentru a obține o predicție mai robustă.
    """
    enhanced_scores = domain_scores.copy()

    # Boost din KMeans
    if "cluster_domains" in kmeans_result:
        cluster_domains = kmeans_result["cluster_domains"]
        distance = kmeans_result.get("distance_to_centroid", 1.0)

        # Cu cât distanța e mai mică, cu atât boost-ul e mai mare
        kmeans_boost_factor = max(0, 0.15 * (1 - min(1, distance / 5)))

        for domain in cluster_domains:
            if domain in enhanced_scores:
                enhanced_scores[domain] += kmeans_boost_factor

    # Boost din RandomForest
    if "probabilities" in rf_result:
        probabilities = rf_result["probabilities"]

        for domain, probability in probabilities.items():
            if domain in enhanced_scores:
                # Adăugăm o parte din probabilitatea RandomForest
                rf_boost = 0.2 * probability
                enhanced_scores[domain] += rf_boost

    # Normalizăm scorurile pentru a asigura că rămân între 0 și 1
    # Doar dacă avem valori peste 1
    if any(score > 1.0 for score in enhanced_scores.values()):
        total = sum(enhanced_scores.values())
        if total > 0:
            for domain in enhanced_scores:
                enhanced_scores[domain] /= total

    return enhanced_scores


def generate_ml_enhanced_recommendation(domain_scores, kmeans_result, rf_result):
    """
    Generează o recomandare îmbunătățită bazată pe scorurile originale și rezultatele ML.
    """
    # Îmbunătățim scorurile folosind modelele ML
    enhanced_scores = enhance_domain_scores_with_ml(domain_scores, kmeans_result, rf_result)

    # Generăm recomandarea de bază folosind scorurile îmbunătățite
    recommendation = generate_recommendation(enhanced_scores)

    # Calculăm încrederea integrată
    integrated_confidence = calculate_integrated_confidence(enhanced_scores, kmeans_result, rf_result)
    recommendation["confidence"] = integrated_confidence

    # Adăugăm scorurile originale și cele îmbunătățite pentru comparație
    recommendation["original_scores"] = domain_scores
    recommendation["enhanced_scores"] = enhanced_scores

    # Adăugăm o explicație a îmbunătățirilor
    enhancement_explanation = explain_score_enhancements(domain_scores, enhanced_scores, kmeans_result, rf_result)
    recommendation["enhancement_explanation"] = enhancement_explanation

    return recommendation


def explain_score_enhancements(original_scores, enhanced_scores, kmeans_result, rf_result):
    """
    Generează o explicație clară despre cum au fost îmbunătățite scorurile 
    prin contribuția modelelor de ML.
    """
    best_domain = max(enhanced_scores.items(), key=lambda x: x[1])[0]
    best_domain_name = domain_profiles[best_domain]["name"]

    original_percent = int(original_scores[best_domain] * 100)
    enhanced_percent = int(enhanced_scores[best_domain] * 100)
    improvement = enhanced_percent - original_percent

    explanation = {
        "summary": f"Scorul pentru {best_domain_name} a crescut de la {original_percent}% la {enhanced_percent}% prin integrarea rezultatelor modelelor avansate de Machine Learning.",
        "details": []
    }

    if "cluster_domains" in kmeans_result and best_domain in kmeans_result["cluster_domains"]:
        explanation["details"].append(
            "Analiza de clustering (KMeans) a confirmat că profilul tău se încadrează în grupul specific pentru acest domeniu.")

    if "predicted_domain" in rf_result and rf_result["predicted_domain"] == best_domain:
        probability = rf_result.get("probabilities", {}).get(best_domain, 0)
        probability_percent = int(probability * 100)
        explanation["details"].append(
            f"Modelul de clasificare (Random Forest) a prezis acest domeniu cu o probabilitate de {probability_percent}%.")

    if "feature_importances" in rf_result and len(rf_result["feature_importances"]) > 0:
        top_features = [feature["keyword"] for feature in rf_result["feature_importances"][:3]]
        explanation["details"].append(
            f"Caracteristicile tale cele mai relevante pentru această predicție sunt: {', '.join(top_features)}.")

    return explanation


# Inițializăm și antrenăm modelul KMeans
def train_kmeans_model():
    """
    Antrenează un model KMeans îmbunătățit cu preprocesare mai robustă
    și un număr flexibil de clustere.
    """
    domain_vectors, keyword_list = create_domain_vectors()
    domain_data = np.array(list(domain_vectors.values()))
    domain_keys = list(domain_vectors.keys())

    # Standardizăm datele
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(domain_data)

    # Îmbunătățim diversitatea datelor pentru antrenament
    augmented_data = []
    augmented_labels = []

    # Pentru fiecare domeniu, generăm variante ale vectorului său cu mici perturbații
    for i, domain_key in enumerate(domain_keys):
        # Adăugăm vectorul original
        augmented_data.append(scaled_data[i])
        augmented_labels.append(domain_key)

        # Generăm câteva variații pentru fiecare domeniu
        for _ in range(10):  # 10 variații per domeniu
            noise = np.random.normal(0, 0.3, size=scaled_data[i].shape)
            variation = scaled_data[i] + noise
            augmented_data.append(variation)
            augmented_labels.append(domain_key)

    augmented_data = np.array(augmented_data)

    # Antrenăm modelul KMeans (folosind 3 clustere pentru cele 3 domenii)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    kmeans.fit(augmented_data)

    # Mapăm clusterele la domenii
    cluster_to_domain = {}
    for i, label in enumerate(augmented_labels):
        cluster = kmeans.labels_[i]
        if cluster not in cluster_to_domain:
            cluster_to_domain[cluster] = {}
        if label not in cluster_to_domain[cluster]:
            cluster_to_domain[cluster][label] = 0
        cluster_to_domain[cluster][label] += 1

    # Pentru fiecare cluster, determinăm domeniul predominant
    dominant_domains = {}
    for cluster, domain_counts in cluster_to_domain.items():
        dominant_domain = max(domain_counts.items(), key=lambda x: x[1])[0]
        dominant_domains[cluster] = dominant_domain

    # Salvăm modelul, scaler-ul, lista de cuvinte cheie și maparea cluster-domeniu
    os.makedirs('models', exist_ok=True)
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(keyword_list, 'models/keyword_list.pkl')
    joblib.dump(dominant_domains, 'models/cluster_domain_mapping.pkl')

    return kmeans, scaler, keyword_list, dominant_domains


def analyze_with_kmeans(responses):
    """
    Analizează răspunsurile utilizatorului folosind modelul KMeans îmbunătățit.
    """
    try:
        # Check if models directory exists
        if not os.path.exists('models'):
            os.makedirs('models', exist_ok=True)
            print("Created models directory")

        # Check if model files exist or train new ones
        if not os.path.exists('models/kmeans_model.pkl') or \
                not os.path.exists('models/scaler.pkl') or \
                not os.path.exists('models/keyword_list.pkl') or \
                not os.path.exists('models/cluster_domain_mapping.pkl'):
            print("Training new KMeans model...")
            kmeans, scaler, keyword_list, dominant_domains = train_kmeans_model()
        else:
            print("Loading existing KMeans model...")
            kmeans = joblib.load('models/kmeans_model.pkl')
            scaler = joblib.load('models/scaler.pkl')
            keyword_list = joblib.load('models/keyword_list.pkl')
            dominant_domains = joblib.load('models/cluster_domain_mapping.pkl')

        # Convert responses to vector
        user_vector = responses_to_vector(responses, keyword_list)

        # Ensure the vector is not empty
        if np.sum(user_vector) == 0:
            print("Warning: User vector is all zeros")
            return {
                "cluster": -1,
                "distance_to_centroid": 0,
                "cluster_domains": list(domain_profiles.keys()),
                "warning": "Insufficient data to determine precise cluster"
            }

        # Standardize the user vector
        scaled_user = scaler.transform(user_vector.reshape(1, -1))

        # Predict the cluster
        cluster = kmeans.predict(scaled_user)[0]

        # Calculate distance to centroid
        distance_to_centroid = np.linalg.norm(scaled_user - kmeans.cluster_centers_[cluster])

        # Calculăm distanțele la toate centroidele pentru o comparație
        all_distances = {}
        for i, centroid in enumerate(kmeans.cluster_centers_):
            distance = np.linalg.norm(scaled_user - centroid)
            all_distances[str(i)] = float(distance)

        # Determinăm domeniul asociat clusterului
        dominant_domain = dominant_domains.get(int(cluster), list(domain_profiles.keys())[0])

        # Calculăm similarități pentru fiecare domeniu
        domain_similarities = {}
        domain_vectors, _ = create_domain_vectors()
        for domain, vector in domain_vectors.items():
            scaled_domain = scaler.transform(vector.reshape(1, -1))
            similarity = 1 / (1 + np.linalg.norm(scaled_user - scaled_domain))
            domain_similarities[domain] = float(similarity)

        # Determinăm toate domeniile posibile pentru acest cluster (în caz de ambiguitate)
        possible_domains = [domain for domain in domain_profiles]
        if distance_to_centroid < 2.0:  # Dacă utilizatorul este suficient de aproape de centroid
            possible_domains = [dominant_domain]

            # Adăugăm și alte domenii apropiate
            for domain, similarity in sorted(domain_similarities.items(), key=lambda x: x[1], reverse=True):
                if domain != dominant_domain and similarity > 0.7:  # Pragul de similaritate
                    possible_domains.append(domain)

        return {
            "cluster": int(cluster),
            "distance_to_centroid": float(distance_to_centroid),
            "cluster_domain": dominant_domain,
            "cluster_domains": possible_domains,
            "all_distances": all_distances,
            "domain_similarities": domain_similarities
        }
    except Exception as e:
        import traceback
        print(f"Error in KMeans analysis: {str(e)}")
        print(traceback.format_exc())
        # Return a fallback result instead of failing
        return {
            "error": str(e),
            "fallback_cluster": 0,
            "fallback_domains": list(domain_profiles.keys())
        }


def normalize_domain_scores(domain_scores):
    """
    Normalizează scorurile domeniilor astfel încât suma lor să fie 100%.
    """
    total_score = sum(domain_scores.values())
    if total_score > 0:
        normalized_scores = {domain: (score / total_score) for domain, score in domain_scores.items()}
        return normalized_scores
    return domain_scores


# ROUTES

# Ruta pentru servirea fișierului questions.json
@app.route('/get-questions', methods=['GET'])
def get_questions():
    return jsonify(questions_data)


@app.route('/analyze-test', methods=['POST'])
def analyze_test():
    try:
        data = request.json
        user_responses = data.get('responses', {})

        # Validate the incoming data
        if not user_responses:
            return jsonify({"error": "No responses provided"}), 400

        # Procesăm răspunsurile și calculăm scorurile
        # În funcția analyze_test
        domain_scores = process_user_responses(user_responses)
        domain_scores = normalize_domain_scores(domain_scores)

        # Adăugăm analiza KMeans
        try:
            kmeans_analysis = analyze_with_kmeans(user_responses)
        except Exception as kmeans_error:
            print(f"KMeans analysis error: {str(kmeans_error)}")
            kmeans_analysis = {"error": "Could not perform clustering analysis"}

        # Adăugăm analiza Random Forest
        try:
            rf_analysis = predict_with_random_forest(user_responses)
        except Exception as rf_error:
            print(f"Random Forest analysis error: {str(rf_error)}")
            rf_analysis = {"error": "Could not perform classification analysis"}

        # Generăm recomandarea îmbunătățită cu ML
        try:
            recommendation = generate_ml_enhanced_recommendation(domain_scores, kmeans_analysis, rf_analysis)
        except Exception as enhance_error:
            print(f"Error generating enhanced recommendation: {str(enhance_error)}")
            # Fallback to basic recommendation
            recommendation = generate_recommendation(domain_scores)
            # Use improved confidence calculation
            confidence_data = calculate_confidence_interval(domain_scores)
            recommendation["confidence"] = confidence_data

        # Adăugăm rezultatele brute ale analizelor pentru debugging/dezvoltare
        recommendation["kmeans_analysis"] = kmeans_analysis
        recommendation["random_forest_analysis"] = rf_analysis

        # Calculăm similaritatea vectorilor
        try:
            domain_vectors, keyword_list = create_domain_vectors()
            user_vector = responses_to_vector(user_responses, keyword_list)
            vector_similarities = calculate_vector_similarity(user_vector, domain_vectors)
            recommendation["vector_similarity"] = vector_similarities
        except Exception as similarity_error:
            print(f"Vector similarity error: {str(similarity_error)}")
            recommendation["vector_similarity"] = {"error": "Could not calculate vector similarities"}

        # Generăm insights bazate pe ML
        try:
            ml_insights = generate_ml_insights(
                kmeans_analysis,
                rf_analysis,
                recommendation.get("confidence", {}),
                recommendation.get("enhanced_scores", domain_scores)
            )
            recommendation["ml_insights"] = ml_insights
        except Exception as insights_error:
            print(f"ML insights error: {str(insights_error)}")
            recommendation["ml_insights"] = {"error": "Could not generate ML insights"}

        return jsonify(recommendation)

    except Exception as e:
        # Log the full exception for debugging
        import traceback
        print(f"Error in analyze-test: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# Ruta pentru servirea fișierelor statice (HTML, CSS, JS)
@app.route('/', methods=['GET'])
def index():
    return send_from_directory('static', 'index.html')


@app.route('/health-check', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})


# Ruta pentru testare
@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({"status": "API funcțională!"})


if __name__ == "__main__":
    # Asigurăm existența directorului pentru modele
    os.makedirs('models', exist_ok=True)

    # Inițializăm modelul KMeans
    if not os.path.exists('models/kmeans_model.pkl'):
        train_kmeans_model()

    # Pornim serverul Flask
    app.run(debug=True, port=5000)