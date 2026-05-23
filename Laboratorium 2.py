import numpy as np
import pandas as pd

from pymcdm.weights.subjective import RANCOM, AHP
from pymcdm.methods.comet_tools import triads_consistency



#implementacja RANCOM


def rancom_matrix_from_ranking(ranking):
    ranking = np.array(ranking)
    n = len(ranking)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if ranking[i] < ranking[j]:
                matrix[i, j] = 1.0
            elif ranking[i] == ranking[j]:
                matrix[i, j] = 0.5
            else:
                matrix[i, j] = 0.0

    return matrix


def rancom_weights_own(ranking):
    matrix = rancom_matrix_from_ranking(ranking)
    scw = matrix.sum(axis=1)
    weights = scw / scw.sum()
    return matrix, scw, weights


def show_weights(title, criteria, weights):
    df = pd.DataFrame({
        "Kryterium": criteria,
        "Waga": np.round(weights, 4),
    })

    print("\n" + title)
    print(df.to_string(index=False))
    print("Suma wag:", round(np.sum(weights), 4))



# Problem decyzyjny wybór smartfona

criteria_5 = ["Cena", "Wydajność", "Bateria", "Aparat", "Aktualizacje"]
ranking_5 = [2, 1, 3, 4, 5]

matrix_5, scw_5, weights_5_own = rancom_weights_own(ranking_5)

print("wybór smartfona")
print("Ranking kryteriów 1 -najważniejsze:")
print(dict(zip(criteria_5, ranking_5)))

print("\nMAC:")
print(pd.DataFrame(matrix_5, index=criteria_5, columns=criteria_5))

show_weights("Wagi - własna implementacja RANCOM", criteria_5, weights_5_own)
'''
Interpretacja:
Największą wagę ma wydajność, czyli jest to najważniejsze kryterium przy wyborze smartfona.
Drugie miejsce zajmuje cena, więc opłacalność nadal mocno wpływa na decyzję.
Bateria ma znaczenie średnie, a aparat i aktualizacje są mniej istotne w przyjętym rankingu.
'''


print("porównanie z pymcdm")

rancom_5_pymcdm = RANCOM(ranking=ranking_5)
weights_5_pymcdm = rancom_5_pymcdm()

compare_5 = pd.DataFrame({
    "Kryterium": criteria_5,
    "RANCOM_własny": np.round(weights_5_own, 4),
    "RANCOM_pymcdm": np.round(weights_5_pymcdm, 4),
    "Różnica": np.round(np.abs(weights_5_own - weights_5_pymcdm), 8),
})

print(compare_5.to_string(index=False))



# 3 problemy decyzyjne, AHP i RANCOM przez pymcdm


problems = [
    {
        "name": "Wybór laptopa do pracy i studiów",
        "criteria": ["Cena", "Wydajność", "Waga", "Bateria"],
        "ranking": [1, 3, 2, 4],
    },
    {
        "name": "Wybór dostawcy IT",
        "criteria": ["Koszt", "Doświadczenie", "Czas realizacji", "Wsparcie", "Bezpieczeństwo", "Elastyczność"],
        "ranking": [2, 1, 4, 3, 5, 6],
    },
    {
        "name": "Wybór samochodu",
        "criteria": ["Cena", "Spalanie", "Bezpieczeństwo", "Komfort", "Niezawodność", "Bagażnik", "Wyposażenie", "Wygląd"],
        "ranking": [3, 2, 1, 5, 4, 6, 7, 8],
    },
]


print("analiza")

summary_rows = []

for problem in problems:
    name = problem["name"]
    criteria = problem["criteria"]
    ranking = problem["ranking"]

    rancom = RANCOM(ranking=ranking)
    w_rancom = rancom()

    ahp = AHP(ranking=ranking)
    w_ahp = ahp()

    cr = ahp.get_cr()
    tc = triads_consistency(rancom.matrix)

    df = pd.DataFrame({
        "Kryterium": criteria,
        "Ranking": ranking,
        "Waga_AHP": np.round(w_ahp, 4),
        "Waga_RANCOM": np.round(w_rancom, 4),
        "Różnica_abs": np.round(np.abs(w_ahp - w_rancom), 4),
    })

    print("\nProblem:", name)
    print(df.to_string(index=False))
    print("Suma wag AHP:", round(np.sum(w_ahp), 4))
    print("Suma wag RANCOM:", round(np.sum(w_rancom), 4))
    print("CR dla AHP:", round(cr, 4))
    print("triads_consistency dla RANCOM:", round(tc, 4))

    summary_rows.append({
        "Problem": name,
        "Liczba kryteriów": len(criteria),
        "CR_AHP": round(cr, 4),
        "Triads_RANCOM": round(tc, 4),
        "Najważniejsze_AHP": criteria[np.argmax(w_ahp)],
        "Najważniejsze_RANCOM": criteria[np.argmax(w_rancom)],
    })

summary = pd.DataFrame(summary_rows)

print("\nPodsumowanie:")
print(summary.to_string(index=False))

'''
RANCOM jest prosty, bo opiera się na rankingu i relacji ważniejsze/równie ważne/mniej ważne
W przyjętych rankingach triads_consistency wynosi 1.0, więc nie występują sprzeczne trójki preferencji
 AHP daje podobny układ najważniejszych kryteriów, ale rozkład wag jest inny niż w RANCOM
Wraz ze wzrostem liczby kryteriów rośnie liczba porównań, dlatego łatwiej o niespójność ocen
Jeżeli CR albo triads_consistency wskazują słabą spójność, należy poprawić oceny eksperckie
'''

