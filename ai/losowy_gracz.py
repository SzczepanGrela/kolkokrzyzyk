from typing import Tuple, Optional
import random
from gra.logika import StanGry


def znajdz_najlepszy_ruch(stan_gry: StanGry) -> Optional[Tuple[int, int]]:
    dostepne_ruchy = stan_gry.otrzymaj_mozliwe_ruchy()

    if not dostepne_ruchy:
        return None

    return random.choice(dostepne_ruchy)



