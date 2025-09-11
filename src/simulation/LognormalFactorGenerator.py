import random
from typing import List, Optional

import random
from typing import List, Optional

class LognormalFactorGenerator:
    """
    Generator für lognormal-verteilte Faktoren mit Erwartungswert 1.
    Diese Faktoren können später mit beliebigen Basisdauern multipliziert werden.
    """

    def __init__(self, sigma: float = 0.2, seed: Optional[int] = None):
        """
        :param sigma: Standardabweichung im Log-Raum (>= 0).
        :param seed: Optionaler Seed für Reproduzierbarkeit.
        """
        self.sigma = abs(sigma)
        self.seed = seed
        self.rng = random.Random(seed)  # eigener RNG, bleibt über die gesamte Instanz gleich

    @property
    def mu(self) -> float:
        """mu so gewählt, dass E[F] = 1."""
        return -0.5 * self.sigma ** 2

    def sample(self) -> float:
        """Einen einzelnen Faktor ziehen."""
        return self.rng.lognormvariate(self.mu, self.sigma)

    def sample_many(self, n: int) -> List[float]:
        """n Faktoren ziehen."""
        if n < 0:
            raise ValueError("n muss >= 0 sein.")
        return [self.sample() for _ in range(n)]





if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Initialisieren (sigma = 0.01, Seed für Reproduzierbarkeit)
    fgen = LognormalFactorGenerator(sigma=0.3, seed=42)

    factors = fgen.sample_many(400)

    # Dichtekurve zeichnen
    sns.kdeplot(factors, fill=True)
    plt.title("Dichte der Lognormal Factors")
    plt.xlabel("Factor")
    plt.ylabel("Dichte")
    plt.show()




