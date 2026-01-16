#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS
Week 5, Lab: Simulări Monte Carlo și Ecuații Diferențiale
═══════════════════════════════════════════════════════════════════════════════

CONTEXT ISTORIC
───────────────
Metodele Monte Carlo au fost dezvoltate în secret la Los Alamos în anii 1940
pentru simulări ale bombei atomice. Numele vine de la cazinoul din Monaco —
o glumă a lui Stanislaw Ulam despre unchiul său care juca acolo.

Ecuațiile diferențiale sunt limbajul în care natura își scrie legile:
- Newton: F = ma ⟹ mẍ = F
- Maxwell: ∇×E = -∂B/∂t
- Schrödinger: iℏ∂ψ/∂t = Ĥψ

OBIECTIVE
─────────
1. Implementarea metodelor Monte Carlo pentru estimări statistice
2. Rezolvarea numerică a ODE-urilor cu multiple metode
3. Compararea Euler vs Runge-Kutta în termeni de precizie și stabilitate
4. Simulare agent-based simplă (Schelling segregation)

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import Callable, TypeVar, Generic
from collections.abc import Sequence
import time

# Pentru calcule vectoriale (opțional)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA I: NUMERE PSEUDO-ALEATOARE
# ═══════════════════════════════════════════════════════════════════════════════

class LCG:
    """
    Linear Congruential Generator — cel mai simplu PRNG.
    
    Formula: X_{n+1} = (a * X_n + c) mod m
    
    Parametrii de mai jos sunt din "Numerical Recipes" și oferă
    perioadă completă de 2^32.
    
    ATENȚIE: LCG-urile sunt NESIGURE pentru criptografie!
    Folosiți doar pentru simulări necritice.
    """
    
    def __init__(self, seed: int = 42) -> None:
        self.a = 1664525
        self.c = 1013904223
        self.m = 2**32
        self.state = seed
    
    def next_int(self) -> int:
        """Generează următorul întreg pseudo-aleator."""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state
    
    def random(self) -> float:
        """Generează un float în [0, 1)."""
        return self.next_int() / self.m
    
    def randint(self, a: int, b: int) -> int:
        """Generează un întreg în [a, b]."""
        return a + int(self.random() * (b - a + 1))
    
    def uniform(self, a: float, b: float) -> float:
        """Generează un float în [a, b)."""
        return a + self.random() * (b - a)


class XorShift:
    """
    XorShift — PRNG mai rapid și de mai bună calitate decât LCG.
    
    Implementarea xorshift128+ care e folosită în multe browsere
    pentru Math.random().
    """
    
    def __init__(self, seed: int = 42) -> None:
        # Inițializare din seed cu splitmix64
        self.s0 = self._splitmix64(seed)
        self.s1 = self._splitmix64(self.s0)
    
    @staticmethod
    def _splitmix64(x: int) -> int:
        """Funcție de inițializare pentru a evita seed-uri proaste."""
        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return x ^ (x >> 31)
    
    def next_int(self) -> int:
        """Generează următorul întreg pe 64 biți."""
        s0, s1 = self.s0, self.s1
        result = (s0 + s1) & 0xFFFFFFFFFFFFFFFF
        
        s1 ^= s0
        self.s0 = ((s0 << 55 | s0 >> 9) ^ s1 ^ (s1 << 14)) & 0xFFFFFFFFFFFFFFFF
        self.s1 = (s1 << 36 | s1 >> 28) & 0xFFFFFFFFFFFFFFFF
        
        return result
    
    def random(self) -> float:
        """Generează un float în [0, 1)."""
        return (self.next_int() >> 11) / (1 << 53)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA II: METODA MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════════

def monte_carlo_pi(n_samples: int, rng: random.Random | None = None) -> tuple[float, float]:
    """
    Estimează π folosind metoda Monte Carlo.
    
    Aruncăm puncte aleator într-un pătrat [-1,1]×[-1,1] și numărăm
    câte cad în cercul de rază 1 înscris în pătrat.
    
    Aria cercului / Aria pătratului = πr² / (2r)² = π/4
    
    Args:
        n_samples: Numărul de puncte de generat
        rng: Generator de numere aleatoare (opțional)
        
    Returns:
        (pi_estimate, standard_error)
    """
    if rng is None:
        rng = random.Random()
    
    inside = 0
    
    for _ in range(n_samples):
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        
        if x*x + y*y <= 1:
            inside += 1
    
    # Estimare
    p_inside = inside / n_samples
    pi_estimate = 4 * p_inside
    
    # Standard error: SE = sqrt(p(1-p)/n) * 4
    standard_error = 4 * math.sqrt(p_inside * (1 - p_inside) / n_samples)
    
    return pi_estimate, standard_error


def monte_carlo_integral(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_samples: int,
    rng: random.Random | None = None
) -> tuple[float, float]:
    """
    Estimează ∫[a,b] f(x) dx folosind Monte Carlo.
    
    Ideea: E[f(X)] = ∫f(x)p(x)dx
    Pentru distribuție uniformă pe [a,b]: p(x) = 1/(b-a)
    Deci: ∫f(x)dx = (b-a) * E[f(X)]
    
    Args:
        f: Funcția de integrat
        a, b: Limitele integrării
        n_samples: Numărul de eșantioane
        rng: Generator de numere aleatoare
        
    Returns:
        (integral_estimate, standard_error)
    """
    if rng is None:
        rng = random.Random()
    
    values = []
    
    for _ in range(n_samples):
        x = rng.uniform(a, b)
        values.append(f(x))
    
    # Estimare: (b-a) * media valorilor
    mean_f = sum(values) / n_samples
    integral_estimate = (b - a) * mean_f
    
    # Standard error
    variance = sum((v - mean_f)**2 for v in values) / (n_samples - 1)
    standard_error = (b - a) * math.sqrt(variance / n_samples)
    
    return integral_estimate, standard_error


def importance_sampling_integral(
    f: Callable[[float], float],
    proposal_pdf: Callable[[float], float],
    proposal_sampler: Callable[[random.Random], float],
    n_samples: int,
    rng: random.Random | None = None
) -> tuple[float, float]:
    """
    Estimează ∫f(x)dx folosind importance sampling.
    
    Ideea: ∫f(x)dx = ∫(f(x)/q(x)) * q(x)dx = E_q[f(X)/q(X)]
    
    Alegem q(x) astfel încât să eșantionăm mai mult din regiunile
    unde f(x) e mare, reducând varianța.
    
    Args:
        f: Funcția de integrat
        proposal_pdf: PDF-ul distribuției de proposal q(x)
        proposal_sampler: Funcție care generează eșantioane din q
        n_samples: Numărul de eșantioane
        rng: Generator de numere aleatoare
        
    Returns:
        (integral_estimate, standard_error)
    """
    if rng is None:
        rng = random.Random()
    
    weighted_values = []
    
    for _ in range(n_samples):
        x = proposal_sampler(rng)
        weight = f(x) / proposal_pdf(x)
        weighted_values.append(weight)
    
    mean = sum(weighted_values) / n_samples
    variance = sum((w - mean)**2 for w in weighted_values) / (n_samples - 1)
    standard_error = math.sqrt(variance / n_samples)
    
    return mean, standard_error


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA III: ECUAȚII DIFERENȚIALE ORDINARE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ODESolution:
    """Rezultatul rezolvării unei ODE."""
    t: list[float]
    y: list[list[float]]  # y[i] = starea la momentul t[i]
    method: str
    
    @property
    def final_state(self) -> list[float]:
        return self.y[-1]


def euler_method(
    f: Callable[[float, list[float]], list[float]],
    y0: list[float],
    t_span: tuple[float, float],
    h: float
) -> ODESolution:
    """
    Metoda Euler — cea mai simplă metodă de integrare numerică.
    
    Formula: y_{n+1} = y_n + h * f(t_n, y_n)
    
    Ordin: 1 (eroare locală O(h²), eroare globală O(h))
    
    Avantaje:
    - Simplă de implementat și înțeles
    - Rapidă
    
    Dezavantaje:
    - Eroare mare
    - Instabilă pentru probleme stiff
    
    Args:
        f: Funcția dy/dt = f(t, y)
        y0: Condiția inițială
        t_span: (t_start, t_end)
        h: Pasul de integrare
        
    Returns:
        ODESolution cu t și y
    """
    t_start, t_end = t_span
    
    t_values = [t_start]
    y_values = [y0.copy()]
    
    t = t_start
    y = y0.copy()
    
    while t < t_end:
        # Ajustăm ultimul pas pentru a termina exact la t_end
        current_h = min(h, t_end - t)
        
        # Pasul Euler
        dydt = f(t, y)
        y = [y[i] + current_h * dydt[i] for i in range(len(y))]
        t += current_h
        
        t_values.append(t)
        y_values.append(y.copy())
    
    return ODESolution(t=t_values, y=y_values, method="Euler")


def rk4_method(
    f: Callable[[float, list[float]], list[float]],
    y0: list[float],
    t_span: tuple[float, float],
    h: float
) -> ODESolution:
    """
    Metoda Runge-Kutta de ordin 4 (RK4) — "workingul horse" al integrării numerice.
    
    Formula:
        k1 = f(t_n, y_n)
        k2 = f(t_n + h/2, y_n + h*k1/2)
        k3 = f(t_n + h/2, y_n + h*k2/2)
        k4 = f(t_n + h, y_n + h*k3)
        y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    Ordin: 4 (eroare locală O(h⁵), eroare globală O(h⁴))
    
    Avantaje:
    - Precizie excelentă pentru majoritatea problemelor
    - Raport bun precizie/cost computațional
    
    Dezavantaje:
    - 4 evaluări ale lui f per pas
    - Tot instabilă pentru probleme stiff
    """
    t_start, t_end = t_span
    n = len(y0)
    
    t_values = [t_start]
    y_values = [y0.copy()]
    
    t = t_start
    y = y0.copy()
    
    while t < t_end:
        current_h = min(h, t_end - t)
        
        # Calculăm k1, k2, k3, k4
        k1 = f(t, y)
        
        y_temp = [y[i] + current_h/2 * k1[i] for i in range(n)]
        k2 = f(t + current_h/2, y_temp)
        
        y_temp = [y[i] + current_h/2 * k2[i] for i in range(n)]
        k3 = f(t + current_h/2, y_temp)
        
        y_temp = [y[i] + current_h * k3[i] for i in range(n)]
        k4 = f(t + current_h, y_temp)
        
        # Update
        y = [y[i] + current_h/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) 
             for i in range(n)]
        t += current_h
        
        t_values.append(t)
        y_values.append(y.copy())
    
    return ODESolution(t=t_values, y=y_values, method="RK4")


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA IV: MODELE CONCRETE
# ═══════════════════════════════════════════════════════════════════════════════

def sir_model(t: float, state: list[float], beta: float, gamma: float, N: float) -> list[float]:
    """
    Modelul SIR pentru epidemii.
    
    dS/dt = -β*S*I/N
    dI/dt = β*S*I/N - γ*I
    dR/dt = γ*I
    
    Parametri:
    - β: rata de transmisie
    - γ: rata de recuperare
    - N: populația totală
    """
    S, I, R = state
    
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    
    return [dS, dI, dR]


def lotka_volterra(t: float, state: list[float], alpha: float, beta: float, 
                   delta: float, gamma: float) -> list[float]:
    """
    Modelul Lotka-Volterra pentru prădător-pradă.
    
    dx/dt = αx - βxy   (pradă)
    dy/dt = δxy - γy   (prădător)
    
    Parametri:
    - α: rata de creștere a prăzii
    - β: rata de predație
    - δ: eficiența conversiei prăzii în prădători
    - γ: rata de mortalitate a prădătorilor
    """
    x, y = state  # x = pradă, y = prădător
    
    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y
    
    return [dx, dy]


def harmonic_oscillator(t: float, state: list[float], omega: float, 
                        damping: float = 0.0) -> list[float]:
    """
    Oscilator armonic (posibil amortizat).
    
    d²x/dt² + 2ζω*dx/dt + ω²x = 0
    
    Scriem ca sistem de ordin 1:
    dx/dt = v
    dv/dt = -ω²x - 2ζωv
    """
    x, v = state
    
    dx = v
    dv = -omega**2 * x - 2 * damping * omega * v
    
    return [dx, dv]


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA V: AGENT-BASED MODEL - SCHELLING SEGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SchellingModel:
    """
    Modelul de segregare al lui Schelling (1971).
    
    Nobel Prize în Economie 2005 pentru această descoperire!
    
    Regulă simplă: Fiecare agent vrea ca cel puțin X% din vecini 
    să fie de același tip. Dacă nu, se mută.
    
    Rezultat emergent: Segregare extremă chiar cu preferințe moderate!
    """
    width: int
    height: int
    density: float  # Fracția de celule ocupate
    similarity_threshold: float  # Cât de similar trebuie să fie vecinătatea
    grid: list[list[int]] = field(init=False)  # 0=gol, 1=tip A, 2=tip B
    
    def __post_init__(self) -> None:
        """Inițializează grid-ul cu agenți random."""
        self.grid = [[0] * self.width for _ in range(self.height)]
        
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < self.density:
                    self.grid[y][x] = 1 if random.random() < 0.5 else 2
    
    def get_neighbors(self, x: int, y: int) -> list[int]:
        """Returnează tipurile vecinilor (Moore neighborhood)."""
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % self.width  # Toroidal wrapping
                ny = (y + dy) % self.height
                if self.grid[ny][nx] != 0:
                    neighbors.append(self.grid[ny][nx])
        return neighbors
    
    def is_happy(self, x: int, y: int) -> bool:
        """Verifică dacă agentul de la (x,y) e mulțumit."""
        agent_type = self.grid[y][x]
        if agent_type == 0:
            return True  # Celulele goale sunt "fericite"
        
        neighbors = self.get_neighbors(x, y)
        if not neighbors:
            return True  # Fără vecini = fericit
        
        same_type = sum(1 for n in neighbors if n == agent_type)
        return same_type / len(neighbors) >= self.similarity_threshold
    
    def find_empty_cell(self) -> tuple[int, int] | None:
        """Găsește o celulă goală random."""
        empty_cells = [
            (x, y) 
            for y in range(self.height) 
            for x in range(self.width) 
            if self.grid[y][x] == 0
        ]
        return random.choice(empty_cells) if empty_cells else None
    
    def step(self) -> int:
        """
        Execută un pas de simulare.
        
        Returns:
            Numărul de agenți care s-au mutat
        """
        moved = 0
        
        # Creăm lista de agenți nefericiți
        unhappy = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.grid[y][x] != 0 and not self.is_happy(x, y)
        ]
        
        random.shuffle(unhappy)
        
        for x, y in unhappy:
            empty = self.find_empty_cell()
            if empty:
                ex, ey = empty
                self.grid[ey][ex] = self.grid[y][x]
                self.grid[y][x] = 0
                moved += 1
        
        return moved
    
    def happiness_ratio(self) -> float:
        """Calculează fracția de agenți fericiți."""
        total = 0
        happy = 0
        
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] != 0:
                    total += 1
                    if self.is_happy(x, y):
                        happy += 1
        
        return happy / total if total > 0 else 1.0
    
    def segregation_index(self) -> float:
        """
        Calculează un indice de segregare (0 = amestecat, 1 = segregat).
        
        Bazat pe fracția de vecini de același tip pentru fiecare agent.
        """
        total_same = 0
        total_neighbors = 0
        
        for y in range(self.height):
            for x in range(self.width):
                agent_type = self.grid[y][x]
                if agent_type == 0:
                    continue
                
                neighbors = self.get_neighbors(x, y)
                if neighbors:
                    same = sum(1 for n in neighbors if n == agent_type)
                    total_same += same
                    total_neighbors += len(neighbors)
        
        # 0.5 = random, 1.0 = complet segregat
        return total_same / total_neighbors if total_neighbors > 0 else 0.5
    
    def to_string(self) -> str:
        """Reprezentare ASCII a grid-ului."""
        chars = {0: '.', 1: 'A', 2: 'B'}
        return '\n'.join(
            ''.join(chars[self.grid[y][x]] for x in range(self.width))
            for y in range(self.height)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA VI: DEMONSTRAȚII
# ═══════════════════════════════════════════════════════════════════════════════

def demo_monte_carlo_pi() -> None:
    """Demonstrație: estimare π cu Monte Carlo."""
    print("=" * 60)
    print("DEMO: Monte Carlo Estimation of π")
    print("=" * 60)
    print()
    
    for n in [100, 1000, 10000, 100000, 1000000]:
        pi_est, se = monte_carlo_pi(n)
        error = abs(pi_est - math.pi)
        print(f"  n={n:>7}: π ≈ {pi_est:.6f} ± {se:.6f}  (error: {error:.6f})")
    
    print(f"\n  True π = {math.pi:.6f}")
    print()


def demo_ode_comparison() -> None:
    """Demonstrație: comparație Euler vs RK4."""
    print("=" * 60)
    print("DEMO: ODE Solver Comparison (Harmonic Oscillator)")
    print("=" * 60)
    print()
    
    # Oscilator armonic simplu: x'' + x = 0
    # Soluția exactă: x(t) = cos(t), v(t) = -sin(t) pentru x(0)=1, v(0)=0
    omega = 1.0
    y0 = [1.0, 0.0]  # x=1, v=0
    t_span = (0, 2 * math.pi)  # O perioadă completă
    
    print(f"  Problem: x'' + x = 0, x(0) = 1, x'(0) = 0")
    print(f"  Exact solution at t=2π: x = 1.0, v = 0.0")
    print()
    
    def f(t: float, y: list[float]) -> list[float]:
        return harmonic_oscillator(t, y, omega)
    
    for h in [0.1, 0.01, 0.001]:
        euler_sol = euler_method(f, y0, t_span, h)
        rk4_sol = rk4_method(f, y0, t_span, h)
        
        euler_final = euler_sol.final_state
        rk4_final = rk4_sol.final_state
        
        euler_error = math.sqrt((euler_final[0] - 1)**2 + euler_final[1]**2)
        rk4_error = math.sqrt((rk4_final[0] - 1)**2 + rk4_final[1]**2)
        
        print(f"  h = {h}:")
        print(f"    Euler: x={euler_final[0]:.6f}, v={euler_final[1]:.6f}, error={euler_error:.2e}")
        print(f"    RK4:   x={rk4_final[0]:.6f}, v={rk4_final[1]:.6f}, error={rk4_error:.2e}")
    
    print()


def demo_sir_epidemic() -> None:
    """Demonstrație: model SIR."""
    print("=" * 60)
    print("DEMO: SIR Epidemic Model")
    print("=" * 60)
    print()
    
    # Parametri pentru o epidemie cu R₀ = 2.5
    N = 10000
    beta = 0.5  # Rata de transmisie
    gamma = 0.2  # Rata de recuperare (R₀ = β/γ = 2.5)
    y0 = [N - 10, 10, 0]  # 10 infectați inițial
    t_span = (0, 100)
    
    print(f"  Population: {N}")
    print(f"  R₀ = β/γ = {beta/gamma:.2f}")
    print(f"  Initial infected: 10")
    print()
    
    def f(t: float, y: list[float]) -> list[float]:
        return sir_model(t, y, beta, gamma, N)
    
    solution = rk4_method(f, y0, t_span, 0.1)
    
    # Găsim peak-ul infecției
    max_infected = 0
    max_time = 0
    for i, y in enumerate(solution.y):
        if y[1] > max_infected:
            max_infected = y[1]
            max_time = solution.t[i]
    
    final = solution.final_state
    print(f"  Peak infection: {max_infected:.0f} at t={max_time:.1f}")
    print(f"  Final state:")
    print(f"    Susceptible: {final[0]:.0f} ({100*final[0]/N:.1f}%)")
    print(f"    Infected: {final[1]:.0f}")
    print(f"    Recovered: {final[2]:.0f} ({100*final[2]/N:.1f}%)")
    print()


def demo_schelling() -> None:
    """Demonstrație: modelul Schelling."""
    print("=" * 60)
    print("DEMO: Schelling Segregation Model")
    print("=" * 60)
    print()
    
    # Parametri
    model = SchellingModel(
        width=20,
        height=20,
        density=0.8,
        similarity_threshold=0.3  # Doar 30% preferință = segregare extremă!
    )
    
    print(f"  Grid: {model.width}x{model.height}")
    print(f"  Density: {model.density}")
    print(f"  Similarity threshold: {model.similarity_threshold}")
    print()
    
    print("  Initial state:")
    print(f"    Happiness: {model.happiness_ratio():.1%}")
    print(f"    Segregation index: {model.segregation_index():.3f}")
    print()
    
    # Rulăm simularea
    for step in range(1, 51):
        moved = model.step()
        
        if step % 10 == 0 or moved == 0:
            print(f"  Step {step}: {moved} agents moved")
            print(f"    Happiness: {model.happiness_ratio():.1%}")
            print(f"    Segregation index: {model.segregation_index():.3f}")
        
        if moved == 0:
            print(f"\n  Equilibrium reached at step {step}!")
            break
    
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  WEEK 5 LAB: MONTE CARLO & ODE SOLVERS")
    print("═" * 60 + "\n")
    
    random.seed(42)
    
    demo_monte_carlo_pi()
    demo_ode_comparison()
    demo_sir_epidemic()
    demo_schelling()
    
    print("=" * 60)
    print("Exerciții de completat:")
    print("  1. Implementați adaptive step-size pentru RK4 (RK45)")
    print("  2. Adăugați MCMC (Metropolis-Hastings) pentru sampling")
    print("  3. Extindeți modelul Schelling cu mai mult de 2 tipuri")
    print("  4. Vizualizați rezultatele cu matplotlib")
    print("=" * 60)
