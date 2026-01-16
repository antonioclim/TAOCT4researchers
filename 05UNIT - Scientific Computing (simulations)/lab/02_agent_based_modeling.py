#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS
Week 5, Lab Part 2: Agent-Based Modeling
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Agent-Based Models (ABM) simulează comportamentul individual al agenților
și observă pattern-urile emergente la nivel macro.

Avantaje față de ecuații diferențiale:
- Heterogenitate: fiecare agent poate fi diferit
- Spațialitate: interacțiuni locale, nu globale
- Stochasticitate: comportament probabilistic
- Emergență: comportament complex din reguli simple

OBIECTIVE
─────────
1. Înțelegerea paradigmei ABM
2. Implementarea modelului Schelling de segregare
3. Crearea unui model Boids (comportament de stol)
4. Analiza rezultatelor și sensitivity analysis

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import Callable, Iterator
from collections.abc import Sequence
from enum import Enum, auto


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA I: FRAMEWORK ABM SIMPLU
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    """Poziție 2D pe grid sau în spațiu continuu."""
    x: float
    y: float
    
    def distance_to(self, other: 'Position') -> float:
        """Distanța euclidiană."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __add__(self, other: 'Position') -> 'Position':
        return Position(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Position') -> 'Position':
        return Position(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Position':
        return Position(self.x * scalar, self.y * scalar)
    
    def magnitude(self) -> float:
        """Lungimea vectorului."""
        return math.sqrt(self.x**2 + self.y**2)
    
    def normalized(self) -> 'Position':
        """Vector unitate în aceeași direcție."""
        mag = self.magnitude()
        if mag > 0:
            return Position(self.x / mag, self.y / mag)
        return Position(0, 0)


class Agent:
    """Clasă de bază pentru agenți."""
    
    _id_counter = 0
    
    def __init__(self, model: 'Model') -> None:
        self.unique_id = Agent._id_counter
        Agent._id_counter += 1
        self.model = model
        self.pos: Position | None = None
    
    def step(self) -> None:
        """Logica de update pentru agent. Override în subclase."""
        pass


class Grid:
    """
    Grid 2D pentru plasarea agenților.
    
    Suportă:
    - Mod torus (capetele se conectează)
    - Multiple agenți per celulă (MultiGrid) sau unul singur (SingleGrid)
    """
    
    def __init__(self, width: int, height: int, torus: bool = False) -> None:
        self.width = width
        self.height = height
        self.torus = torus
        self._grid: list[list[list[Agent]]] = [
            [[] for _ in range(height)] for _ in range(width)
        ]
    
    def _normalize_pos(self, x: int, y: int) -> tuple[int, int]:
        """Normalizează poziția pentru torus."""
        if self.torus:
            x = x % self.width
            y = y % self.height
        return x, y
    
    def place_agent(self, agent: Agent, x: int, y: int) -> None:
        """Plasează un agent pe grid."""
        x, y = self._normalize_pos(x, y)
        if 0 <= x < self.width and 0 <= y < self.height:
            self._grid[x][y].append(agent)
            agent.pos = Position(x, y)
    
    def remove_agent(self, agent: Agent) -> None:
        """Elimină un agent de pe grid."""
        if agent.pos:
            x, y = int(agent.pos.x), int(agent.pos.y)
            if agent in self._grid[x][y]:
                self._grid[x][y].remove(agent)
            agent.pos = None
    
    def move_agent(self, agent: Agent, new_x: int, new_y: int) -> None:
        """Mută un agent la o nouă poziție."""
        self.remove_agent(agent)
        self.place_agent(agent, new_x, new_y)
    
    def get_cell_contents(self, x: int, y: int) -> list[Agent]:
        """Returnează agenții din celula (x, y)."""
        x, y = self._normalize_pos(x, y)
        if 0 <= x < self.width and 0 <= y < self.height:
            return self._grid[x][y].copy()
        return []
    
    def get_neighbors(
        self, 
        pos: Position, 
        moore: bool = True, 
        include_center: bool = False,
        radius: int = 1
    ) -> list[Agent]:
        """
        Returnează vecinii unei poziții.
        
        Args:
            pos: Poziția centrală
            moore: True pentru vecinătate Moore (8 direcții), 
                   False pentru von Neumann (4 direcții)
            include_center: Include agenții din celula centrală
            radius: Raza vecinătății
        """
        neighbors = []
        x, y = int(pos.x), int(pos.y)
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                # Skip centru dacă nu e cerut
                if dx == 0 and dy == 0 and not include_center:
                    continue
                
                # Skip diagonale pentru von Neumann
                if not moore and abs(dx) + abs(dy) > radius:
                    continue
                
                nx, ny = self._normalize_pos(x + dx, y + dy)
                
                # Verifică limite pentru grid non-torus
                if not self.torus and (nx != x + dx or ny != y + dy):
                    continue
                
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.extend(self._grid[nx][ny])
        
        return neighbors
    
    def find_empty_cells(self) -> list[tuple[int, int]]:
        """Găsește toate celulele goale."""
        empty = []
        for x in range(self.width):
            for y in range(self.height):
                if not self._grid[x][y]:
                    empty.append((x, y))
        return empty
    
    def move_to_empty(self, agent: Agent) -> bool:
        """Mută agentul într-o celulă goală aleatorie."""
        empty = self.find_empty_cells()
        if empty:
            new_x, new_y = random.choice(empty)
            self.move_agent(agent, new_x, new_y)
            return True
        return False


class Model:
    """Clasă de bază pentru modele ABM."""
    
    def __init__(self, seed: int | None = None) -> None:
        self.running = True
        self.step_count = 0
        self.agents: list[Agent] = []
        self.random = random.Random(seed)
        random.seed(seed)  # Global seed pentru compatibilitate
    
    def add_agent(self, agent: Agent) -> None:
        """Adaugă un agent la model."""
        self.agents.append(agent)
    
    def remove_agent(self, agent: Agent) -> None:
        """Elimină un agent din model."""
        if agent in self.agents:
            self.agents.remove(agent)
    
    def step(self) -> None:
        """Un pas de simulare. Override în subclase."""
        # Activare aleatorie a agenților
        agents_shuffled = self.agents.copy()
        self.random.shuffle(agents_shuffled)
        
        for agent in agents_shuffled:
            agent.step()
        
        self.step_count += 1
    
    def run(self, steps: int) -> None:
        """Rulează modelul pentru un număr de pași."""
        for _ in range(steps):
            if not self.running:
                break
            self.step()


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA II: MODELUL SCHELLING DE SEGREGARE
# ═══════════════════════════════════════════════════════════════════════════════

class AgentType(Enum):
    """Tipuri de agenți în modelul Schelling."""
    TYPE_A = auto()
    TYPE_B = auto()


class SchellingAgent(Agent):
    """
    Agent în modelul Schelling de segregare.
    
    Regula simplă: Dacă mai puțin de X% din vecini sunt de același tip,
    agentul este "nefericit" și se mută într-o celulă goală aleatorie.
    
    Parametru cheie: homophily_threshold — cât de "tolerant" e agentul
    """
    
    def __init__(self, model: 'SchellingModel', agent_type: AgentType) -> None:
        super().__init__(model)
        self.agent_type = agent_type
        self.is_happy = True
    
    def step(self) -> None:
        """Verifică satisfacția și mută-te dacă ești nefericit."""
        model: SchellingModel = self.model  # type: ignore
        
        if self.pos is None:
            return
        
        neighbors = model.grid.get_neighbors(self.pos, moore=True)
        
        if not neighbors:
            self.is_happy = True
            return
        
        # Numără vecini de același tip
        same_type = sum(1 for n in neighbors 
                        if isinstance(n, SchellingAgent) and n.agent_type == self.agent_type)
        
        ratio = same_type / len(neighbors)
        
        self.is_happy = ratio >= model.homophily_threshold
        
        if not self.is_happy:
            model.grid.move_to_empty(self)


class SchellingModel(Model):
    """
    Modelul Schelling de segregare (1971).
    
    Demonstrează că segregarea poate apărea chiar și când indivizii
    au preferințe moderate pentru a locui lângă persoane similare.
    
    Parametri:
        width, height: dimensiunile gridului
        density: fracțiunea celulelor ocupate (0-1)
        minority_fraction: fracțiunea de tip B (0-1)
        homophily_threshold: pragul de satisfacție (0-1)
    """
    
    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        density: float = 0.8,
        minority_fraction: float = 0.5,
        homophily_threshold: float = 0.3,
        seed: int | None = None
    ) -> None:
        super().__init__(seed)
        
        self.width = width
        self.height = height
        self.density = density
        self.minority_fraction = minority_fraction
        self.homophily_threshold = homophily_threshold
        
        self.grid = Grid(width, height, torus=True)
        
        # Plasare agenți
        for x in range(width):
            for y in range(height):
                if self.random.random() < density:
                    agent_type = (AgentType.TYPE_B 
                                  if self.random.random() < minority_fraction 
                                  else AgentType.TYPE_A)
                    agent = SchellingAgent(self, agent_type)
                    self.add_agent(agent)
                    self.grid.place_agent(agent, x, y)
    
    def get_happiness_ratio(self) -> float:
        """Fracțiunea de agenți fericiți."""
        if not self.agents:
            return 1.0
        happy = sum(1 for a in self.agents if isinstance(a, SchellingAgent) and a.is_happy)
        return happy / len(self.agents)
    
    def get_segregation_index(self) -> float:
        """
        Index de segregare bazat pe vecinătăți.
        
        0 = mixare perfectă
        1 = segregare completă
        """
        if not self.agents:
            return 0.0
        
        total_same = 0
        total_neighbors = 0
        
        for agent in self.agents:
            if not isinstance(agent, SchellingAgent) or agent.pos is None:
                continue
            
            neighbors = self.grid.get_neighbors(agent.pos, moore=True)
            schelling_neighbors = [n for n in neighbors if isinstance(n, SchellingAgent)]
            
            if schelling_neighbors:
                same = sum(1 for n in schelling_neighbors if n.agent_type == agent.agent_type)
                total_same += same
                total_neighbors += len(schelling_neighbors)
        
        if total_neighbors == 0:
            return 0.0
        
        # Normalizare: 0.5 = random, 1 = complet segregat
        observed_ratio = total_same / total_neighbors
        expected_ratio = (1 - self.minority_fraction)**2 + self.minority_fraction**2
        
        # Index între 0 și 1
        return max(0, (observed_ratio - 0.5) / 0.5)
    
    def step(self) -> None:
        """Un pas de simulare."""
        super().step()
        
        # Oprire când toți sunt fericiți
        if self.get_happiness_ratio() == 1.0:
            self.running = False
    
    def get_grid_state(self) -> list[list[int]]:
        """Returnează starea gridului ca matrice: 0=gol, 1=tip A, 2=tip B."""
        state = [[0] * self.height for _ in range(self.width)]
        
        for agent in self.agents:
            if isinstance(agent, SchellingAgent) and agent.pos:
                x, y = int(agent.pos.x), int(agent.pos.y)
                state[x][y] = 1 if agent.agent_type == AgentType.TYPE_A else 2
        
        return state
    
    def print_grid(self) -> None:
        """Afișează gridul în terminal."""
        state = self.get_grid_state()
        symbols = {0: '·', 1: '█', 2: '░'}
        
        for y in range(self.height):
            row = ''.join(symbols[state[x][y]] for x in range(self.width))
            print(row)
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA III: MODELUL BOIDS (COMPORTAMENT DE STOL)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Velocity:
    """Viteză 2D cu limite."""
    vx: float
    vy: float
    
    def magnitude(self) -> float:
        return math.sqrt(self.vx**2 + self.vy**2)
    
    def limit(self, max_speed: float) -> 'Velocity':
        """Limitează viteza la o valoare maximă."""
        mag = self.magnitude()
        if mag > max_speed:
            return Velocity(self.vx * max_speed / mag, self.vy * max_speed / mag)
        return Velocity(self.vx, self.vy)
    
    def __add__(self, other: 'Velocity') -> 'Velocity':
        return Velocity(self.vx + other.vx, self.vy + other.vy)
    
    def __mul__(self, scalar: float) -> 'Velocity':
        return Velocity(self.vx * scalar, self.vy * scalar)


class Boid(Agent):
    """
    Un "boid" (bird-oid) în simularea lui Craig Reynolds (1986).
    
    Trei reguli simple:
    1. SEPARATION: Evită coliziunea cu vecinii apropiați
    2. ALIGNMENT: Aliniază direcția cu vecinii medii
    3. COHESION: Mișcă-te spre centrul grupului local
    
    Din aceste reguli simple emerge comportament complex de stol!
    """
    
    def __init__(
        self, 
        model: 'BoidsModel',
        x: float, 
        y: float,
        vx: float = 0,
        vy: float = 0
    ) -> None:
        super().__init__(model)
        self.pos = Position(x, y)
        self.velocity = Velocity(vx, vy)
    
    def step(self) -> None:
        """Aplică cele trei reguli și actualizează poziția."""
        model: BoidsModel = self.model  # type: ignore
        
        # Găsește vecinii în raza de percepție
        neighbors = [b for b in model.agents 
                     if isinstance(b, Boid) and b != self and
                     self.pos.distance_to(b.pos) < model.perception_radius]
        
        if not neighbors:
            # Fără vecini, continuă drept
            self._update_position(model)
            return
        
        # Calculează cele trei forțe
        separation = self._separation(neighbors, model)
        alignment = self._alignment(neighbors, model)
        cohesion = self._cohesion(neighbors, model)
        
        # Combină forțele cu ponderi
        self.velocity = (
            self.velocity + 
            separation * model.separation_weight +
            alignment * model.alignment_weight +
            cohesion * model.cohesion_weight
        )
        
        # Limitează viteza
        self.velocity = self.velocity.limit(model.max_speed)
        
        self._update_position(model)
    
    def _separation(self, neighbors: list['Boid'], model: 'BoidsModel') -> Velocity:
        """Evită vecinii prea apropiați."""
        steer_x, steer_y = 0.0, 0.0
        
        for other in neighbors:
            dist = self.pos.distance_to(other.pos)
            if dist < model.separation_distance and dist > 0:
                # Vector care îndepărtează
                diff_x = self.pos.x - other.pos.x
                diff_y = self.pos.y - other.pos.y
                # Ponderează invers cu distanța
                steer_x += diff_x / dist
                steer_y += diff_y / dist
        
        return Velocity(steer_x, steer_y)
    
    def _alignment(self, neighbors: list['Boid'], model: 'BoidsModel') -> Velocity:
        """Aliniază direcția cu vecinii."""
        avg_vx = sum(b.velocity.vx for b in neighbors) / len(neighbors)
        avg_vy = sum(b.velocity.vy for b in neighbors) / len(neighbors)
        
        return Velocity(avg_vx - self.velocity.vx, avg_vy - self.velocity.vy)
    
    def _cohesion(self, neighbors: list['Boid'], model: 'BoidsModel') -> Velocity:
        """Mișcă-te spre centrul grupului."""
        center_x = sum(b.pos.x for b in neighbors) / len(neighbors)
        center_y = sum(b.pos.y for b in neighbors) / len(neighbors)
        
        return Velocity(center_x - self.pos.x, center_y - self.pos.y)
    
    def _update_position(self, model: 'BoidsModel') -> None:
        """Actualizează poziția cu wrap-around."""
        new_x = self.pos.x + self.velocity.vx
        new_y = self.pos.y + self.velocity.vy
        
        # Wrap-around (torus)
        new_x = new_x % model.width
        new_y = new_y % model.height
        
        self.pos = Position(new_x, new_y)


class BoidsModel(Model):
    """
    Model Boids pentru simularea comportamentului de stol.
    
    Parametri:
        width, height: dimensiunile spațiului
        n_boids: numărul de agenți
        perception_radius: cât de departe "văd" boids
        separation_distance: distanța minimă dorită
        max_speed: viteza maximă
        separation_weight, alignment_weight, cohesion_weight: ponderile regulilor
    """
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        n_boids: int = 50,
        perception_radius: float = 10.0,
        separation_distance: float = 3.0,
        max_speed: float = 2.0,
        separation_weight: float = 1.5,
        alignment_weight: float = 1.0,
        cohesion_weight: float = 1.0,
        seed: int | None = None
    ) -> None:
        super().__init__(seed)
        
        self.width = width
        self.height = height
        self.perception_radius = perception_radius
        self.separation_distance = separation_distance
        self.max_speed = max_speed
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
        
        # Creează boids cu poziții și viteze aleatoare
        for _ in range(n_boids):
            x = self.random.uniform(0, width)
            y = self.random.uniform(0, height)
            vx = self.random.uniform(-max_speed, max_speed)
            vy = self.random.uniform(-max_speed, max_speed)
            
            boid = Boid(self, x, y, vx, vy)
            self.add_agent(boid)
    
    def get_positions(self) -> list[tuple[float, float]]:
        """Returnează pozițiile tuturor boids."""
        return [(b.pos.x, b.pos.y) for b in self.agents if isinstance(b, Boid)]
    
    def get_velocities(self) -> list[tuple[float, float]]:
        """Returnează vitezele tuturor boids."""
        return [(b.velocity.vx, b.velocity.vy) for b in self.agents if isinstance(b, Boid)]
    
    def get_average_velocity(self) -> float:
        """Viteza medie a stolului."""
        velocities = self.get_velocities()
        if not velocities:
            return 0.0
        return sum(math.sqrt(vx**2 + vy**2) for vx, vy in velocities) / len(velocities)
    
    def get_polarization(self) -> float:
        """
        Polarizare: cât de aliniate sunt direcțiile (0-1).
        
        1 = toate în aceeași direcție
        0 = direcții complet aleatorii
        """
        velocities = self.get_velocities()
        if not velocities:
            return 0.0
        
        # Media vectorilor normalizați
        sum_x, sum_y = 0.0, 0.0
        for vx, vy in velocities:
            mag = math.sqrt(vx**2 + vy**2)
            if mag > 0:
                sum_x += vx / mag
                sum_y += vy / mag
        
        return math.sqrt(sum_x**2 + sum_y**2) / len(velocities)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA IV: DEMONSTRAȚII
# ═══════════════════════════════════════════════════════════════════════════════

def demo_schelling() -> None:
    """Demonstrație: modelul Schelling de segregare."""
    print("=" * 60)
    print("DEMO: Schelling Segregation Model")
    print("=" * 60)
    print()
    
    model = SchellingModel(
        width=15,
        height=15,
        density=0.75,
        minority_fraction=0.4,
        homophily_threshold=0.3,  # Moderat: 30% vecini similari
        seed=42
    )
    
    print(f"Grid: {model.width}×{model.height}")
    print(f"Densitate: {model.density}")
    print(f"Prag satisfacție: {model.homophily_threshold} (30% vecini similari)")
    print()
    
    print("STARE INIȚIALĂ:")
    model.print_grid()
    print(f"Fericire: {model.get_happiness_ratio():.1%}")
    print(f"Segregare: {model.get_segregation_index():.2f}")
    print()
    
    # Rulare
    max_steps = 100
    for step in range(max_steps):
        if not model.running:
            break
        model.step()
    
    print(f"DUPĂ {model.step_count} PAȘI:")
    model.print_grid()
    print(f"Fericire: {model.get_happiness_ratio():.1%}")
    print(f"Segregare: {model.get_segregation_index():.2f}")
    print()
    print("Observați: segregare semnificativă chiar cu preferințe moderate!")
    print()


def demo_boids() -> None:
    """Demonstrație: modelul Boids."""
    print("=" * 60)
    print("DEMO: Boids Flocking Model")
    print("=" * 60)
    print()
    
    model = BoidsModel(
        width=50,
        height=50,
        n_boids=30,
        perception_radius=8,
        seed=42
    )
    
    print(f"Spațiu: {model.width}×{model.height}")
    print(f"Număr boids: {len(model.agents)}")
    print(f"Rază percepție: {model.perception_radius}")
    print()
    
    print(f"{'Step':>6} {'Avg Speed':>12} {'Polarization':>14}")
    print("-" * 35)
    
    for step in range(0, 101, 20):
        if step > 0:
            model.run(20)
        
        avg_speed = model.get_average_velocity()
        polarization = model.get_polarization()
        print(f"{model.step_count:>6} {avg_speed:>12.2f} {polarization:>14.2f}")
    
    print()
    print("Observați: polarizarea crește pe măsură ce stolul se formează!")
    print()


def demo_sensitivity_analysis() -> None:
    """Demonstrație: sensitivity analysis pentru Schelling."""
    print("=" * 60)
    print("DEMO: Sensitivity Analysis - Schelling Model")
    print("=" * 60)
    print()
    
    print(f"{'Threshold':>10} {'Steps':>8} {'Happiness':>12} {'Segregation':>12}")
    print("-" * 45)
    
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        model = SchellingModel(
            width=20,
            height=20,
            density=0.8,
            homophily_threshold=threshold,
            seed=42
        )
        
        model.run(200)
        
        print(f"{threshold:>10.1f} {model.step_count:>8} "
              f"{model.get_happiness_ratio():>12.1%} "
              f"{model.get_segregation_index():>12.2f}")
    
    print()
    print("Observați: chiar un prag mic (30%) produce segregare semnificativă!")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCIȚII
# ═══════════════════════════════════════════════════════════════════════════════

"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCIȚIU 1: Vizualizare Schelling                                            ║
║                                                                               ║
║ Folosind matplotlib, creați o animație a evoluției modelului Schelling.       ║
║ Hint: matplotlib.animation.FuncAnimation                                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCIȚIU 2: Predator în Boids                                                ║
║                                                                               ║
║ Adăugați un agent "predator" care urmărește stolul. Boids ar trebui să       ║
║ fugă de predator (o a patra regulă: avoidance).                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCIȚIU 3: Model Propriu                                                    ║
║                                                                               ║
║ Implementați un ABM relevant pentru domeniul vostru de cercetare.             ║
║ Documentați regulile, parametrii și comportamentul emergent observat.         ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  WEEK 5 LAB: AGENT-BASED MODELING")
    print("═" * 60 + "\n")
    
    demo_schelling()
    demo_boids()
    demo_sensitivity_analysis()
    
    print("=" * 60)
    print("Exerciții de completat:")
    print("  1. Vizualizare animată Schelling cu matplotlib")
    print("  2. Adăugare predator în modelul Boids")
    print("  3. Implementare model propriu pentru cercetare")
    print("=" * 60)
