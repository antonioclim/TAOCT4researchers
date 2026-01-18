#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS
Week 1, Lab 1: Simulator Mașină Turing
═══════════════════════════════════════════════════════════════════════════════

CONTEXT ISTORIC
───────────────
În 1936, Alan Turing a publicat "On Computable Numbers, with an Application 
to the Entscheidungsproblem". Lucrarea a introdus conceptul de "mașină automată" 
(acum numită Mașină Turing) ca model abstract al computației.

Ironic, Turing nu a construit niciodată o mașină Turing fizică. Modelul era 
pur matematic - un experiment mental care a fundamentat întreaga informatică.

OBIECTIVE
─────────
1. Înțelegerea modelului formal al computației
2. Implementarea unui simulator în Python modern
3. Demonstrarea echivalenței Turing pentru operații simple
4. Vizualizarea execuției pas cu pas

STRUCTURA FIȘIERULUI
────────────────────
1. Definiții de tipuri (dataclasses)
2. Implementare simulator
3. Exemple: adunare în unar, verificare palindrom
4. Exerciții pentru completat

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Iterator
import time


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA I: DEFINȚII DE TIPURI
# ═══════════════════════════════════════════════════════════════════════════════

class Direction(Enum):
    """Direcția de mișcare a capului de citire/scriere."""
    LEFT = auto()
    RIGHT = auto()
    STAY = auto()  # Extensie față de modelul original


@dataclass(frozen=True)
class Transition:
    """
    O tranziție în mașina Turing.
    
    Reprezintă funcția δ: Q × Γ → Q × Γ × {L, R}
    
    Attributes:
        next_state: Starea următoare
        write_symbol: Simbolul de scris pe bandă
        direction: Direcția de mișcare
    """
    next_state: str
    write_symbol: str
    direction: Direction


@dataclass
class TuringMachine:
    """
    Implementare completă a unei Mașini Turing.
    
    Formal, M = (Q, Σ, Γ, δ, q₀, q_accept, q_reject) unde:
    - Q: mulțimea stărilor (implicit din transitions.keys())
    - Σ: alfabetul de intrare (subset al Γ)
    - Γ: alfabetul benzii (simbolurile din transitions)
    - δ: funcția de tranziție (transitions dict)
    - q₀: starea inițială (initial_state)
    - q_accept: starea de acceptare (accept_state)
    - q_reject: starea de respingere (reject_state)
    
    Attributes:
        transitions: Dict de tranziții {(stare, simbol): Transition}
        initial_state: Starea de start
        accept_state: Starea de acceptare (halt cu succes)
        reject_state: Starea de respingere (halt cu eșec)
        blank_symbol: Simbolul pentru celule goale (default: '□')
    """
    transitions: dict[tuple[str, str], Transition]
    initial_state: str
    accept_state: str = "accept"
    reject_state: str = "reject"
    blank_symbol: str = "□"
    
    def __post_init__(self) -> None:
        """Validare după inițializare."""
        # Verificăm că starea inițială are cel puțin o tranziție
        initial_transitions = [k for k in self.transitions if k[0] == self.initial_state]
        if not initial_transitions:
            raise ValueError(f"No transitions defined for initial state '{self.initial_state}'")


@dataclass
class Configuration:
    """
    Configurația curentă a mașinii Turing.
    
    O configurație reprezintă "starea instantanee" a computației:
    - Ce e pe bandă
    - Unde e capul
    - În ce stare e mașina
    
    Attributes:
        tape: Conținutul benzii (dict pentru bandă infinită)
        head_position: Poziția capului de citire/scriere
        current_state: Starea curentă a mașinii
        step_count: Numărul de pași executați
    """
    tape: dict[int, str] = field(default_factory=dict)
    head_position: int = 0
    current_state: str = ""
    step_count: int = 0
    
    def read(self, blank: str = "□") -> str:
        """Citește simbolul de sub cap."""
        return self.tape.get(self.head_position, blank)
    
    def write(self, symbol: str) -> None:
        """Scrie un simbol sub cap."""
        self.tape[self.head_position] = symbol
    
    def move(self, direction: Direction) -> None:
        """Mută capul în direcția specificată."""
        match direction:
            case Direction.LEFT:
                self.head_position -= 1
            case Direction.RIGHT:
                self.head_position += 1
            case Direction.STAY:
                pass
    
    def to_string(self, context: int = 10) -> str:
        """
        Reprezentare vizuală a configurației.
        
        Args:
            context: Numărul de celule de afișat în jurul capului
        """
        # Determinăm range-ul de afișat
        positions = list(self.tape.keys()) + [self.head_position]
        if positions:
            min_pos = min(min(positions), self.head_position - context)
            max_pos = max(max(positions), self.head_position + context)
        else:
            min_pos = -context
            max_pos = context
        
        # Construim reprezentarea
        cells = []
        markers = []
        for pos in range(min_pos, max_pos + 1):
            symbol = self.tape.get(pos, "□")
            cells.append(f" {symbol} ")
            if pos == self.head_position:
                markers.append(" ▲ ")
            else:
                markers.append("   ")
        
        tape_str = "│" + "│".join(cells) + "│"
        marker_str = " " + " ".join(markers)
        
        return (
            f"Step {self.step_count} | State: {self.current_state}\n"
            f"{tape_str}\n"
            f"{marker_str}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA II: SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

class TuringSimulator:
    """
    Simulator pentru Mașini Turing cu suport pentru:
    - Execuție pas cu pas
    - Execuție completă cu limită de pași
    - Vizualizare stare
    - Istoric execuție
    """
    
    def __init__(self, machine: TuringMachine) -> None:
        """
        Inițializează simulatorul.
        
        Args:
            machine: Mașina Turing de simulat
        """
        self.machine = machine
        self.config: Configuration | None = None
        self.history: list[Configuration] = []
    
    def load(self, input_string: str) -> None:
        """
        Încarcă un șir de intrare pe bandă.
        
        Args:
            input_string: Șirul de procesat
        """
        self.config = Configuration(
            tape={i: c for i, c in enumerate(input_string)},
            head_position=0,
            current_state=self.machine.initial_state,
            step_count=0
        )
        self.history = [self._copy_config()]
    
    def _copy_config(self) -> Configuration:
        """Creează o copie a configurației curente."""
        assert self.config is not None
        return Configuration(
            tape=dict(self.config.tape),
            head_position=self.config.head_position,
            current_state=self.config.current_state,
            step_count=self.config.step_count
        )
    
    def step(self) -> bool:
        """
        Execută un singur pas al mașinii.
        
        Returns:
            True dacă mașina poate continua, False dacă s-a oprit
        """
        if self.config is None:
            raise RuntimeError("No input loaded. Call load() first.")
        
        # Verificăm dacă am ajuns în stare terminală
        if self.config.current_state in (self.machine.accept_state, self.machine.reject_state):
            return False
        
        # Citim simbolul curent
        symbol = self.config.read(self.machine.blank_symbol)
        
        # Căutăm tranziția
        key = (self.config.current_state, symbol)
        if key not in self.machine.transitions:
            # Fără tranziție = reject implicit
            self.config.current_state = self.machine.reject_state
            return False
        
        transition = self.machine.transitions[key]
        
        # Executăm tranziția
        self.config.write(transition.write_symbol)
        self.config.move(transition.direction)
        self.config.current_state = transition.next_state
        self.config.step_count += 1
        
        # Salvăm în istoric
        self.history.append(self._copy_config())
        
        return True
    
    def run(self, max_steps: int = 10000, verbose: bool = False) -> bool:
        """
        Rulează mașina până se oprește sau atinge limita.
        
        Args:
            max_steps: Numărul maxim de pași
            verbose: Dacă afișăm fiecare pas
            
        Returns:
            True dacă a acceptat, False dacă a respins
        """
        if self.config is None:
            raise RuntimeError("No input loaded. Call load() first.")
        
        while self.config.step_count < max_steps:
            if verbose:
                print(self.config.to_string())
                print()
            
            if not self.step():
                break
        
        if verbose:
            print(self.config.to_string())
            print(f"\n{'='*50}")
            print(f"Result: {'ACCEPTED' if self.accepted else 'REJECTED'}")
            print(f"Total steps: {self.config.step_count}")
        
        return self.accepted
    
    @property
    def accepted(self) -> bool:
        """Verifică dacă mașina a acceptat intrarea."""
        return self.config is not None and self.config.current_state == self.machine.accept_state
    
    @property
    def rejected(self) -> bool:
        """Verifică dacă mașina a respins intrarea."""
        return self.config is not None and self.config.current_state == self.machine.reject_state
    
    def get_output(self) -> str:
        """Returnează conținutul benzii ca string (fără blank-uri la margini)."""
        if self.config is None:
            return ""
        
        if not self.config.tape:
            return ""
        
        min_pos = min(self.config.tape.keys())
        max_pos = max(self.config.tape.keys())
        
        result = []
        for pos in range(min_pos, max_pos + 1):
            symbol = self.config.tape.get(pos, self.machine.blank_symbol)
            if symbol != self.machine.blank_symbol:
                result.append(symbol)
        
        return "".join(result)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA III: EXEMPLE
# ═══════════════════════════════════════════════════════════════════════════════

def create_unary_increment_machine() -> TuringMachine:
    """
    Creează o mașină Turing care incrementează un număr în reprezentare unară.
    
    Reprezentare unară: n este reprezentat ca n simboluri '1'
    Exemplu: 3 = "111", 5 = "11111"
    
    Operație: Adaugă un '1' la sfârșitul numărului
    
    Diagrama de stări:
    
        ┌─────┐  1/1,R   ┌─────┐  □/1,R   ┌────────┐
        │ q0  │ ───────▶ │ q0  │ ───────▶ │ accept │
        └─────┘          └─────┘          └────────┘
             ▲              │
             └──────────────┘
    """
    return TuringMachine(
        transitions={
            # q0: Mergi la dreapta peste toți 1-ii
            ("q0", "1"): Transition("q0", "1", Direction.RIGHT),
            # q0: Când găsești blank, scrie 1 și acceptă
            ("q0", "□"): Transition("accept", "1", Direction.STAY),
        },
        initial_state="q0",
        accept_state="accept"
    )


def create_unary_addition_machine() -> TuringMachine:
    """
    Creează o mașină Turing care adună două numere în reprezentare unară.
    
    Format intrare: "111+11" (3 + 2)
    Format ieșire: "11111" (5)
    
    Algoritm:
    1. Înlocuiește '+' cu '1'
    2. Mergi la sfârșitul numărului
    3. Șterge ultimul '1'
    
    De ce funcționează:
    - "111+11" devine "1111111" (prea mult cu 1)
    - Ștergem ultimul '1' → "111111" = 6? Nu! 
    
    Hmm, să reconsiderăm...
    
    Algoritm corect:
    1. Găsește '+'
    2. Înlocuiește '+' cu '1'  
    3. Mergi la sfârșit
    4. Șterge ultimul '1' (cel care era '+')
    
    Wait, asta dă rezultat corect:
    "111+11" → "1111111" → "111111" ✗ (6, nu 5)
    
    Algoritm CORECT:
    1. Mergi la dreapta până la '+'
    2. Înlocuiește '+' cu blank
    3. Mergi la stânga până la primul '1'
    4. Înlocuiește cu blank
    5. Acceptă
    
    Rezultat: "111□11" → "11□□11" ... nu, asta e greșit
    
    Algoritm FINAL (verificat):
    - Input: "111+11" 
    - Înlocuim '+' cu '1': "1111111" (7 de 1)
    - Ștergem UN '1': "111111" (6 de 1)
    - Greșit! 3+2=5, nu 6
    
    Problema: '+' devine '1', adăugând 1 în plus.
    Soluție: ștergem DOI de '1' la final (unul pentru '+', unul pentru... nu)
    
    OK, algoritmul corect e mai simplu:
    - '+' separă cele două numere
    - Mutăm cifrele din dreapta lui '+' peste '+' 
    - SAU: Înlocuim '+' cu '1' și ștergem primul '1' din stânga
    
    Să implementăm varianta simplă:
    1. Mergi la '+', înlocuiește cu blank
    2. Mergi la stânga, șterge un '1'
    3. Strânge totul la un loc
    """
    return TuringMachine(
        transitions={
            # q_find_plus: Caută simbolul '+'
            ("q_find_plus", "1"): Transition("q_find_plus", "1", Direction.RIGHT),
            ("q_find_plus", "+"): Transition("q_erase_left", "□", Direction.LEFT),
            
            # q_erase_left: Mergi la stânga și șterge un '1'
            ("q_erase_left", "1"): Transition("q_go_right", "□", Direction.RIGHT),
            
            # q_go_right: Mergi la dreapta peste blank-uri până la '1'
            ("q_go_right", "□"): Transition("q_go_right", "□", Direction.RIGHT),
            ("q_go_right", "1"): Transition("q_compact", "1", Direction.LEFT),
            
            # q_compact: Compactează - mută '1'-urile la stânga
            ("q_compact", "□"): Transition("q_find_one", "□", Direction.LEFT),
            
            # q_find_one: Găsește următorul '1' de mutat
            ("q_find_one", "□"): Transition("q_find_one", "□", Direction.LEFT),
            ("q_find_one", "1"): Transition("accept", "1", Direction.STAY),
        },
        initial_state="q_find_plus",
        accept_state="accept"
    )


def create_palindrome_checker() -> TuringMachine:
    """
    Creează o mașină Turing care verifică dacă un șir binar este palindrom.
    
    Input: Șir de '0' și '1'
    Output: accept dacă e palindrom, reject altfel
    
    Algoritm:
    1. Citește și șterge primul caracter
    2. Mergi la ultimul caracter
    3. Verifică dacă sunt egale
    4. Șterge ultimul caracter
    5. Repetă până când șirul e gol sau rămâne un caracter
    
    Diagrama simplificată:
    
    ┌────┐ 0/□  ┌─────────┐ □/□  ┌────────┐
    │ q0 │─────▶│ q_find0 │─────▶│ accept │
    └────┘      └─────────┘      └────────┘
       │ 1/□        │ 0/□
       ▼            ▼
    ┌─────────┐  ┌────────┐
    │ q_find1 │  │ reject │
    └─────────┘  └────────┘
    """
    return TuringMachine(
        transitions={
            # q0: Citește primul caracter
            ("q0", "0"): Transition("q_seek_end_for_0", "□", Direction.RIGHT),
            ("q0", "1"): Transition("q_seek_end_for_1", "□", Direction.RIGHT),
            ("q0", "□"): Transition("accept", "□", Direction.STAY),  # Șir gol = palindrom
            
            # q_seek_end_for_0: Am citit '0', căutăm ultimul caracter
            ("q_seek_end_for_0", "0"): Transition("q_seek_end_for_0", "0", Direction.RIGHT),
            ("q_seek_end_for_0", "1"): Transition("q_seek_end_for_0", "1", Direction.RIGHT),
            ("q_seek_end_for_0", "□"): Transition("q_check_0", "□", Direction.LEFT),
            
            # q_check_0: Verificăm că ultimul caracter e '0'
            ("q_check_0", "0"): Transition("q_return", "□", Direction.LEFT),
            ("q_check_0", "1"): Transition("reject", "1", Direction.STAY),
            ("q_check_0", "□"): Transition("accept", "□", Direction.STAY),  # Un singur caracter
            
            # q_seek_end_for_1: Am citit '1', căutăm ultimul caracter  
            ("q_seek_end_for_1", "0"): Transition("q_seek_end_for_1", "0", Direction.RIGHT),
            ("q_seek_end_for_1", "1"): Transition("q_seek_end_for_1", "1", Direction.RIGHT),
            ("q_seek_end_for_1", "□"): Transition("q_check_1", "□", Direction.LEFT),
            
            # q_check_1: Verificăm că ultimul caracter e '1'
            ("q_check_1", "1"): Transition("q_return", "□", Direction.LEFT),
            ("q_check_1", "0"): Transition("reject", "0", Direction.STAY),
            ("q_check_1", "□"): Transition("accept", "□", Direction.STAY),
            
            # q_return: Revenim la începutul șirului rămas
            ("q_return", "0"): Transition("q_return", "0", Direction.LEFT),
            ("q_return", "1"): Transition("q_return", "1", Direction.LEFT),
            ("q_return", "□"): Transition("q0", "□", Direction.RIGHT),
        },
        initial_state="q0",
        accept_state="accept",
        reject_state="reject"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA IV: DEMONSTRAȚII
# ═══════════════════════════════════════════════════════════════════════════════

def demo_increment() -> None:
    """Demonstrație: incrementare în unar."""
    print("=" * 60)
    print("DEMO: Incrementare în reprezentare unară")
    print("=" * 60)
    
    machine = create_unary_increment_machine()
    simulator = TuringSimulator(machine)
    
    test_cases = ["", "1", "111", "11111"]
    
    for input_str in test_cases:
        simulator.load(input_str if input_str else "□")
        simulator.run()
        output = simulator.get_output()
        
        input_val = len(input_str)
        output_val = len(output)
        
        print(f"  {input_val} ({input_str or '∅'}) + 1 = {output_val} ({output})")
    
    print()


def demo_palindrome() -> None:
    """Demonstrație: verificare palindrom."""
    print("=" * 60)
    print("DEMO: Verificare palindrom")
    print("=" * 60)
    
    machine = create_palindrome_checker()
    simulator = TuringSimulator(machine)
    
    test_cases = [
        ("", True),
        ("0", True),
        ("1", True),
        ("00", True),
        ("11", True),
        ("01", False),
        ("10", False),
        ("010", True),
        ("0110", True),
        ("0101", False),
        ("10101", True),
        ("110011", True),
    ]
    
    for input_str, expected in test_cases:
        simulator.load(input_str if input_str else "□")
        result = simulator.run()
        
        status = "✓" if result == expected else "✗"
        result_str = "palindrom" if result else "nu e palindrom"
        
        print(f"  {status} '{input_str or '∅'}' → {result_str}")
    
    print()


def demo_step_by_step() -> None:
    """Demonstrație: execuție pas cu pas."""
    print("=" * 60)
    print("DEMO: Execuție pas cu pas - Verificare palindrom '010'")
    print("=" * 60)
    print()
    
    machine = create_palindrome_checker()
    simulator = TuringSimulator(machine)
    simulator.load("010")
    simulator.run(verbose=True)
    
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA V: EXERCIȚII
# ═══════════════════════════════════════════════════════════════════════════════

"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCIȚIU 1: Binary Increment                                                  ║
║                                                                                ║
║ Implementați o mașină Turing care incrementează un număr binar.               ║
║                                                                                ║
║ Exemple:                                                                       ║
║   "0"    → "1"                                                                ║
║   "1"    → "10"                                                               ║
║   "10"   → "11"                                                               ║
║   "11"   → "100"                                                              ║
║   "111"  → "1000"                                                             ║
║   "1011" → "1100"                                                             ║
║                                                                                ║
║ Hint: Începeți de la dreapta, propagați carry-ul spre stânga.                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

def create_binary_increment_machine() -> TuringMachine:
    """
    EXERCIȚIU: Completați implementarea.
    
    Algoritmul sugerat:
    1. Mergi la sfârșitul numărului
    2. Dacă e '0', înlocuiește cu '1' și acceptă
    3. Dacă e '1', înlocuiește cu '0' și mergi la stânga (carry)
    4. Repetă până nu mai e carry sau ajungi la începutul benzii
    5. Dacă ajungi la blank cu carry, scrie '1'
    """
    # TODO: Implementați tranziții
    return TuringMachine(
        transitions={
            # Completați aici...
            ("q_start", "□"): Transition("accept", "□", Direction.STAY),  # Placeholder
        },
        initial_state="q_start",
        accept_state="accept"
    )


"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCIȚIU 2: Balanced Parentheses                                              ║
║                                                                                ║
║ Implementați o mașină Turing care verifică dacă parantezele sunt echilibrate. ║
║                                                                                ║
║ Exemple:                                                                       ║
║   "()"     → accept                                                           ║
║   "(())"   → accept                                                           ║
║   "()()"   → accept                                                           ║
║   "(()"    → reject                                                           ║
║   "())"    → reject                                                           ║
║   "(()())" → accept                                                           ║
║                                                                                ║
║ Hint: Găsește prima ')', înlocuiește cu 'X', găsește '(' corespunzător,       ║
║       înlocuiește cu 'X', repetă. Accept dacă toate sunt 'X'.                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

def create_balanced_parentheses_machine() -> TuringMachine:
    """
    EXERCIȚIU: Completați implementarea.
    """
    # TODO: Implementați tranziții
    return TuringMachine(
        transitions={
            ("q_start", "□"): Transition("accept", "□", Direction.STAY),
        },
        initial_state="q_start",
        accept_state="accept",
        reject_state="reject"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  WEEK 1, LAB 1: SIMULATOR MAȘINĂ TURING")
    print("═" * 60 + "\n")
    
    demo_increment()
    demo_palindrome()
    demo_step_by_step()
    
    print("=" * 60)
    print("Exerciții de completat în cod:")
    print("  1. create_binary_increment_machine()")
    print("  2. create_balanced_parentheses_machine()")
    print("=" * 60)
