#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
02UNIT Practice: Hard Exercise 2 — Type-Safe State Machine
═══════════════════════════════════════════════════════════════════════════════

Difficulty: ⭐⭐⭐ (Hard)
Estimated Time: 60 minutes

TASK
────
Implement a type-safe finite state machine (FSM) that:
1. Enforces valid transitions at compile time (using typing)
2. Tracks state history
3. Supports guards (conditional transitions)
4. Supports entry/exit actions

LEARNING OBJECTIVES
───────────────────
- Model state machines with types
- Use generics for type-safe state representation
- Implement the State pattern
- Understand phantom types for compile-time validation

BACKGROUND
──────────
Finite State Machines are fundamental to many domains:
- Protocol handlers (TCP states)
- UI workflows (order processing)
- Game AI (enemy behaviour)
- Document workflows (draft → review → published)

Making transitions type-safe catches invalid state sequences early.

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Generic,
    TypeVar,
    Protocol,
    Any,
)
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# STATE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class OrderState(Enum):
    """States for an order processing system."""
    DRAFT = auto()
    SUBMITTED = auto()
    PROCESSING = auto()
    SHIPPED = auto()
    DELIVERED = auto()
    CANCELLED = auto()


class DocumentState(Enum):
    """States for a document workflow."""
    DRAFT = auto()
    REVIEW = auto()
    APPROVED = auto()
    PUBLISHED = auto()
    ARCHIVED = auto()


StateT = TypeVar('StateT', bound=Enum)
ContextT = TypeVar('ContextT')


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Transition(Generic[StateT, ContextT]):
    """Represents a state transition.
    
    Attributes:
        from_state: The source state.
        to_state: The destination state.
        guard: Optional condition that must be true for transition.
        action: Optional action to execute during transition.
    """
    from_state: StateT
    to_state: StateT
    guard: Callable[[ContextT], bool] | None = None
    action: Callable[[ContextT], None] | None = None
    
    def can_execute(self, context: ContextT) -> bool:
        """Check if this transition can execute given the context."""
        if self.guard is None:
            return True
        return self.guard(context)
    
    def execute(self, context: ContextT) -> None:
        """Execute the transition action if present."""
        if self.action is not None:
            self.action(context)


@dataclass
class StateConfig(Generic[StateT, ContextT]):
    """Configuration for a single state.
    
    Attributes:
        state: The state being configured.
        on_enter: Action to execute when entering this state.
        on_exit: Action to execute when exiting this state.
    """
    state: StateT
    on_enter: Callable[[ContextT], None] | None = None
    on_exit: Callable[[ContextT], None] | None = None


@dataclass
class TransitionRecord(Generic[StateT]):
    """Record of a state transition for history tracking.
    
    Attributes:
        from_state: The previous state.
        to_state: The new state.
        timestamp: When the transition occurred.
    """
    from_state: StateT
    to_state: StateT
    timestamp: datetime = field(default_factory=datetime.now)


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


class GuardFailedError(Exception):
    """Raised when a transition guard returns False."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# YOUR TASK: Implement the StateMachine class
# ═══════════════════════════════════════════════════════════════════════════════

class StateMachine(Generic[StateT, ContextT]):
    """A type-safe finite state machine.
    
    Example:
        # Define context
        @dataclass
        class OrderContext:
            total: float
            is_paid: bool = False
        
        # Create machine
        machine = StateMachine(OrderState.DRAFT, OrderContext(total=99.99))
        
        # Add transitions
        machine.add_transition(
            Transition(OrderState.DRAFT, OrderState.SUBMITTED)
        )
        machine.add_transition(
            Transition(
                OrderState.SUBMITTED,
                OrderState.PROCESSING,
                guard=lambda ctx: ctx.is_paid
            )
        )
        
        # Execute
        machine.transition_to(OrderState.SUBMITTED)
    """
    
    def __init__(self, initial_state: StateT, context: ContextT) -> None:
        """Initialise the state machine.
        
        Args:
            initial_state: The starting state.
            context: The context object passed to guards and actions.
        """
        self._current_state: StateT = initial_state
        self._context: ContextT = context
        self._transitions: list[Transition[StateT, ContextT]] = []
        self._state_configs: dict[StateT, StateConfig[StateT, ContextT]] = {}
        self._history: list[TransitionRecord[StateT]] = []
    
    @property
    def current_state(self) -> StateT:
        """Get the current state."""
        return self._current_state
    
    @property
    def context(self) -> ContextT:
        """Get the context object."""
        return self._context
    
    @property
    def history(self) -> list[TransitionRecord[StateT]]:
        """Get the transition history."""
        return self._history.copy()
    
    def add_transition(self, transition: Transition[StateT, ContextT]) -> None:
        """Register a valid transition.
        
        Args:
            transition: The transition to register.
        """
        # TODO: Store the transition
        pass
    
    def configure_state(self, config: StateConfig[StateT, ContextT]) -> None:
        """Configure entry/exit actions for a state.
        
        Args:
            config: The state configuration.
        """
        # TODO: Store the state configuration
        pass
    
    def can_transition_to(self, target_state: StateT) -> bool:
        """Check if a transition to the target state is valid.
        
        Args:
            target_state: The state to check.
        
        Returns:
            True if the transition is valid and guard passes.
        """
        # TODO: Check if there's a valid transition that can execute
        pass
    
    def get_available_transitions(self) -> list[StateT]:
        """Get all states that can be transitioned to from current state.
        
        Returns:
            List of valid target states.
        """
        # TODO: Return all states reachable from current state
        pass
    
    def transition_to(self, target_state: StateT) -> None:
        """Execute a transition to the target state.
        
        Args:
            target_state: The state to transition to.
        
        Raises:
            InvalidTransitionError: If no valid transition exists.
            GuardFailedError: If the guard condition fails.
        """
        # TODO: Implement transition:
        # 1. Find valid transition from current to target
        # 2. Check guard condition
        # 3. Execute current state's on_exit
        # 4. Execute transition action
        # 5. Update current state
        # 6. Execute new state's on_enter
        # 7. Record in history
        pass
    
    def reset(self, state: StateT) -> None:
        """Reset the machine to a specific state without validation.
        
        This is useful for testing or recovery scenarios.
        
        Args:
            state: The state to reset to.
        """
        # TODO: Reset current state and clear history
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# YOUR TASK: Implement the StateMachineBuilder for fluent configuration
# ═══════════════════════════════════════════════════════════════════════════════

class StateMachineBuilder(Generic[StateT, ContextT]):
    """Fluent builder for state machines.
    
    Example:
        machine = (
            StateMachineBuilder(OrderState.DRAFT, context)
            .permit(OrderState.DRAFT, OrderState.SUBMITTED)
            .permit(OrderState.SUBMITTED, OrderState.PROCESSING,
                    guard=lambda c: c.is_paid)
            .on_enter(OrderState.SHIPPED, lambda c: send_email(c))
            .build()
        )
    """
    
    def __init__(self, initial_state: StateT, context: ContextT) -> None:
        """Initialise the builder."""
        self._initial_state = initial_state
        self._context = context
        self._transitions: list[Transition[StateT, ContextT]] = []
        self._state_configs: dict[StateT, StateConfig[StateT, ContextT]] = {}
    
    def permit(
        self,
        from_state: StateT,
        to_state: StateT,
        guard: Callable[[ContextT], bool] | None = None,
        action: Callable[[ContextT], None] | None = None,
    ) -> "StateMachineBuilder[StateT, ContextT]":
        """Add a permitted transition.
        
        Args:
            from_state: Source state.
            to_state: Target state.
            guard: Optional guard condition.
            action: Optional transition action.
        
        Returns:
            Self for chaining.
        """
        # TODO: Add transition and return self
        pass
    
    def on_enter(
        self,
        state: StateT,
        action: Callable[[ContextT], None],
    ) -> "StateMachineBuilder[StateT, ContextT]":
        """Set entry action for a state.
        
        Args:
            state: The state to configure.
            action: Action to execute on entry.
        
        Returns:
            Self for chaining.
        """
        # TODO: Configure entry action and return self
        pass
    
    def on_exit(
        self,
        state: StateT,
        action: Callable[[ContextT], None],
    ) -> "StateMachineBuilder[StateT, ContextT]":
        """Set exit action for a state.
        
        Args:
            state: The state to configure.
            action: Action to execute on exit.
        
        Returns:
            Self for chaining.
        """
        # TODO: Configure exit action and return self
        pass
    
    def build(self) -> StateMachine[StateT, ContextT]:
        """Build the configured state machine.
        
        Returns:
            A configured StateMachine instance.
        """
        # TODO: Create and configure the state machine
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE CONTEXT CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OrderContext:
    """Context for order state machine."""
    order_id: str
    total: float
    is_paid: bool = False
    shipped_at: datetime | None = None
    events: list[str] = field(default_factory=list)


@dataclass
class DocumentContext:
    """Context for document workflow."""
    title: str
    author: str
    approver: str | None = None
    version: int = 1


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_basic_transition() -> None:
    """Test basic state transition."""
    context = OrderContext(order_id="ORD-001", total=99.99)
    machine = StateMachine(OrderState.DRAFT, context)
    
    machine.add_transition(Transition(OrderState.DRAFT, OrderState.SUBMITTED))
    
    assert machine.current_state == OrderState.DRAFT
    machine.transition_to(OrderState.SUBMITTED)
    assert machine.current_state == OrderState.SUBMITTED


def test_guard_condition() -> None:
    """Test transition with guard condition."""
    context = OrderContext(order_id="ORD-002", total=50.0, is_paid=False)
    machine = StateMachine(OrderState.SUBMITTED, context)
    
    machine.add_transition(Transition(
        OrderState.SUBMITTED,
        OrderState.PROCESSING,
        guard=lambda c: c.is_paid
    ))
    
    # Should fail - not paid
    assert not machine.can_transition_to(OrderState.PROCESSING)
    
    # Pay and try again
    context.is_paid = True
    assert machine.can_transition_to(OrderState.PROCESSING)
    machine.transition_to(OrderState.PROCESSING)
    assert machine.current_state == OrderState.PROCESSING


def test_invalid_transition() -> None:
    """Test that invalid transitions raise errors."""
    context = OrderContext(order_id="ORD-003", total=75.0)
    machine = StateMachine(OrderState.DRAFT, context)
    
    # No transitions registered
    try:
        machine.transition_to(OrderState.SHIPPED)
        assert False, "Should have raised InvalidTransitionError"
    except InvalidTransitionError:
        pass


def test_entry_exit_actions() -> None:
    """Test state entry and exit actions."""
    context = OrderContext(order_id="ORD-004", total=100.0)
    machine = StateMachine(OrderState.DRAFT, context)
    
    machine.add_transition(Transition(OrderState.DRAFT, OrderState.SUBMITTED))
    machine.configure_state(StateConfig(
        OrderState.DRAFT,
        on_exit=lambda c: c.events.append("left_draft")
    ))
    machine.configure_state(StateConfig(
        OrderState.SUBMITTED,
        on_enter=lambda c: c.events.append("entered_submitted")
    ))
    
    machine.transition_to(OrderState.SUBMITTED)
    
    assert "left_draft" in context.events
    assert "entered_submitted" in context.events


def test_history_tracking() -> None:
    """Test transition history."""
    context = OrderContext(order_id="ORD-005", total=200.0, is_paid=True)
    machine = StateMachine(OrderState.DRAFT, context)
    
    machine.add_transition(Transition(OrderState.DRAFT, OrderState.SUBMITTED))
    machine.add_transition(Transition(OrderState.SUBMITTED, OrderState.PROCESSING))
    
    machine.transition_to(OrderState.SUBMITTED)
    machine.transition_to(OrderState.PROCESSING)
    
    history = machine.history
    assert len(history) == 2
    assert history[0].from_state == OrderState.DRAFT
    assert history[0].to_state == OrderState.SUBMITTED
    assert history[1].from_state == OrderState.SUBMITTED
    assert history[1].to_state == OrderState.PROCESSING


def test_available_transitions() -> None:
    """Test getting available transitions."""
    context = OrderContext(order_id="ORD-006", total=150.0)
    machine = StateMachine(OrderState.DRAFT, context)
    
    machine.add_transition(Transition(OrderState.DRAFT, OrderState.SUBMITTED))
    machine.add_transition(Transition(OrderState.DRAFT, OrderState.CANCELLED))
    
    available = machine.get_available_transitions()
    assert OrderState.SUBMITTED in available
    assert OrderState.CANCELLED in available
    assert len(available) == 2


def test_builder_pattern() -> None:
    """Test fluent builder."""
    context = OrderContext(order_id="ORD-007", total=300.0, is_paid=True)
    
    machine = (
        StateMachineBuilder(OrderState.DRAFT, context)
        .permit(OrderState.DRAFT, OrderState.SUBMITTED)
        .permit(OrderState.SUBMITTED, OrderState.PROCESSING,
                guard=lambda c: c.is_paid)
        .permit(OrderState.PROCESSING, OrderState.SHIPPED)
        .on_enter(OrderState.SHIPPED, 
                  lambda c: setattr(c, 'shipped_at', datetime.now()))
        .build()
    )
    
    machine.transition_to(OrderState.SUBMITTED)
    machine.transition_to(OrderState.PROCESSING)
    machine.transition_to(OrderState.SHIPPED)
    
    assert machine.current_state == OrderState.SHIPPED
    assert context.shipped_at is not None


if __name__ == "__main__":
    test_basic_transition()
    test_guard_condition()
    test_invalid_transition()
    test_entry_exit_actions()
    test_history_tracking()
    test_available_transitions()
    test_builder_pattern()
    print("All tests passed! ✓")
