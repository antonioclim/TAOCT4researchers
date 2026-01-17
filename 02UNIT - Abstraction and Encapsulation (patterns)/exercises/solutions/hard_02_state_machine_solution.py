#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
02UNIT, Hard Exercise 02: State Machine Framework — SOLUTION
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Finite State Machines (FSMs) model systems exhibiting discrete states with
well-defined transitions triggered by events. This exercise implements a
generic, type-safe state machine framework supporting entry/exit actions,
transition guards and history tracking — patterns fundamental to workflow
engines, game AI and protocol implementations.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Implement the State pattern with generic typing
2. Define transition tables with guard conditions
3. Execute entry and exit actions on state changes
4. Track state history for debugging and replay

ESTIMATED TIME
──────────────
- Reading: 10 minutes
- Implementation: 25 minutes
- Total: 35 minutes

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Generic, TypeVar

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

StateT = TypeVar("StateT", bound=Enum)
EventT = TypeVar("EventT", bound=Enum)
ContextT = TypeVar("ContextT")


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class StateMachineError(Exception):
    """Base exception for state machine errors."""
    pass


class InvalidTransitionError(StateMachineError):
    """Raised when an invalid state transition is attempted."""
    
    def __init__(self, current_state: Enum, event: Enum) -> None:
        self.current_state = current_state
        self.event = event
        super().__init__(
            f"No valid transition from {current_state.name} on event {event.name}"
        )


class GuardRejectionError(StateMachineError):
    """Raised when a transition guard rejects the transition."""
    
    def __init__(self, current_state: Enum, target_state: Enum, reason: str) -> None:
        self.current_state = current_state
        self.target_state = target_state
        self.reason = reason
        super().__init__(
            f"Transition {current_state.name} → {target_state.name} "
            f"rejected by guard: {reason}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORY TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TransitionRecord(Generic[StateT, EventT]):
    """
    Immutable record of a state transition.
    
    Attributes:
        timestamp: When the transition occurred
        from_state: The state before transition
        to_state: The state after transition
        event: The event that triggered the transition
        context_snapshot: Optional snapshot of context at transition time
    """
    
    timestamp: datetime
    from_state: StateT
    to_state: StateT
    event: EventT
    context_snapshot: dict[str, Any] | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSITION DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Transition(Generic[StateT, EventT, ContextT]):
    """
    Defines a state transition with optional guard and action.
    
    Attributes:
        source: The source state
        event: The triggering event
        target: The target state
        guard: Optional predicate that must return True for transition
        action: Optional callback executed during transition
    """
    
    source: StateT
    event: EventT
    target: StateT
    guard: Callable[[ContextT], bool] | None = None
    action: Callable[[ContextT], None] | None = None
    
    def is_valid(self, context: ContextT) -> bool:
        """Check if this transition is valid given the current context."""
        if self.guard is None:
            return True
        return self.guard(context)


# ═══════════════════════════════════════════════════════════════════════════════
# STATE HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StateHandler(Generic[StateT, ContextT]):
    """
    Defines entry and exit actions for a state.
    
    Attributes:
        state: The state this handler is for
        on_entry: Callback executed when entering the state
        on_exit: Callback executed when leaving the state
    """
    
    state: StateT
    on_entry: Callable[[ContextT], None] | None = None
    on_exit: Callable[[ContextT], None] | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

class StateMachine(Generic[StateT, EventT, ContextT]):
    """
    A generic finite state machine with typed states, events and context.
    
    Features:
    - Type-safe state and event handling
    - Transition guards (predicates)
    - Entry/exit actions per state
    - Transition actions
    - Full history tracking
    
    Example:
        >>> class TrafficState(Enum):
        ...     RED = auto()
        ...     GREEN = auto()
        ...     YELLOW = auto()
        >>> 
        >>> class TrafficEvent(Enum):
        ...     TIMER = auto()
        >>> 
        >>> fsm = StateMachine(TrafficState.RED, context={})
        >>> fsm.add_transition(TrafficState.RED, TrafficEvent.TIMER, TrafficState.GREEN)
        >>> fsm.fire(TrafficEvent.TIMER)
        >>> assert fsm.current_state == TrafficState.GREEN
    """
    
    def __init__(
        self,
        initial_state: StateT,
        context: ContextT,
        *,
        track_history: bool = True,
        max_history: int = 1000,
    ) -> None:
        """
        Initialise the state machine.
        
        Args:
            initial_state: The starting state
            context: Shared context object passed to guards and actions
            track_history: Whether to record transition history
            max_history: Maximum history entries to retain
        """
        self._current_state: StateT = initial_state
        self._context: ContextT = context
        self._track_history = track_history
        self._max_history = max_history
        
        self._transitions: dict[tuple[StateT, EventT], list[Transition[StateT, EventT, ContextT]]] = {}
        self._state_handlers: dict[StateT, StateHandler[StateT, ContextT]] = {}
        self._history: list[TransitionRecord[StateT, EventT]] = []
        
        logger.debug("StateMachine initialised in state: %s", initial_state.name)
    
    @property
    def current_state(self) -> StateT:
        """Return the current state."""
        return self._current_state
    
    @property
    def context(self) -> ContextT:
        """Return the context object."""
        return self._context
    
    @property
    def history(self) -> list[TransitionRecord[StateT, EventT]]:
        """Return the transition history (read-only copy)."""
        return list(self._history)
    
    def add_transition(
        self,
        source: StateT,
        event: EventT,
        target: StateT,
        *,
        guard: Callable[[ContextT], bool] | None = None,
        action: Callable[[ContextT], None] | None = None,
    ) -> StateMachine[StateT, EventT, ContextT]:
        """
        Add a transition to the state machine.
        
        Args:
            source: The source state
            event: The triggering event
            target: The target state
            guard: Optional predicate for conditional transitions
            action: Optional callback executed during transition
        
        Returns:
            Self for method chaining
        """
        key = (source, event)
        transition = Transition(
            source=source,
            event=event,
            target=target,
            guard=guard,
            action=action,
        )
        
        if key not in self._transitions:
            self._transitions[key] = []
        self._transitions[key].append(transition)
        
        logger.debug(
            "Added transition: %s -[%s]-> %s",
            source.name,
            event.name,
            target.name,
        )
        
        return self
    
    def set_state_handler(
        self,
        state: StateT,
        *,
        on_entry: Callable[[ContextT], None] | None = None,
        on_exit: Callable[[ContextT], None] | None = None,
    ) -> StateMachine[StateT, EventT, ContextT]:
        """
        Set entry/exit handlers for a state.
        
        Args:
            state: The state to configure
            on_entry: Callback when entering the state
            on_exit: Callback when exiting the state
        
        Returns:
            Self for method chaining
        """
        self._state_handlers[state] = StateHandler(
            state=state,
            on_entry=on_entry,
            on_exit=on_exit,
        )
        
        logger.debug("Set handler for state: %s", state.name)
        return self
    
    def can_fire(self, event: EventT) -> bool:
        """
        Check if an event can be fired from the current state.
        
        Args:
            event: The event to check
        
        Returns:
            True if a valid transition exists
        """
        key = (self._current_state, event)
        transitions = self._transitions.get(key, [])
        
        return any(t.is_valid(self._context) for t in transitions)
    
    def fire(self, event: EventT) -> StateT:
        """
        Fire an event, triggering a state transition if valid.
        
        Args:
            event: The event to fire
        
        Returns:
            The new current state
        
        Raises:
            InvalidTransitionError: If no valid transition exists
        """
        key = (self._current_state, event)
        transitions = self._transitions.get(key, [])
        
        # Find first valid transition
        valid_transition: Transition[StateT, EventT, ContextT] | None = None
        for transition in transitions:
            if transition.is_valid(self._context):
                valid_transition = transition
                break
        
        if valid_transition is None:
            raise InvalidTransitionError(self._current_state, event)
        
        # Execute transition
        self._execute_transition(valid_transition, event)
        
        return self._current_state
    
    def _execute_transition(
        self,
        transition: Transition[StateT, EventT, ContextT],
        event: EventT,
    ) -> None:
        """Execute a validated transition."""
        from_state = self._current_state
        to_state = transition.target
        
        logger.info(
            "Transition: %s -[%s]-> %s",
            from_state.name,
            event.name,
            to_state.name,
        )
        
        # Execute exit action for current state
        if from_state in self._state_handlers:
            handler = self._state_handlers[from_state]
            if handler.on_exit is not None:
                logger.debug("Executing exit action for %s", from_state.name)
                handler.on_exit(self._context)
        
        # Execute transition action
        if transition.action is not None:
            logger.debug("Executing transition action")
            transition.action(self._context)
        
        # Update state
        self._current_state = to_state
        
        # Execute entry action for new state
        if to_state in self._state_handlers:
            handler = self._state_handlers[to_state]
            if handler.on_entry is not None:
                logger.debug("Executing entry action for %s", to_state.name)
                handler.on_entry(self._context)
        
        # Record history
        if self._track_history:
            self._record_transition(from_state, to_state, event)
    
    def _record_transition(
        self,
        from_state: StateT,
        to_state: StateT,
        event: EventT,
    ) -> None:
        """Record a transition in history."""
        # Create context snapshot if context supports it
        snapshot: dict[str, Any] | None = None
        if hasattr(self._context, "__dict__"):
            snapshot = dict(self._context.__dict__)
        elif isinstance(self._context, dict):
            snapshot = dict(self._context)
        
        record = TransitionRecord(
            timestamp=datetime.now(),
            from_state=from_state,
            to_state=to_state,
            event=event,
            context_snapshot=snapshot,
        )
        
        self._history.append(record)
        
        # Trim history if necessary
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
    
    def reset(self, state: StateT | None = None) -> None:
        """
        Reset the state machine.
        
        Args:
            state: Optional state to reset to (defaults to initial state)
        """
        if state is not None:
            self._current_state = state
        self._history.clear()
        logger.debug("StateMachine reset to state: %s", self._current_state.name)


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE: DOCUMENT WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════════

class DocumentState(Enum):
    """States in a document approval workflow."""
    
    DRAFT = auto()
    PENDING_REVIEW = auto()
    APPROVED = auto()
    REJECTED = auto()
    PUBLISHED = auto()
    ARCHIVED = auto()


class DocumentEvent(Enum):
    """Events that trigger document state changes."""
    
    SUBMIT = auto()
    APPROVE = auto()
    REJECT = auto()
    PUBLISH = auto()
    ARCHIVE = auto()
    REVISE = auto()


@dataclass
class DocumentContext:
    """Context for document workflow."""
    
    title: str
    author: str
    reviewer: str | None = None
    version: int = 1
    approval_count: int = 0
    required_approvals: int = 2
    rejection_reason: str | None = None


def create_document_workflow(context: DocumentContext) -> StateMachine[DocumentState, DocumentEvent, DocumentContext]:
    """Create a document approval workflow state machine."""
    
    fsm: StateMachine[DocumentState, DocumentEvent, DocumentContext] = StateMachine(
        initial_state=DocumentState.DRAFT,
        context=context,
    )
    
    # Define transitions
    fsm.add_transition(
        DocumentState.DRAFT,
        DocumentEvent.SUBMIT,
        DocumentState.PENDING_REVIEW,
        action=lambda ctx: logger.info("Document '%s' submitted for review", ctx.title),
    )
    
    # Approval with guard — needs enough approvals
    def check_approvals(ctx: DocumentContext) -> bool:
        return ctx.approval_count >= ctx.required_approvals
    
    def increment_approval(ctx: DocumentContext) -> None:
        ctx.approval_count += 1
        logger.info("Approval %d/%d received", ctx.approval_count, ctx.required_approvals)
    
    # Partial approval — stays in PENDING_REVIEW
    fsm.add_transition(
        DocumentState.PENDING_REVIEW,
        DocumentEvent.APPROVE,
        DocumentState.PENDING_REVIEW,
        guard=lambda ctx: ctx.approval_count < ctx.required_approvals - 1,
        action=increment_approval,
    )
    
    # Final approval — moves to APPROVED
    fsm.add_transition(
        DocumentState.PENDING_REVIEW,
        DocumentEvent.APPROVE,
        DocumentState.APPROVED,
        guard=lambda ctx: ctx.approval_count >= ctx.required_approvals - 1,
        action=increment_approval,
    )
    
    fsm.add_transition(
        DocumentState.PENDING_REVIEW,
        DocumentEvent.REJECT,
        DocumentState.REJECTED,
        action=lambda ctx: logger.info("Document '%s' rejected", ctx.title),
    )
    
    fsm.add_transition(
        DocumentState.REJECTED,
        DocumentEvent.REVISE,
        DocumentState.DRAFT,
        action=lambda ctx: setattr(ctx, "version", ctx.version + 1),
    )
    
    fsm.add_transition(
        DocumentState.APPROVED,
        DocumentEvent.PUBLISH,
        DocumentState.PUBLISHED,
    )
    
    fsm.add_transition(
        DocumentState.PUBLISHED,
        DocumentEvent.ARCHIVE,
        DocumentState.ARCHIVED,
    )
    
    # State handlers
    fsm.set_state_handler(
        DocumentState.DRAFT,
        on_entry=lambda ctx: logger.info("Entering DRAFT state (v%d)", ctx.version),
    )
    
    fsm.set_state_handler(
        DocumentState.PUBLISHED,
        on_entry=lambda ctx: logger.info("Document '%s' is now PUBLISHED", ctx.title),
    )
    
    return fsm


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE: TRAFFIC LIGHT
# ═══════════════════════════════════════════════════════════════════════════════

class TrafficState(Enum):
    """Traffic light states."""
    
    RED = auto()
    GREEN = auto()
    YELLOW = auto()
    FLASHING_RED = auto()  # Emergency mode


class TrafficEvent(Enum):
    """Traffic light events."""
    
    TIMER = auto()
    EMERGENCY = auto()
    RESUME = auto()


@dataclass
class TrafficContext:
    """Context for traffic light."""
    
    cycle_count: int = 0
    emergency_active: bool = False


def create_traffic_light(context: TrafficContext) -> StateMachine[TrafficState, TrafficEvent, TrafficContext]:
    """Create a traffic light state machine."""
    
    fsm: StateMachine[TrafficState, TrafficEvent, TrafficContext] = StateMachine(
        initial_state=TrafficState.RED,
        context=context,
    )
    
    # Normal cycle: RED → GREEN → YELLOW → RED
    fsm.add_transition(TrafficState.RED, TrafficEvent.TIMER, TrafficState.GREEN)
    fsm.add_transition(TrafficState.GREEN, TrafficEvent.TIMER, TrafficState.YELLOW)
    fsm.add_transition(
        TrafficState.YELLOW,
        TrafficEvent.TIMER,
        TrafficState.RED,
        action=lambda ctx: setattr(ctx, "cycle_count", ctx.cycle_count + 1),
    )
    
    # Emergency transitions from any state
    for state in [TrafficState.RED, TrafficState.GREEN, TrafficState.YELLOW]:
        fsm.add_transition(
            state,
            TrafficEvent.EMERGENCY,
            TrafficState.FLASHING_RED,
            action=lambda ctx: setattr(ctx, "emergency_active", True),
        )
    
    fsm.add_transition(
        TrafficState.FLASHING_RED,
        TrafficEvent.RESUME,
        TrafficState.RED,
        action=lambda ctx: setattr(ctx, "emergency_active", False),
    )
    
    return fsm


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def demonstrate_state_machines() -> None:
    """Demonstrate the state machine framework."""
    logger.info("=" * 70)
    logger.info("STATE MACHINE FRAMEWORK DEMONSTRATION")
    logger.info("=" * 70)
    
    # Document workflow demonstration
    logger.info("-" * 70)
    logger.info("DOCUMENT WORKFLOW")
    logger.info("-" * 70)
    
    doc_context = DocumentContext(
        title="Research Paper",
        author="Alice",
        required_approvals=2,
    )
    
    doc_fsm = create_document_workflow(doc_context)
    
    logger.info("Initial state: %s", doc_fsm.current_state.name)
    
    # Submit document
    doc_fsm.fire(DocumentEvent.SUBMIT)
    logger.info("After SUBMIT: %s", doc_fsm.current_state.name)
    
    # First approval
    doc_fsm.fire(DocumentEvent.APPROVE)
    logger.info("After first APPROVE: %s (approvals: %d)", 
                doc_fsm.current_state.name, doc_context.approval_count)
    
    # Second approval
    doc_fsm.fire(DocumentEvent.APPROVE)
    logger.info("After second APPROVE: %s (approvals: %d)",
                doc_fsm.current_state.name, doc_context.approval_count)
    
    # Publish
    doc_fsm.fire(DocumentEvent.PUBLISH)
    logger.info("After PUBLISH: %s", doc_fsm.current_state.name)
    
    # Archive
    doc_fsm.fire(DocumentEvent.ARCHIVE)
    logger.info("After ARCHIVE: %s", doc_fsm.current_state.name)
    
    # Print history
    logger.info("-" * 70)
    logger.info("Transition History:")
    for i, record in enumerate(doc_fsm.history, 1):
        logger.info(
            "  %d. %s → %s (event: %s)",
            i,
            record.from_state.name,
            record.to_state.name,
            record.event.name,
        )
    
    # Traffic light demonstration
    logger.info("-" * 70)
    logger.info("TRAFFIC LIGHT")
    logger.info("-" * 70)
    
    traffic_context = TrafficContext()
    traffic_fsm = create_traffic_light(traffic_context)
    
    logger.info("Initial state: %s", traffic_fsm.current_state.name)
    
    # Normal cycle
    for _ in range(4):
        traffic_fsm.fire(TrafficEvent.TIMER)
        logger.info("After TIMER: %s (cycles: %d)", 
                    traffic_fsm.current_state.name, traffic_context.cycle_count)
    
    # Emergency
    traffic_fsm.fire(TrafficEvent.EMERGENCY)
    logger.info("After EMERGENCY: %s", traffic_fsm.current_state.name)
    
    # Resume
    traffic_fsm.fire(TrafficEvent.RESUME)
    logger.info("After RESUME: %s", traffic_fsm.current_state.name)
    
    # Invalid transition demonstration
    logger.info("-" * 70)
    logger.info("INVALID TRANSITION HANDLING")
    logger.info("-" * 70)
    
    try:
        traffic_fsm.fire(TrafficEvent.EMERGENCY)  # Already in RED from RESUME
        traffic_fsm.fire(TrafficEvent.TIMER)  # Can't TIMER from FLASHING_RED... wait, we're in RED
        # Actually let's trigger an invalid one
        fresh_fsm = create_traffic_light(TrafficContext())
        fresh_fsm._current_state = TrafficState.FLASHING_RED  # Manually set
        fresh_fsm.fire(TrafficEvent.TIMER)  # Invalid — no TIMER from FLASHING_RED
    except InvalidTransitionError as e:
        logger.info("Caught expected error: %s", e)
    
    logger.info("=" * 70)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="State Machine Framework Solution"
    )
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.demo:
        demonstrate_state_machines()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
