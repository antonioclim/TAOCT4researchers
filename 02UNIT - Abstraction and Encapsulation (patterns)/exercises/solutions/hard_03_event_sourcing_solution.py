#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
02UNIT, Hard Exercise 03: Event Sourcing — SOLUTION
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Event Sourcing persists the state of a business entity as a sequence of
domain events. Rather than storing current state, the system stores the
complete history of state-changing events, enabling temporal queries,
audit trails and state reconstruction at any point in time.

This exercise implements a type-safe event sourcing framework with:
- Immutable domain events
- Aggregate roots with event replay
- Event store with append-only semantics
- Projections for read models (CQRS)

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Model domain events as immutable records
2. Implement aggregate roots that emit and apply events
3. Build an event store with optimistic concurrency
4. Create projections for materialised views

ESTIMATED TIME
──────────────
- Reading: 15 minutes
- Implementation: 25 minutes
- Total: 40 minutes

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Generic, TypeVar
from collections.abc import Iterator

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

TEvent = TypeVar("TEvent", bound="DomainEvent")
TAggregate = TypeVar("TAggregate", bound="AggregateRoot[Any]")
AggregateId = str


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class EventSourcingError(Exception):
    """Base exception for event sourcing errors."""
    pass


class ConcurrencyError(EventSourcingError):
    """Raised when optimistic concurrency check fails."""
    
    def __init__(self, aggregate_id: AggregateId, expected: int, actual: int) -> None:
        self.aggregate_id = aggregate_id
        self.expected_version = expected
        self.actual_version = actual
        super().__init__(
            f"Concurrency conflict for aggregate {aggregate_id}: "
            f"expected version {expected}, but found {actual}"
        )


class AggregateNotFoundError(EventSourcingError):
    """Raised when an aggregate cannot be found."""
    
    def __init__(self, aggregate_id: AggregateId) -> None:
        self.aggregate_id = aggregate_id
        super().__init__(f"Aggregate not found: {aggregate_id}")


class InvalidOperationError(EventSourcingError):
    """Raised when an invalid operation is attempted on an aggregate."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN EVENTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DomainEvent:
    """
    Base class for all domain events.
    
    Domain events are immutable records of something that happened in the
    domain. They capture the intent and outcome of a state change.
    
    Attributes:
        event_id: Unique identifier for this event instance
        aggregate_id: ID of the aggregate this event belongs to
        timestamp: When the event occurred
        version: Event version within the aggregate's stream
    """
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    aggregate_id: AggregateId = ""
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 0
    
    @property
    def event_type(self) -> str:
        """Return the event type name."""
        return self.__class__.__name__


@dataclass(frozen=True)
class EventEnvelope(Generic[TEvent]):
    """
    Wrapper for domain events with metadata.
    
    Attributes:
        event: The domain event
        stream_id: Identifier for the event stream
        position: Position in the global event log
        metadata: Additional metadata (correlation ID, user ID, etc.)
    """
    
    event: TEvent
    stream_id: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATE ROOT
# ═══════════════════════════════════════════════════════════════════════════════

class AggregateRoot(ABC, Generic[TEvent]):
    """
    Base class for aggregate roots using event sourcing.
    
    Aggregates are the consistency boundary in the domain model. All state
    changes occur through events, which are recorded and can be replayed
    to reconstruct state.
    
    Subclasses must implement _apply methods for each event type they handle.
    """
    
    def __init__(self, aggregate_id: AggregateId | None = None) -> None:
        """
        Initialise the aggregate.
        
        Args:
            aggregate_id: Unique identifier. Generated if not provided.
        """
        self._id: AggregateId = aggregate_id or str(uuid.uuid4())
        self._version: int = 0
        self._uncommitted_events: list[TEvent] = []
    
    @property
    def id(self) -> AggregateId:
        """Return the aggregate ID."""
        return self._id
    
    @property
    def version(self) -> int:
        """Return the current version (number of applied events)."""
        return self._version
    
    @property
    def uncommitted_events(self) -> list[TEvent]:
        """Return events that haven't been persisted yet."""
        return list(self._uncommitted_events)
    
    def clear_uncommitted_events(self) -> None:
        """Clear the list of uncommitted events after persistence."""
        self._uncommitted_events.clear()
    
    def _raise_event(self, event: TEvent) -> None:
        """
        Raise a new domain event.
        
        The event is applied to update state and added to uncommitted events.
        
        Args:
            event: The domain event to raise
        """
        self._version += 1
        
        # Create a new event with aggregate ID and version
        updated_event = self._with_metadata(event)
        
        self._apply(updated_event)
        self._uncommitted_events.append(updated_event)
        
        logger.debug(
            "Event raised: %s (v%d)",
            updated_event.event_type,
            self._version,
        )
    
    def _with_metadata(self, event: TEvent) -> TEvent:
        """Add aggregate metadata to an event."""
        # Create a new instance with updated fields
        # This works for dataclasses
        event_dict = {
            k: v for k, v in event.__dict__.items()
            if not k.startswith("_")
        }
        event_dict["aggregate_id"] = self._id
        event_dict["version"] = self._version
        event_dict["timestamp"] = datetime.now()
        
        return type(event)(**event_dict)
    
    def load_from_history(self, events: list[TEvent]) -> None:
        """
        Reconstruct aggregate state from a sequence of events.
        
        Args:
            events: Historical events in order
        """
        for event in events:
            self._apply(event)
            self._version = event.version
        
        logger.debug(
            "Loaded aggregate %s from %d events (v%d)",
            self._id,
            len(events),
            self._version,
        )
    
    @abstractmethod
    def _apply(self, event: TEvent) -> None:
        """
        Apply an event to update aggregate state.
        
        Subclasses must implement this to handle each event type.
        This method should be idempotent and deterministic.
        
        Args:
            event: The domain event to apply
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# EVENT STORE
# ═══════════════════════════════════════════════════════════════════════════════

class EventStore:
    """
    In-memory event store with append-only semantics.
    
    Features:
    - Append events with optimistic concurrency control
    - Load events for a specific aggregate
    - Subscribe to new events
    - Global event log for projections
    """
    
    def __init__(self) -> None:
        """Initialise an empty event store."""
        self._streams: dict[AggregateId, list[DomainEvent]] = {}
        self._global_log: list[EventEnvelope[DomainEvent]] = []
        self._subscribers: list[Callable[[EventEnvelope[DomainEvent]], None]] = []
    
    def append(
        self,
        aggregate_id: AggregateId,
        events: list[DomainEvent],
        expected_version: int,
    ) -> None:
        """
        Append events to an aggregate's stream.
        
        Args:
            aggregate_id: The aggregate ID
            events: Events to append
            expected_version: Expected current version for optimistic concurrency
        
        Raises:
            ConcurrencyError: If expected_version doesn't match
        """
        current_version = self._get_version(aggregate_id)
        
        if current_version != expected_version:
            raise ConcurrencyError(aggregate_id, expected_version, current_version)
        
        if aggregate_id not in self._streams:
            self._streams[aggregate_id] = []
        
        for event in events:
            # Add to stream
            self._streams[aggregate_id].append(event)
            
            # Add to global log
            envelope = EventEnvelope(
                event=event,
                stream_id=aggregate_id,
                position=len(self._global_log),
            )
            self._global_log.append(envelope)
            
            # Notify subscribers
            for subscriber in self._subscribers:
                subscriber(envelope)
            
            logger.debug(
                "Appended event %s to stream %s (position %d)",
                event.event_type,
                aggregate_id,
                envelope.position,
            )
    
    def load(self, aggregate_id: AggregateId) -> list[DomainEvent]:
        """
        Load all events for an aggregate.
        
        Args:
            aggregate_id: The aggregate ID
        
        Returns:
            List of events in order
        """
        return list(self._streams.get(aggregate_id, []))
    
    def _get_version(self, aggregate_id: AggregateId) -> int:
        """Get the current version of an aggregate's stream."""
        events = self._streams.get(aggregate_id, [])
        return events[-1].version if events else 0
    
    def subscribe(
        self,
        handler: Callable[[EventEnvelope[DomainEvent]], None],
    ) -> None:
        """
        Subscribe to new events.
        
        Args:
            handler: Callback invoked for each new event
        """
        self._subscribers.append(handler)
    
    def get_all_events(self) -> Iterator[EventEnvelope[DomainEvent]]:
        """Iterate over all events in the global log."""
        yield from self._global_log
    
    def get_events_since(self, position: int) -> Iterator[EventEnvelope[DomainEvent]]:
        """Get events since a specific position."""
        yield from self._global_log[position:]


# ═══════════════════════════════════════════════════════════════════════════════
# REPOSITORY
# ═══════════════════════════════════════════════════════════════════════════════

class Repository(Generic[TAggregate]):
    """
    Repository for loading and saving aggregates.
    
    Handles the mechanics of event sourcing: loading aggregates from
    event history and persisting new events.
    """
    
    def __init__(
        self,
        event_store: EventStore,
        aggregate_factory: Callable[[AggregateId], TAggregate],
    ) -> None:
        """
        Initialise the repository.
        
        Args:
            event_store: The event store to use
            aggregate_factory: Factory function to create aggregate instances
        """
        self._event_store = event_store
        self._aggregate_factory = aggregate_factory
    
    def get(self, aggregate_id: AggregateId) -> TAggregate:
        """
        Load an aggregate by ID.
        
        Args:
            aggregate_id: The aggregate ID
        
        Returns:
            The reconstituted aggregate
        
        Raises:
            AggregateNotFoundError: If no events exist for the ID
        """
        events = self._event_store.load(aggregate_id)
        
        if not events:
            raise AggregateNotFoundError(aggregate_id)
        
        aggregate = self._aggregate_factory(aggregate_id)
        aggregate.load_from_history(events)
        
        return aggregate
    
    def save(self, aggregate: TAggregate) -> None:
        """
        Persist an aggregate's uncommitted events.
        
        Args:
            aggregate: The aggregate to save
        """
        events = aggregate.uncommitted_events
        
        if not events:
            return
        
        # Calculate expected version (version before these events)
        expected_version = aggregate.version - len(events)
        
        self._event_store.append(aggregate.id, events, expected_version)
        aggregate.clear_uncommitted_events()
        
        logger.debug(
            "Saved %d events for aggregate %s",
            len(events),
            aggregate.id,
        )
    
    def exists(self, aggregate_id: AggregateId) -> bool:
        """Check if an aggregate exists."""
        events = self._event_store.load(aggregate_id)
        return len(events) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE: BANK ACCOUNT AGGREGATE
# ═══════════════════════════════════════════════════════════════════════════════

# Domain Events

@dataclass(frozen=True)
class AccountOpened(DomainEvent):
    """Event: A bank account was opened."""
    
    owner_name: str = ""
    initial_balance: float = 0.0


@dataclass(frozen=True)
class MoneyDeposited(DomainEvent):
    """Event: Money was deposited into the account."""
    
    amount: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class MoneyWithdrawn(DomainEvent):
    """Event: Money was withdrawn from the account."""
    
    amount: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class AccountClosed(DomainEvent):
    """Event: The account was closed."""
    
    reason: str = ""
    final_balance: float = 0.0


# Type alias for bank account events
BankAccountEvent = AccountOpened | MoneyDeposited | MoneyWithdrawn | AccountClosed


# Aggregate

class BankAccount(AggregateRoot[BankAccountEvent]):
    """
    Bank account aggregate demonstrating event sourcing.
    
    All state changes occur through events. The current state (balance,
    owner, etc.) is derived by replaying events.
    """
    
    def __init__(self, aggregate_id: AggregateId | None = None) -> None:
        super().__init__(aggregate_id)
        self._owner_name: str = ""
        self._balance: float = 0.0
        self._is_closed: bool = False
        self._transaction_count: int = 0
    
    @property
    def owner_name(self) -> str:
        """Return the account owner's name."""
        return self._owner_name
    
    @property
    def balance(self) -> float:
        """Return the current balance."""
        return self._balance
    
    @property
    def is_closed(self) -> bool:
        """Return whether the account is closed."""
        return self._is_closed
    
    @property
    def transaction_count(self) -> int:
        """Return the number of transactions."""
        return self._transaction_count
    
    # Commands
    
    @classmethod
    def open(
        cls,
        owner_name: str,
        initial_balance: float = 0.0,
        account_id: AggregateId | None = None,
    ) -> BankAccount:
        """
        Open a new bank account.
        
        Args:
            owner_name: Name of the account owner
            initial_balance: Optional initial deposit
            account_id: Optional account ID
        
        Returns:
            A new BankAccount aggregate
        """
        if initial_balance < 0:
            raise InvalidOperationError("Initial balance cannot be negative")
        
        account = cls(account_id)
        account._raise_event(AccountOpened(
            owner_name=owner_name,
            initial_balance=initial_balance,
        ))
        
        return account
    
    def deposit(self, amount: float, description: str = "") -> None:
        """
        Deposit money into the account.
        
        Args:
            amount: Amount to deposit (must be positive)
            description: Optional transaction description
        """
        self._ensure_not_closed()
        
        if amount <= 0:
            raise InvalidOperationError("Deposit amount must be positive")
        
        self._raise_event(MoneyDeposited(
            amount=amount,
            description=description,
        ))
    
    def withdraw(self, amount: float, description: str = "") -> None:
        """
        Withdraw money from the account.
        
        Args:
            amount: Amount to withdraw (must be positive)
            description: Optional transaction description
        """
        self._ensure_not_closed()
        
        if amount <= 0:
            raise InvalidOperationError("Withdrawal amount must be positive")
        
        if amount > self._balance:
            raise InvalidOperationError(
                f"Insufficient funds: balance is {self._balance}, "
                f"requested {amount}"
            )
        
        self._raise_event(MoneyWithdrawn(
            amount=amount,
            description=description,
        ))
    
    def close(self, reason: str = "") -> None:
        """
        Close the account.
        
        Args:
            reason: Reason for closing
        """
        self._ensure_not_closed()
        
        self._raise_event(AccountClosed(
            reason=reason,
            final_balance=self._balance,
        ))
    
    def _ensure_not_closed(self) -> None:
        """Raise an error if the account is closed."""
        if self._is_closed:
            raise InvalidOperationError("Account is closed")
    
    # Event handlers
    
    def _apply(self, event: BankAccountEvent) -> None:
        """Apply an event to update state."""
        match event:
            case AccountOpened():
                self._apply_account_opened(event)
            case MoneyDeposited():
                self._apply_money_deposited(event)
            case MoneyWithdrawn():
                self._apply_money_withdrawn(event)
            case AccountClosed():
                self._apply_account_closed(event)
    
    def _apply_account_opened(self, event: AccountOpened) -> None:
        """Handle AccountOpened event."""
        self._owner_name = event.owner_name
        self._balance = event.initial_balance
        self._is_closed = False
    
    def _apply_money_deposited(self, event: MoneyDeposited) -> None:
        """Handle MoneyDeposited event."""
        self._balance += event.amount
        self._transaction_count += 1
    
    def _apply_money_withdrawn(self, event: MoneyWithdrawn) -> None:
        """Handle MoneyWithdrawn event."""
        self._balance -= event.amount
        self._transaction_count += 1
    
    def _apply_account_closed(self, event: AccountClosed) -> None:
        """Handle AccountClosed event."""
        self._is_closed = True


# ═══════════════════════════════════════════════════════════════════════════════
# PROJECTIONS (READ MODELS)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AccountSummary:
    """Read model: Summary of an account."""
    
    account_id: str
    owner_name: str
    balance: float
    transaction_count: int
    is_closed: bool
    last_updated: datetime


class AccountSummaryProjection:
    """
    Projection that maintains account summaries.
    
    Subscribes to events and updates a denormalised read model.
    """
    
    def __init__(self) -> None:
        """Initialise the projection."""
        self._summaries: dict[str, AccountSummary] = {}
    
    def handle(self, envelope: EventEnvelope[DomainEvent]) -> None:
        """Handle incoming events."""
        event = envelope.event
        
        match event:
            case AccountOpened():
                self._handle_account_opened(event)
            case MoneyDeposited():
                self._handle_money_deposited(event)
            case MoneyWithdrawn():
                self._handle_money_withdrawn(event)
            case AccountClosed():
                self._handle_account_closed(event)
    
    def _handle_account_opened(self, event: AccountOpened) -> None:
        """Create summary for new account."""
        self._summaries[event.aggregate_id] = AccountSummary(
            account_id=event.aggregate_id,
            owner_name=event.owner_name,
            balance=event.initial_balance,
            transaction_count=0,
            is_closed=False,
            last_updated=event.timestamp,
        )
    
    def _handle_money_deposited(self, event: MoneyDeposited) -> None:
        """Update balance for deposit."""
        summary = self._summaries.get(event.aggregate_id)
        if summary:
            summary.balance += event.amount
            summary.transaction_count += 1
            summary.last_updated = event.timestamp
    
    def _handle_money_withdrawn(self, event: MoneyWithdrawn) -> None:
        """Update balance for withdrawal."""
        summary = self._summaries.get(event.aggregate_id)
        if summary:
            summary.balance -= event.amount
            summary.transaction_count += 1
            summary.last_updated = event.timestamp
    
    def _handle_account_closed(self, event: AccountClosed) -> None:
        """Mark account as closed."""
        summary = self._summaries.get(event.aggregate_id)
        if summary:
            summary.is_closed = True
            summary.last_updated = event.timestamp
    
    def get_summary(self, account_id: str) -> AccountSummary | None:
        """Get summary for an account."""
        return self._summaries.get(account_id)
    
    def get_all_summaries(self) -> list[AccountSummary]:
        """Get all account summaries."""
        return list(self._summaries.values())
    
    def get_total_balance(self) -> float:
        """Get total balance across all open accounts."""
        return sum(
            s.balance for s in self._summaries.values()
            if not s.is_closed
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def demonstrate_event_sourcing() -> None:
    """Demonstrate the event sourcing framework."""
    logger.info("=" * 70)
    logger.info("EVENT SOURCING DEMONSTRATION")
    logger.info("=" * 70)
    
    # Set up infrastructure
    event_store = EventStore()
    projection = AccountSummaryProjection()
    event_store.subscribe(projection.handle)
    
    repository: Repository[BankAccount] = Repository(
        event_store=event_store,
        aggregate_factory=lambda id: BankAccount(id),
    )
    
    # Open a new account
    logger.info("-" * 70)
    logger.info("OPENING ACCOUNT")
    logger.info("-" * 70)
    
    account = BankAccount.open(
        owner_name="Alice Smith",
        initial_balance=1000.0,
        account_id="ACC-001",
    )
    repository.save(account)
    
    logger.info("Account opened: %s", account.id)
    logger.info("Balance: £%.2f", account.balance)
    
    # Perform transactions
    logger.info("-" * 70)
    logger.info("PERFORMING TRANSACTIONS")
    logger.info("-" * 70)
    
    account.deposit(500.0, "Salary")
    account.withdraw(200.0, "Groceries")
    account.deposit(100.0, "Refund")
    account.withdraw(50.0, "Coffee")
    repository.save(account)
    
    logger.info("Balance after transactions: £%.2f", account.balance)
    logger.info("Transaction count: %d", account.transaction_count)
    
    # Load from event store (reconstitute)
    logger.info("-" * 70)
    logger.info("RECONSTITUTING FROM EVENTS")
    logger.info("-" * 70)
    
    loaded_account = repository.get("ACC-001")
    logger.info("Loaded account: %s", loaded_account.id)
    logger.info("Owner: %s", loaded_account.owner_name)
    logger.info("Balance: £%.2f", loaded_account.balance)
    logger.info("Version: %d", loaded_account.version)
    
    # Check projection
    logger.info("-" * 70)
    logger.info("READ MODEL (PROJECTION)")
    logger.info("-" * 70)
    
    summary = projection.get_summary("ACC-001")
    if summary:
        logger.info("Account Summary:")
        logger.info("  Owner: %s", summary.owner_name)
        logger.info("  Balance: £%.2f", summary.balance)
        logger.info("  Transactions: %d", summary.transaction_count)
        logger.info("  Closed: %s", summary.is_closed)
    
    # Open another account
    logger.info("-" * 70)
    logger.info("OPENING SECOND ACCOUNT")
    logger.info("-" * 70)
    
    account2 = BankAccount.open(
        owner_name="Bob Jones",
        initial_balance=500.0,
        account_id="ACC-002",
    )
    account2.deposit(250.0, "Gift")
    repository.save(account2)
    
    logger.info("Total balance across accounts: £%.2f", projection.get_total_balance())
    
    # Event history
    logger.info("-" * 70)
    logger.info("EVENT HISTORY (GLOBAL LOG)")
    logger.info("-" * 70)
    
    for envelope in event_store.get_all_events():
        logger.info(
            "  [%d] %s: %s (v%d)",
            envelope.position,
            envelope.stream_id,
            envelope.event.event_type,
            envelope.event.version,
        )
    
    # Close first account
    logger.info("-" * 70)
    logger.info("CLOSING ACCOUNT")
    logger.info("-" * 70)
    
    # Reload to ensure we have latest version
    account_to_close = repository.get("ACC-001")
    account_to_close.close("Customer request")
    repository.save(account_to_close)
    
    logger.info("Account closed")
    logger.info("Total balance (open accounts only): £%.2f", projection.get_total_balance())
    
    # Demonstrate invalid operation
    logger.info("-" * 70)
    logger.info("INVALID OPERATION HANDLING")
    logger.info("-" * 70)
    
    try:
        account_to_close.deposit(100.0, "Should fail")
    except InvalidOperationError as e:
        logger.info("Caught expected error: %s", e)
    
    # Demonstrate concurrency conflict
    logger.info("-" * 70)
    logger.info("CONCURRENCY CONFLICT DEMONSTRATION")
    logger.info("-" * 70)
    
    # Simulate two users loading the same account
    user1_account = repository.get("ACC-002")
    user2_account = repository.get("ACC-002")
    
    # User 1 makes a change and saves
    user1_account.deposit(100.0, "User 1 deposit")
    repository.save(user1_account)
    
    # User 2 tries to save with stale version
    user2_account.withdraw(50.0, "User 2 withdrawal")
    try:
        repository.save(user2_account)
    except ConcurrencyError as e:
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
        description="Event Sourcing Framework Solution"
    )
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.demo:
        demonstrate_event_sourcing()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
