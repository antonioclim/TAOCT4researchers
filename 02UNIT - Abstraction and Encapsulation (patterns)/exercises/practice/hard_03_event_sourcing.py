#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
02UNIT Practice: Hard Exercise 3 — Event Sourcing Pattern
═══════════════════════════════════════════════════════════════════════════════

Difficulty: ⭐⭐⭐ (Hard)
Estimated Time: 75 minutes

TASK
────
Implement an event sourcing system for a bank account that:
1. Stores all changes as immutable events
2. Rebuilds state by replaying events
3. Supports snapshots for performance
4. Implements event versioning

LEARNING OBJECTIVES
───────────────────
- Understand event sourcing vs traditional state storage
- Implement immutable event streams
- Use generics for type-safe event handling
- Apply the Command Query Responsibility Segregation (CQRS) pattern

BACKGROUND
──────────
Event Sourcing stores application state as a sequence of events rather than
current state. Benefits include:
- Complete audit trail
- Time-travel debugging
- Easier event replay and analysis
- Natural fit for distributed systems

Traditional:  Account { balance: 500 }
Event-Sourced: [Opened(100), Deposited(200), Deposited(300), Withdrawn(100)]

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)
from uuid import UUID, uuid4
import json


# ═══════════════════════════════════════════════════════════════════════════════
# EVENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Event:
    """Base class for all events.
    
    Events are immutable records of something that happened.
    
    Attributes:
        event_id: Unique identifier for this event.
        timestamp: When the event occurred.
        version: Event schema version for migration support.
    """
    event_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1


@dataclass(frozen=True)
class AccountOpened(Event):
    """Event: A new bank account was opened.
    
    Attributes:
        account_id: The new account's identifier.
        owner_name: Name of the account owner.
        initial_balance: Starting balance.
    """
    account_id: str = ""
    owner_name: str = ""
    initial_balance: float = 0.0


@dataclass(frozen=True)
class MoneyDeposited(Event):
    """Event: Money was deposited into an account.
    
    Attributes:
        account_id: The account identifier.
        amount: Amount deposited (positive).
        description: Optional transaction description.
    """
    account_id: str = ""
    amount: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class MoneyWithdrawn(Event):
    """Event: Money was withdrawn from an account.
    
    Attributes:
        account_id: The account identifier.
        amount: Amount withdrawn (positive).
        description: Optional transaction description.
    """
    account_id: str = ""
    amount: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class AccountClosed(Event):
    """Event: An account was closed.
    
    Attributes:
        account_id: The account identifier.
        reason: Reason for closure.
        final_balance: Balance at time of closure.
    """
    account_id: str = ""
    reason: str = ""
    final_balance: float = 0.0


@dataclass(frozen=True)
class InterestApplied(Event):
    """Event: Interest was applied to an account.
    
    Attributes:
        account_id: The account identifier.
        rate: Interest rate applied (as decimal).
        amount: Computed interest amount.
    """
    account_id: str = ""
    rate: float = 0.0
    amount: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# STATE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AccountState:
    """The current state of an account, derived from events.
    
    Attributes:
        account_id: Account identifier.
        owner_name: Account owner.
        balance: Current balance.
        is_active: Whether the account is open.
        transaction_count: Number of transactions.
        created_at: When the account was opened.
        closed_at: When the account was closed (if applicable).
    """
    account_id: str = ""
    owner_name: str = ""
    balance: float = 0.0
    is_active: bool = True
    transaction_count: int = 0
    created_at: datetime | None = None
    closed_at: datetime | None = None


@dataclass(frozen=True)
class Snapshot(Generic[TypeVar('T')]):
    """A point-in-time snapshot of aggregate state.
    
    Snapshots optimise replay by storing state at intervals.
    
    Attributes:
        aggregate_id: The aggregate this snapshot is for.
        state: The snapshotted state.
        version: Event sequence number at snapshot time.
        timestamp: When the snapshot was taken.
    """
    aggregate_id: str
    state: Any
    version: int
    timestamp: datetime = field(default_factory=datetime.now)


EventT = TypeVar('EventT', bound=Event)
StateT = TypeVar('StateT')


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class InsufficientFundsError(Exception):
    """Raised when a withdrawal exceeds available balance."""
    pass


class AccountClosedError(Exception):
    """Raised when operating on a closed account."""
    pass


class ConcurrencyError(Exception):
    """Raised when concurrent modifications conflict."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# YOUR TASK: Implement the EventStore
# ═══════════════════════════════════════════════════════════════════════════════

class EventStore:
    """Stores and retrieves events for aggregates.
    
    This is a simple in-memory implementation. Production systems would
    use a database with optimistic concurrency control.
    
    Example:
        store = EventStore()
        store.append("ACC-001", AccountOpened(...))
        store.append("ACC-001", MoneyDeposited(...))
        events = store.get_events("ACC-001")
    """
    
    def __init__(self) -> None:
        """Initialise the event store."""
        self._events: dict[str, list[Event]] = {}
        self._snapshots: dict[str, Snapshot[Any]] = {}
    
    def append(
        self, 
        aggregate_id: str, 
        event: Event,
        expected_version: int | None = None,
    ) -> int:
        """Append an event to an aggregate's stream.
        
        Args:
            aggregate_id: The aggregate identifier.
            event: The event to append.
            expected_version: For optimistic concurrency (optional).
        
        Returns:
            The new version number.
        
        Raises:
            ConcurrencyError: If expected_version doesn't match.
        """
        # TODO: Implement event appending with optional concurrency check
        pass
    
    def get_events(
        self,
        aggregate_id: str,
        after_version: int = 0,
    ) -> list[Event]:
        """Get events for an aggregate.
        
        Args:
            aggregate_id: The aggregate identifier.
            after_version: Only return events after this version.
        
        Returns:
            List of events in order.
        """
        # TODO: Return events, optionally filtered by version
        pass
    
    def get_version(self, aggregate_id: str) -> int:
        """Get the current version (event count) for an aggregate.
        
        Args:
            aggregate_id: The aggregate identifier.
        
        Returns:
            The current version number.
        """
        # TODO: Return the number of events
        pass
    
    def save_snapshot(self, snapshot: Snapshot[Any]) -> None:
        """Save a snapshot for an aggregate.
        
        Args:
            snapshot: The snapshot to save.
        """
        # TODO: Store the snapshot
        pass
    
    def get_snapshot(self, aggregate_id: str) -> Snapshot[Any] | None:
        """Get the latest snapshot for an aggregate.
        
        Args:
            aggregate_id: The aggregate identifier.
        
        Returns:
            The snapshot or None if no snapshot exists.
        """
        # TODO: Return the snapshot if it exists
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# YOUR TASK: Implement the BankAccount aggregate
# ═══════════════════════════════════════════════════════════════════════════════

class BankAccount:
    """Event-sourced bank account aggregate.
    
    The account state is derived entirely from events. No state is stored
    directly - it's always computed by replaying events.
    
    Example:
        store = EventStore()
        account = BankAccount.open(store, "ACC-001", "Alice", 100.0)
        account.deposit(50.0, "Birthday gift")
        account.withdraw(25.0, "Coffee")
        print(account.balance)  # 125.0
    """
    
    def __init__(self, store: EventStore, account_id: str) -> None:
        """Initialise from an event store.
        
        Args:
            store: The event store.
            account_id: The account identifier.
        """
        self._store = store
        self._account_id = account_id
        self._state = AccountState()
        self._version = 0
        self._reload()
    
    @classmethod
    def open(
        cls,
        store: EventStore,
        account_id: str,
        owner_name: str,
        initial_balance: float = 0.0,
    ) -> "BankAccount":
        """Open a new bank account.
        
        Args:
            store: The event store.
            account_id: The new account's identifier.
            owner_name: Name of the account owner.
            initial_balance: Starting balance.
        
        Returns:
            The new BankAccount instance.
        """
        # TODO: Create AccountOpened event, append to store, return instance
        pass
    
    @property
    def account_id(self) -> str:
        """Get the account identifier."""
        return self._account_id
    
    @property
    def balance(self) -> float:
        """Get the current balance."""
        return self._state.balance
    
    @property
    def is_active(self) -> bool:
        """Check if the account is active."""
        return self._state.is_active
    
    @property
    def transaction_count(self) -> int:
        """Get the number of transactions."""
        return self._state.transaction_count
    
    def deposit(self, amount: float, description: str = "") -> None:
        """Deposit money into the account.
        
        Args:
            amount: Amount to deposit (must be positive).
            description: Optional transaction description.
        
        Raises:
            AccountClosedError: If the account is closed.
            ValueError: If amount is not positive.
        """
        # TODO: Validate, create event, append, reload state
        pass
    
    def withdraw(self, amount: float, description: str = "") -> None:
        """Withdraw money from the account.
        
        Args:
            amount: Amount to withdraw (must be positive).
            description: Optional transaction description.
        
        Raises:
            AccountClosedError: If the account is closed.
            ValueError: If amount is not positive.
            InsufficientFundsError: If balance is insufficient.
        """
        # TODO: Validate, create event, append, reload state
        pass
    
    def apply_interest(self, rate: float) -> float:
        """Apply interest to the account.
        
        Args:
            rate: Interest rate as decimal (e.g., 0.05 for 5%).
        
        Returns:
            The interest amount applied.
        
        Raises:
            AccountClosedError: If the account is closed.
        """
        # TODO: Calculate interest, create event, append, reload, return amount
        pass
    
    def close(self, reason: str = "Customer request") -> None:
        """Close the account.
        
        Args:
            reason: Reason for closure.
        
        Raises:
            AccountClosedError: If already closed.
        """
        # TODO: Create AccountClosed event, append, reload
        pass
    
    def _reload(self) -> None:
        """Reload state from events.
        
        This method rebuilds the account state by replaying all events.
        It first checks for a snapshot to optimise replay.
        """
        # TODO: Check for snapshot, replay events from that point
        # Apply each event type to update state
        pass
    
    def _apply_event(self, event: Event) -> None:
        """Apply a single event to the current state.
        
        Args:
            event: The event to apply.
        """
        # TODO: Update self._state based on event type
        # Handle: AccountOpened, MoneyDeposited, MoneyWithdrawn,
        #         AccountClosed, InterestApplied
        pass
    
    def create_snapshot(self) -> None:
        """Create a snapshot of current state for optimisation."""
        # TODO: Save snapshot to store
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# BONUS: Implement event projection for reporting
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AccountSummary:
    """Projected summary of an account for reporting."""
    account_id: str
    owner_name: str
    current_balance: float
    total_deposits: float
    total_withdrawals: float
    total_interest: float
    transaction_count: int
    is_active: bool


def project_account_summary(store: EventStore, account_id: str) -> AccountSummary:
    """Project an account summary from events.
    
    This demonstrates CQRS - a separate read model from the write model.
    
    Args:
        store: The event store.
        account_id: The account to summarise.
    
    Returns:
        An AccountSummary projection.
    """
    # TODO: Replay events and build summary
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_open_account() -> None:
    """Test opening a new account."""
    store = EventStore()
    account = BankAccount.open(store, "ACC-001", "Alice", 100.0)
    
    assert account.account_id == "ACC-001"
    assert account.balance == 100.0
    assert account.is_active


def test_deposit() -> None:
    """Test depositing money."""
    store = EventStore()
    account = BankAccount.open(store, "ACC-002", "Bob", 50.0)
    
    account.deposit(100.0, "Salary")
    assert account.balance == 150.0
    assert account.transaction_count == 1


def test_withdraw() -> None:
    """Test withdrawing money."""
    store = EventStore()
    account = BankAccount.open(store, "ACC-003", "Charlie", 200.0)
    
    account.withdraw(75.0, "Shopping")
    assert account.balance == 125.0


def test_insufficient_funds() -> None:
    """Test that overdraft is prevented."""
    store = EventStore()
    account = BankAccount.open(store, "ACC-004", "Diana", 50.0)
    
    try:
        account.withdraw(100.0, "Too much")
        assert False, "Should have raised InsufficientFundsError"
    except InsufficientFundsError:
        pass


def test_close_account() -> None:
    """Test closing an account."""
    store = EventStore()
    account = BankAccount.open(store, "ACC-005", "Eve", 100.0)
    
    account.close("Moving abroad")
    assert not account.is_active
    
    # Should not be able to deposit
    try:
        account.deposit(50.0)
        assert False, "Should have raised AccountClosedError"
    except AccountClosedError:
        pass


def test_interest() -> None:
    """Test applying interest."""
    store = EventStore()
    account = BankAccount.open(store, "ACC-006", "Frank", 1000.0)
    
    interest = account.apply_interest(0.05)  # 5%
    assert interest == 50.0
    assert account.balance == 1050.0


def test_event_replay() -> None:
    """Test that state is correctly rebuilt from events."""
    store = EventStore()
    
    # Create and modify account
    account1 = BankAccount.open(store, "ACC-007", "Grace", 100.0)
    account1.deposit(200.0)
    account1.withdraw(50.0)
    account1.apply_interest(0.02)
    
    # Create new instance from same events
    account2 = BankAccount(store, "ACC-007")
    
    assert account2.balance == account1.balance


def test_snapshot() -> None:
    """Test snapshot creation and usage."""
    store = EventStore()
    account = BankAccount.open(store, "ACC-008", "Henry", 100.0)
    
    # Make many transactions
    for i in range(10):
        account.deposit(10.0)
    
    account.create_snapshot()
    
    # Modify after snapshot
    account.deposit(50.0)
    
    # Reload should use snapshot
    account2 = BankAccount(store, "ACC-008")
    assert account2.balance == account.balance


def test_event_store_versioning() -> None:
    """Test event store concurrency control."""
    store = EventStore()
    store.append("AGG-001", AccountOpened(account_id="AGG-001"))
    
    assert store.get_version("AGG-001") == 1
    
    # Should succeed with correct expected version
    store.append("AGG-001", MoneyDeposited(account_id="AGG-001", amount=100.0), 
                 expected_version=1)
    
    # Should fail with wrong expected version
    try:
        store.append("AGG-001", MoneyDeposited(account_id="AGG-001", amount=50.0),
                     expected_version=1)  # Should be 2
        assert False, "Should have raised ConcurrencyError"
    except ConcurrencyError:
        pass


def test_projection() -> None:
    """Test account summary projection."""
    store = EventStore()
    account = BankAccount.open(store, "ACC-009", "Iris", 100.0)
    account.deposit(500.0)
    account.deposit(300.0)
    account.withdraw(150.0)
    account.apply_interest(0.01)
    
    summary = project_account_summary(store, "ACC-009")
    
    assert summary.account_id == "ACC-009"
    assert summary.owner_name == "Iris"
    assert summary.total_deposits == 800.0
    assert summary.total_withdrawals == 150.0


if __name__ == "__main__":
    test_open_account()
    test_deposit()
    test_withdraw()
    test_insufficient_funds()
    test_close_account()
    test_interest()
    test_event_replay()
    test_snapshot()
    test_event_store_versioning()
    test_projection()
    print("All tests passed! ✓")
