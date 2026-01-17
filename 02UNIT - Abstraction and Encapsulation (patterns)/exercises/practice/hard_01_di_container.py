#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
02UNIT Practice: Hard Exercise 1 — Dependency Injection Container
═══════════════════════════════════════════════════════════════════════════════

Difficulty: ⭐⭐⭐ (Hard)
Estimated Time: 60 minutes

TASK
────
Implement a simple dependency injection (DI) container that can:
1. Register services (singleton or transient)
2. Resolve dependencies automatically
3. Handle circular dependency detection

LEARNING OBJECTIVES
───────────────────
- Understand Inversion of Control (IoC)
- Implement service lifetimes (singleton vs transient)
- Use type introspection for automatic resolution
- Handle edge cases like circular dependencies

BACKGROUND
──────────
Dependency Injection is a technique where an object receives its dependencies
from external sources rather than creating them internally. This promotes:
- Loose coupling between components
- Easier testing through mock injection
- Better separation of concerns

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Protocol,
    TypeVar,
    get_type_hints,
    runtime_checkable,
)
import inspect


T = TypeVar('T')


class ServiceLifetime(Enum):
    """Defines how long a service instance lives."""
    SINGLETON = auto()  # One instance for the entire container
    TRANSIENT = auto()  # New instance every time


@dataclass
class ServiceDescriptor:
    """Describes a registered service.
    
    Attributes:
        service_type: The type/interface being registered.
        implementation: The concrete implementation or factory.
        lifetime: How long instances should live.
        instance: Cached instance for singletons.
    """
    service_type: type
    implementation: type | Callable[..., Any]
    lifetime: ServiceLifetime
    instance: Any = None


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected."""
    pass


class ServiceNotRegisteredError(Exception):
    """Raised when requesting an unregistered service."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# YOUR TASK: Implement the DIContainer class
# ═══════════════════════════════════════════════════════════════════════════════

class DIContainer:
    """A simple dependency injection container.
    
    Example usage:
        container = DIContainer()
        
        # Register services
        container.register(ILogger, ConsoleLogger, ServiceLifetime.SINGLETON)
        container.register(IDatabase, SqlDatabase, ServiceLifetime.TRANSIENT)
        
        # Resolve services
        logger = container.resolve(ILogger)
        db = container.resolve(IDatabase)
    """
    
    def __init__(self) -> None:
        """Initialise the container."""
        self._services: dict[type, ServiceDescriptor] = {}
        self._resolving: set[type] = set()  # For circular dependency detection
    
    def register(
        self,
        service_type: type[T],
        implementation: type[T] | Callable[..., T],
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
    ) -> None:
        """Register a service with the container.
        
        Args:
            service_type: The interface/base type to register.
            implementation: The concrete type or factory function.
            lifetime: SINGLETON or TRANSIENT.
        
        Example:
            container.register(ILogger, ConsoleLogger, ServiceLifetime.SINGLETON)
        """
        # TODO: Store the service descriptor
        pass
    
    def register_instance(self, service_type: type[T], instance: T) -> None:
        """Register a pre-existing instance as a singleton.
        
        Args:
            service_type: The interface/base type.
            instance: The pre-created instance.
        
        Example:
            config = Configuration(debug=True)
            container.register_instance(Configuration, config)
        """
        # TODO: Register with the instance already set
        pass
    
    def resolve(self, service_type: type[T]) -> T:
        """Resolve a service, creating it if necessary.
        
        Args:
            service_type: The type to resolve.
        
        Returns:
            An instance of the requested type.
        
        Raises:
            ServiceNotRegisteredError: If the type is not registered.
            CircularDependencyError: If circular dependencies are detected.
        
        Example:
            logger = container.resolve(ILogger)
        """
        # TODO: Implement resolution with:
        # 1. Check if service is registered
        # 2. Detect circular dependencies
        # 3. Return cached instance for singletons
        # 4. Create new instance for transients
        # 5. Auto-resolve constructor dependencies
        pass
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create a new instance of a service.
        
        This method should:
        1. Get the implementation's __init__ signature
        2. Resolve all typed parameters from the container
        3. Create and return the instance
        
        Args:
            descriptor: The service descriptor.
        
        Returns:
            A new instance.
        """
        # TODO: Implement instance creation with dependency resolution
        # Hint: Use get_type_hints() and inspect.signature()
        pass
    
    def is_registered(self, service_type: type) -> bool:
        """Check if a service type is registered.
        
        Args:
            service_type: The type to check.
        
        Returns:
            True if registered, False otherwise.
        """
        # TODO: Implement
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE SERVICES FOR TESTING
# ═══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class ILogger(Protocol):
    """Logger interface."""
    def log(self, message: str) -> None: ...


@runtime_checkable
class IDatabase(Protocol):
    """Database interface."""
    def query(self, sql: str) -> list[dict[str, Any]]: ...


@runtime_checkable
class ICache(Protocol):
    """Cache interface."""
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any) -> None: ...


@dataclass
class ConsoleLogger:
    """Console logger implementation."""
    prefix: str = "[LOG]"
    
    def log(self, message: str) -> None:
        print(f"{self.prefix} {message}")


@dataclass
class InMemoryDatabase:
    """In-memory database implementation."""
    logger: ILogger  # Dependency!
    
    def query(self, sql: str) -> list[dict[str, Any]]:
        self.logger.log(f"Executing: {sql}")
        return [{"result": "data"}]


@dataclass
class InMemoryCache:
    """In-memory cache implementation."""
    _data: dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str) -> Any | None:
        return self._data.get(key)
    
    def set(self, key: str, value: Any) -> None:
        self._data[key] = value


@dataclass
class UserService:
    """Service that depends on multiple other services."""
    database: IDatabase
    logger: ILogger
    cache: ICache
    
    def get_user(self, user_id: int) -> dict[str, Any]:
        cached = self.cache.get(f"user:{user_id}")
        if cached:
            self.logger.log(f"Cache hit for user {user_id}")
            return cached
        
        result = self.database.query(f"SELECT * FROM users WHERE id = {user_id}")
        if result:
            self.cache.set(f"user:{user_id}", result[0])
        return result[0] if result else {}


# For circular dependency testing
@dataclass
class ServiceA:
    """Service that depends on ServiceB."""
    b: "ServiceB"


@dataclass
class ServiceB:
    """Service that depends on ServiceA (circular!)."""
    a: "ServiceA"


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_register_and_resolve() -> None:
    """Test basic registration and resolution."""
    container = DIContainer()
    container.register(ILogger, ConsoleLogger)
    
    logger = container.resolve(ILogger)
    assert isinstance(logger, ConsoleLogger)


def test_singleton_lifetime() -> None:
    """Test that singletons return the same instance."""
    container = DIContainer()
    container.register(ILogger, ConsoleLogger, ServiceLifetime.SINGLETON)
    
    logger1 = container.resolve(ILogger)
    logger2 = container.resolve(ILogger)
    
    assert logger1 is logger2


def test_transient_lifetime() -> None:
    """Test that transients return new instances."""
    container = DIContainer()
    container.register(ICache, InMemoryCache, ServiceLifetime.TRANSIENT)
    
    cache1 = container.resolve(ICache)
    cache2 = container.resolve(ICache)
    
    assert cache1 is not cache2


def test_register_instance() -> None:
    """Test registering a pre-existing instance."""
    container = DIContainer()
    
    existing_logger = ConsoleLogger(prefix="[CUSTOM]")
    container.register_instance(ILogger, existing_logger)
    
    resolved = container.resolve(ILogger)
    assert resolved is existing_logger
    assert resolved.prefix == "[CUSTOM]"


def test_auto_resolve_dependencies() -> None:
    """Test that constructor dependencies are auto-resolved."""
    container = DIContainer()
    container.register(ILogger, ConsoleLogger, ServiceLifetime.SINGLETON)
    container.register(IDatabase, InMemoryDatabase)
    
    db = container.resolve(IDatabase)
    assert isinstance(db, InMemoryDatabase)
    assert isinstance(db.logger, ConsoleLogger)


def test_complex_dependency_chain() -> None:
    """Test resolution of services with multiple dependencies."""
    container = DIContainer()
    container.register(ILogger, ConsoleLogger, ServiceLifetime.SINGLETON)
    container.register(IDatabase, InMemoryDatabase)
    container.register(ICache, InMemoryCache)
    container.register(UserService, UserService)
    
    service = container.resolve(UserService)
    assert isinstance(service.database, InMemoryDatabase)
    assert isinstance(service.logger, ConsoleLogger)
    assert isinstance(service.cache, InMemoryCache)


def test_service_not_registered() -> None:
    """Test that unregistered services raise an error."""
    container = DIContainer()
    
    try:
        container.resolve(ILogger)
        assert False, "Should have raised ServiceNotRegisteredError"
    except ServiceNotRegisteredError:
        pass


def test_circular_dependency_detection() -> None:
    """Test that circular dependencies are detected."""
    container = DIContainer()
    container.register(ServiceA, ServiceA)
    container.register(ServiceB, ServiceB)
    
    try:
        container.resolve(ServiceA)
        assert False, "Should have raised CircularDependencyError"
    except CircularDependencyError:
        pass


def test_is_registered() -> None:
    """Test the is_registered method."""
    container = DIContainer()
    
    assert not container.is_registered(ILogger)
    
    container.register(ILogger, ConsoleLogger)
    assert container.is_registered(ILogger)


if __name__ == "__main__":
    test_register_and_resolve()
    test_singleton_lifetime()
    test_transient_lifetime()
    test_register_instance()
    test_auto_resolve_dependencies()
    test_complex_dependency_chain()
    test_service_not_registered()
    test_circular_dependency_detection()
    test_is_registered()
    print("All tests passed! ✓")
