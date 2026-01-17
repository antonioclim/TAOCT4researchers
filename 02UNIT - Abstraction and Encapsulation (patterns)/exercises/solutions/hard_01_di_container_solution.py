#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
02UNIT, Hard Exercise 01: Dependency Injection Container — SOLUTION
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Dependency Injection (DI) containers constitute a cornerstone of enterprise
software architecture, enabling loose coupling between components through
inversion of control. This exercise implements a type-safe DI container
supporting constructor injection, service lifetime management and automatic
dependency resolution via type introspection.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Implement service registration with configurable lifetimes
2. Resolve dependencies automatically through type inspection
3. Detect and prevent circular dependency chains
4. Apply the Dependency Inversion Principle in practice

ESTIMATED TIME
──────────────
- Reading: 10 minutes
- Implementation: 20 minutes
- Total: 30 minutes

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generic, TypeVar, get_type_hints

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

T = TypeVar("T")
TService = TypeVar("TService")


class ServiceLifetime(Enum):
    """Specifies the lifetime of a registered service."""
    
    TRANSIENT = auto()  # New instance per resolution
    SINGLETON = auto()  # Single shared instance


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class DependencyInjectionError(Exception):
    """Base exception for DI-related errors."""
    pass


class ServiceNotRegisteredError(DependencyInjectionError):
    """Raised when attempting to resolve an unregistered service."""
    
    def __init__(self, service_type: type) -> None:
        self.service_type = service_type
        super().__init__(f"Service not registered: {service_type.__name__}")


class CircularDependencyError(DependencyInjectionError):
    """Raised when a circular dependency chain is detected."""
    
    def __init__(self, chain: list[type]) -> None:
        self.chain = chain
        chain_str = " → ".join(t.__name__ for t in chain)
        super().__init__(f"Circular dependency detected: {chain_str}")


class DuplicateRegistrationError(DependencyInjectionError):
    """Raised when attempting to register a service type twice."""
    
    def __init__(self, service_type: type) -> None:
        self.service_type = service_type
        super().__init__(f"Service already registered: {service_type.__name__}")


# ═══════════════════════════════════════════════════════════════════════════════
# SERVICE REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ServiceRegistration(Generic[TService]):
    """
    Represents a registered service in the container.
    
    Attributes:
        service_type: The abstract type (interface) being registered
        implementation_type: The concrete type implementing the service
        factory: Optional factory function for custom instantiation
        lifetime: Service lifetime (TRANSIENT or SINGLETON)
        instance: Cached instance for SINGLETON lifetime
    """
    
    service_type: type[TService]
    implementation_type: type[TService]
    factory: Callable[..., TService] | None = None
    lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT
    instance: TService | None = field(default=None, repr=False)


# ═══════════════════════════════════════════════════════════════════════════════
# DEPENDENCY INJECTION CONTAINER
# ═══════════════════════════════════════════════════════════════════════════════

class Container:
    """
    A lightweight dependency injection container supporting constructor injection.
    
    The container manages service registration and resolution, automatically
    injecting dependencies based on constructor type hints. Supports both
    transient (new instance per request) and singleton (shared instance)
    lifetimes.
    
    Example:
        >>> container = Container()
        >>> container.register(ILogger, ConsoleLogger, ServiceLifetime.SINGLETON)
        >>> container.register(IRepository, SqlRepository)
        >>> logger = container.resolve(ILogger)
    """
    
    def __init__(self) -> None:
        """Initialise an empty container."""
        self._registrations: dict[type, ServiceRegistration[Any]] = {}
        self._resolution_stack: list[type] = []
    
    def register(
        self,
        service_type: type[TService],
        implementation_type: type[TService] | None = None,
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
    ) -> Container:
        """
        Register a service type with its implementation.
        
        Args:
            service_type: The abstract type (interface) to register
            implementation_type: The concrete implementation type.
                                 If None, service_type is used (self-registration)
            lifetime: Service lifetime (TRANSIENT or SINGLETON)
        
        Returns:
            Self for method chaining
        
        Raises:
            DuplicateRegistrationError: If service_type already registered
        """
        if service_type in self._registrations:
            raise DuplicateRegistrationError(service_type)
        
        impl = implementation_type or service_type
        
        registration = ServiceRegistration(
            service_type=service_type,
            implementation_type=impl,
            lifetime=lifetime,
        )
        
        self._registrations[service_type] = registration
        logger.debug(
            "Registered %s → %s (%s)",
            service_type.__name__,
            impl.__name__,
            lifetime.name,
        )
        
        return self
    
    def register_factory(
        self,
        service_type: type[TService],
        factory: Callable[..., TService],
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
    ) -> Container:
        """
        Register a service with a custom factory function.
        
        Args:
            service_type: The abstract type to register
            factory: Callable that creates service instances
            lifetime: Service lifetime
        
        Returns:
            Self for method chaining
        """
        if service_type in self._registrations:
            raise DuplicateRegistrationError(service_type)
        
        registration = ServiceRegistration(
            service_type=service_type,
            implementation_type=service_type,
            factory=factory,
            lifetime=lifetime,
        )
        
        self._registrations[service_type] = registration
        logger.debug("Registered %s with factory (%s)", service_type.__name__, lifetime.name)
        
        return self
    
    def register_instance(
        self,
        service_type: type[TService],
        instance: TService,
    ) -> Container:
        """
        Register a pre-existing instance as a singleton.
        
        Args:
            service_type: The abstract type to register
            instance: The pre-existing instance to use
        
        Returns:
            Self for method chaining
        """
        if service_type in self._registrations:
            raise DuplicateRegistrationError(service_type)
        
        registration = ServiceRegistration(
            service_type=service_type,
            implementation_type=type(instance),
            lifetime=ServiceLifetime.SINGLETON,
            instance=instance,
        )
        
        self._registrations[service_type] = registration
        logger.debug("Registered %s instance", service_type.__name__)
        
        return self
    
    def resolve(self, service_type: type[TService]) -> TService:
        """
        Resolve a service, automatically injecting dependencies.
        
        Args:
            service_type: The type to resolve
        
        Returns:
            An instance of the requested service
        
        Raises:
            ServiceNotRegisteredError: If service_type not registered
            CircularDependencyError: If circular dependency detected
        """
        if service_type not in self._registrations:
            raise ServiceNotRegisteredError(service_type)
        
        # Check for circular dependencies
        if service_type in self._resolution_stack:
            cycle = self._resolution_stack[self._resolution_stack.index(service_type):]
            cycle.append(service_type)
            raise CircularDependencyError(cycle)
        
        registration = self._registrations[service_type]
        
        # Return cached singleton if available
        if registration.lifetime == ServiceLifetime.SINGLETON and registration.instance is not None:
            logger.debug("Returning cached singleton: %s", service_type.__name__)
            return registration.instance
        
        # Track resolution for circular dependency detection
        self._resolution_stack.append(service_type)
        
        try:
            instance = self._create_instance(registration)
            
            # Cache singleton instances
            if registration.lifetime == ServiceLifetime.SINGLETON:
                registration.instance = instance
            
            return instance
        finally:
            self._resolution_stack.pop()
    
    def _create_instance(self, registration: ServiceRegistration[TService]) -> TService:
        """
        Create an instance of the registered service.
        
        Resolves constructor dependencies recursively via type hints.
        """
        # Use factory if provided
        if registration.factory is not None:
            return registration.factory()
        
        impl_type = registration.implementation_type
        
        # Get constructor parameters
        try:
            hints = get_type_hints(impl_type.__init__)
        except Exception:
            hints = {}
        
        # Remove 'return' hint if present
        hints.pop("return", None)
        
        # Resolve each dependency
        resolved_deps: dict[str, Any] = {}
        
        sig = inspect.signature(impl_type.__init__)
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            
            # Get type hint for parameter
            param_type = hints.get(param_name)
            
            if param_type is None:
                # No type hint — check for default value
                if param.default is not inspect.Parameter.empty:
                    continue
                raise DependencyInjectionError(
                    f"Cannot resolve parameter '{param_name}' of {impl_type.__name__}: "
                    "no type hint and no default value"
                )
            
            # Handle optional parameters with defaults
            if param.default is not inspect.Parameter.empty:
                if param_type not in self._registrations:
                    continue
            
            # Recursively resolve dependency
            resolved_deps[param_name] = self.resolve(param_type)
        
        logger.debug("Creating instance of %s", impl_type.__name__)
        return impl_type(**resolved_deps)
    
    def is_registered(self, service_type: type) -> bool:
        """Check if a service type is registered."""
        return service_type in self._registrations
    
    def clear(self) -> None:
        """Clear all registrations."""
        self._registrations.clear()
        self._resolution_stack.clear()
        logger.debug("Container cleared")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE SERVICES FOR DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

class ILogger(ABC):
    """Abstract logging service interface."""
    
    @abstractmethod
    def log(self, message: str) -> None:
        """Log a message."""
        ...


class IRepository(ABC):
    """Abstract data repository interface."""
    
    @abstractmethod
    def find_by_id(self, entity_id: int) -> dict[str, Any] | None:
        """Find an entity by ID."""
        ...
    
    @abstractmethod
    def save(self, entity: dict[str, Any]) -> int:
        """Save an entity and return its ID."""
        ...


class INotificationService(ABC):
    """Abstract notification service interface."""
    
    @abstractmethod
    def notify(self, user_id: int, message: str) -> None:
        """Send a notification to a user."""
        ...


class ConsoleLogger(ILogger):
    """Console-based logger implementation."""
    
    def log(self, message: str) -> None:
        """Log a message to the console."""
        logger.info("[ConsoleLogger] %s", message)


class InMemoryRepository(IRepository):
    """In-memory repository implementation."""
    
    def __init__(self, logger: ILogger) -> None:
        self._logger = logger
        self._data: dict[int, dict[str, Any]] = {}
        self._next_id = 1
    
    def find_by_id(self, entity_id: int) -> dict[str, Any] | None:
        """Find an entity by ID."""
        self._logger.log(f"Finding entity with ID: {entity_id}")
        return self._data.get(entity_id)
    
    def save(self, entity: dict[str, Any]) -> int:
        """Save an entity and return its ID."""
        entity_id = self._next_id
        self._next_id += 1
        self._data[entity_id] = entity.copy()
        self._data[entity_id]["id"] = entity_id
        self._logger.log(f"Saved entity with ID: {entity_id}")
        return entity_id


class EmailNotificationService(INotificationService):
    """Email-based notification service."""
    
    def __init__(self, logger: ILogger) -> None:
        self._logger = logger
    
    def notify(self, user_id: int, message: str) -> None:
        """Send an email notification (simulated)."""
        self._logger.log(f"Sending email to user {user_id}: {message}")


class UserService:
    """
    High-level service demonstrating constructor injection.
    
    Depends on ILogger, IRepository and INotificationService — all injected
    automatically by the DI container.
    """
    
    def __init__(
        self,
        logger: ILogger,
        repository: IRepository,
        notification_service: INotificationService,
    ) -> None:
        self._logger = logger
        self._repository = repository
        self._notification_service = notification_service
    
    def create_user(self, name: str, email: str) -> int:
        """Create a new user and send a welcome notification."""
        self._logger.log(f"Creating user: {name}")
        
        user_id = self._repository.save({"name": name, "email": email})
        self._notification_service.notify(user_id, "Welcome to our platform!")
        
        return user_id
    
    def get_user(self, user_id: int) -> dict[str, Any] | None:
        """Retrieve a user by ID."""
        return self._repository.find_by_id(user_id)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def demonstrate_di_container() -> None:
    """Demonstrate the dependency injection container in action."""
    logger.info("=" * 70)
    logger.info("DEPENDENCY INJECTION CONTAINER DEMONSTRATION")
    logger.info("=" * 70)
    
    # Create and configure container
    container = Container()
    
    # Register services with appropriate lifetimes
    container.register(ILogger, ConsoleLogger, ServiceLifetime.SINGLETON)
    container.register(IRepository, InMemoryRepository, ServiceLifetime.SINGLETON)
    container.register(INotificationService, EmailNotificationService, ServiceLifetime.TRANSIENT)
    container.register(UserService, UserService, ServiceLifetime.TRANSIENT)
    
    logger.info("-" * 70)
    logger.info("Services registered. Resolving UserService...")
    logger.info("-" * 70)
    
    # Resolve the top-level service — dependencies are injected automatically
    user_service = container.resolve(UserService)
    
    # Use the service
    user_id = user_service.create_user("Alice", "alice@example.com")
    
    logger.info("-" * 70)
    logger.info("Retrieving created user...")
    logger.info("-" * 70)
    
    user = user_service.get_user(user_id)
    logger.info("Retrieved user: %s", user)
    
    # Demonstrate singleton behaviour
    logger.info("-" * 70)
    logger.info("Verifying singleton behaviour...")
    logger.info("-" * 70)
    
    logger1 = container.resolve(ILogger)
    logger2 = container.resolve(ILogger)
    logger.info("Logger instances identical: %s", logger1 is logger2)
    
    # Demonstrate circular dependency detection
    logger.info("-" * 70)
    logger.info("Demonstrating circular dependency detection...")
    logger.info("-" * 70)
    
    class ServiceA:
        def __init__(self, b: "ServiceB") -> None:
            self.b = b
    
    class ServiceB:
        def __init__(self, a: ServiceA) -> None:
            self.a = a
    
    circular_container = Container()
    circular_container.register(ServiceA, ServiceA)
    circular_container.register(ServiceB, ServiceB)
    
    try:
        circular_container.resolve(ServiceA)
    except CircularDependencyError as e:
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
        description="Dependency Injection Container Solution"
    )
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.demo:
        demonstrate_di_container()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
