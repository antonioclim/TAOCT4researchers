#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
02UNIT: Tests for Lab 2.02 — Design Patterns
═══════════════════════════════════════════════════════════════════════════════

Test suite for design pattern implementations.

Coverage targets:
- Strategy pattern (integration strategies)
- Observer pattern (observable events)
- Factory pattern (agent creation)
- Decorator pattern (function wrappers)
- Command pattern (undo/redo)

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

import pytest
import time
import sys
from pathlib import Path
from typing import Any

# Add lab directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lab"))

from lab_02_02_design_patterns import (
    # Strategy pattern
    RectangleRule,
    TrapezoidRule,
    SimpsonRule,
    NumericalIntegrator,
    # Observer pattern
    Observable,
    MetricsCollector,
    # Factory pattern
    AgentFactory,
    CooperativeAgentFactory,
    CompetitiveAgentFactory,
    # Decorators
    timed,
    cached,
    validated,
    retry,
    # Command pattern
    CommandHistory,
)


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY PATTERN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategyPattern:
    """Tests for numerical integration strategies."""
    
    def test_rectangle_rule_constant(self):
        """Test rectangle rule with constant function."""
        integrator = NumericalIntegrator(RectangleRule())
        
        # Integral of 1 from 0 to 1 should be 1
        def func(x: float) -> float:
            return 1.0
        result = integrator.integrate(func, 0.0, 1.0, n=100)
        assert abs(result - 1.0) < 0.1
    
    def test_rectangle_rule_linear(self):
        """Test rectangle rule with linear function."""
        integrator = NumericalIntegrator(RectangleRule())
        
        # Integral of x from 0 to 1 should be 0.5
        def func(x: float) -> float:
            return x
        result = integrator.integrate(func, 0.0, 1.0, n=1000)
        assert abs(result - 0.5) < 0.01
    
    def test_trapezoid_rule_quadratic(self):
        """Test trapezoid rule with quadratic function."""
        integrator = NumericalIntegrator(TrapezoidRule())
        
        # Integral of x² from 0 to 1 should be 1/3
        def func(x: float) -> float:
            return x ** 2
        result = integrator.integrate(func, 0.0, 1.0, n=1000)
        assert abs(result - 1/3) < 0.001
    
    def test_simpson_rule_polynomial(self):
        """Test Simpson's rule with polynomial function."""
        integrator = NumericalIntegrator(SimpsonRule())
        
        # Integral of x³ from 0 to 1 should be 0.25
        def func(x: float) -> float:
            return x ** 3
        result = integrator.integrate(func, 0.0, 1.0, n=100)
        assert abs(result - 0.25) < 0.001
    
    def test_strategy_swap(self):
        """Test swapping integration strategies at runtime."""
        integrator = NumericalIntegrator(RectangleRule())
        
        def func(x: float) -> float:
            return x ** 2
        result1 = integrator.integrate(func, 0.0, 1.0, n=100)
        
        integrator.strategy = SimpsonRule()
        result2 = integrator.integrate(func, 0.0, 1.0, n=100)
        
        # Simpson's should be more accurate
        expected = 1/3
        assert abs(result2 - expected) < abs(result1 - expected)
    
    def test_strategy_name(self):
        """Test that strategies have descriptive names."""
        assert "Rectangle" in RectangleRule().name
        assert "Trapezoid" in TrapezoidRule().name
        assert "Simpson" in SimpsonRule().name


# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVER PATTERN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestObserverPattern:
    """Tests for observable pattern implementation."""
    
    def test_subscribe_and_notify(self, mock_observer):
        """Test basic subscribe and notification."""
        observable = Observable[int]()
        observable.subscribe(mock_observer.update)
        
        observable.notify(42)
        
        assert mock_observer.call_count == 1
        assert mock_observer.notifications[0] == 42
    
    def test_multiple_subscribers(self, multiple_observers):
        """Test multiple subscribers receive notifications."""
        observable = Observable[str]()
        
        for obs in multiple_observers:
            observable.subscribe(obs.update)
        
        observable.notify("hello")
        
        for obs in multiple_observers:
            assert obs.call_count == 1
            assert obs.notifications[0] == "hello"
    
    def test_unsubscribe(self, mock_observer):
        """Test unsubscribing stops notifications."""
        observable = Observable[int]()
        observable.subscribe(mock_observer.update)
        
        observable.notify(1)
        assert mock_observer.call_count == 1
        
        observable.unsubscribe(mock_observer.update)
        observable.notify(2)
        
        assert mock_observer.call_count == 1  # No new notification
    
    def test_metrics_collector(self):
        """Test MetricsCollector observer."""
        observable = Observable[float]()
        collector = MetricsCollector()
        
        observable.subscribe(collector.update)
        
        observable.notify(10.0)
        observable.notify(20.0)
        observable.notify(30.0)
        
        assert collector.count == 3
        assert abs(collector.mean - 20.0) < 1e-9
        assert collector.min_value == 10.0
        assert collector.max_value == 30.0


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY PATTERN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFactoryPattern:
    """Tests for agent factory implementations."""
    
    def test_cooperative_factory(self):
        """Test cooperative agent factory."""
        factory = CooperativeAgentFactory()
        agent = factory.create("agent_1")
        
        assert agent is not None
        assert hasattr(agent, 'act')
        assert hasattr(agent, 'agent_id')
        assert agent.agent_id == "agent_1"
    
    def test_competitive_factory(self):
        """Test competitive agent factory."""
        factory = CompetitiveAgentFactory()
        agent = factory.create("competitor")
        
        assert agent is not None
        assert hasattr(agent, 'act')
    
    def test_factory_produces_different_agents(self):
        """Test that different factories produce different agent types."""
        coop_factory = CooperativeAgentFactory()
        comp_factory = CompetitiveAgentFactory()
        
        coop_agent = coop_factory.create("coop")
        comp_agent = comp_factory.create("comp")
        
        # They should have different behaviour
        assert coop_agent.__class__ is not comp_agent.__class__
    
    def test_multiple_agent_creation(self):
        """Test creating multiple agents."""
        factory = CooperativeAgentFactory()
        
        agents = [factory.create(f"agent_{i}") for i in range(5)]
        
        assert len(agents) == 5
        assert len(set(a.agent_id for a in agents)) == 5  # All unique IDs


# ═══════════════════════════════════════════════════════════════════════════════
# DECORATOR PATTERN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDecoratorPattern:
    """Tests for function decorators."""
    
    def test_timed_decorator(self):
        """Test that timed decorator measures execution time."""
        @timed
        def slow_function() -> int:
            time.sleep(0.01)
            return 42
        
        result = slow_function()
        assert result == 42
    
    def test_cached_decorator(self):
        """Test that cached decorator memoises results."""
        call_count = 0
        
        @cached
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First calls
        assert expensive_function(5) == 10
        assert expensive_function(5) == 10
        assert expensive_function(5) == 10
        
        # Should only have computed once
        assert call_count == 1
        
        # Different argument should compute again
        assert expensive_function(10) == 20
        assert call_count == 2
    
    def test_validated_decorator(self):
        """Test that validated decorator checks inputs."""
        @validated
        def divide(a: float, b: float) -> float:
            return a / b
        
        # Valid inputs should work
        assert divide(10.0, 2.0) == 5.0
        
        # Invalid inputs should raise
        with pytest.raises((ValueError, ZeroDivisionError)):
            divide(10.0, 0.0)
    
    def test_retry_decorator(self):
        """Test that retry decorator retries on failure."""
        attempt_count = 0
        
        @retry(max_attempts=3)
        def flaky_function() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise RuntimeError("Temporary failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert attempt_count == 3
    
    def test_retry_exhausted(self):
        """Test that retry raises after max attempts."""
        @retry(max_attempts=2)
        def always_fails() -> None:
            raise RuntimeError("Always fails")
        
        with pytest.raises(RuntimeError):
            always_fails()


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND PATTERN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCommandPattern:
    """Tests for command pattern with undo/redo."""
    
    def test_execute_command(self):
        """Test basic command execution."""
        history = CommandHistory()
        data = [1, 2, 3]
        
        class AppendCommand:
            def __init__(self, lst: list, value: int):
                self.lst = lst
                self.value = value
                self.executed = False
            
            def execute(self) -> None:
                self.lst.append(self.value)
                self.executed = True
            
            def undo(self) -> None:
                self.lst.pop()
                self.executed = False
        
        cmd = AppendCommand(data, 4)
        history.execute(cmd)
        
        assert data == [1, 2, 3, 4]
    
    def test_undo_command(self):
        """Test undoing a command."""
        history = CommandHistory()
        data = [1, 2, 3]
        
        class AppendCommand:
            def __init__(self, lst: list, value: int):
                self.lst = lst
                self.value = value
            
            def execute(self) -> None:
                self.lst.append(self.value)
            
            def undo(self) -> None:
                self.lst.pop()
        
        cmd = AppendCommand(data, 4)
        history.execute(cmd)
        assert data == [1, 2, 3, 4]
        
        history.undo()
        assert data == [1, 2, 3]
    
    def test_redo_command(self):
        """Test redoing an undone command."""
        history = CommandHistory()
        data = [1, 2, 3]
        
        class AppendCommand:
            def __init__(self, lst: list, value: int):
                self.lst = lst
                self.value = value
            
            def execute(self) -> None:
                self.lst.append(self.value)
            
            def undo(self) -> None:
                self.lst.pop()
        
        cmd = AppendCommand(data, 4)
        history.execute(cmd)
        history.undo()
        assert data == [1, 2, 3]
        
        history.redo()
        assert data == [1, 2, 3, 4]
    
    def test_multiple_undo_redo(self):
        """Test multiple undo/redo operations."""
        history = CommandHistory()
        data = []
        
        class AppendCommand:
            def __init__(self, lst: list, value: int):
                self.lst = lst
                self.value = value
            
            def execute(self) -> None:
                self.lst.append(self.value)
            
            def undo(self) -> None:
                self.lst.pop()
        
        # Execute multiple commands
        for i in range(5):
            history.execute(AppendCommand(data, i))
        
        assert data == [0, 1, 2, 3, 4]
        
        # Undo all
        for _ in range(5):
            history.undo()
        
        assert data == []
        
        # Redo some
        for _ in range(3):
            history.redo()
        
        assert data == [0, 1, 2]
    
    def test_can_undo_redo(self):
        """Test can_undo and can_redo methods."""
        history = CommandHistory()
        data = []
        
        class AppendCommand:
            def __init__(self, lst: list, value: int):
                self.lst = lst
                self.value = value
            
            def execute(self) -> None:
                self.lst.append(self.value)
            
            def undo(self) -> None:
                self.lst.pop()
        
        assert not history.can_undo()
        assert not history.can_redo()
        
        history.execute(AppendCommand(data, 1))
        assert history.can_undo()
        assert not history.can_redo()
        
        history.undo()
        assert not history.can_undo()
        assert history.can_redo()


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPatternIntegration:
    """Integration tests combining multiple patterns."""
    
    def test_strategy_with_observer(self):
        """Test strategy pattern with observer notifications."""
        observable = Observable[float]()
        collector = MetricsCollector()
        observable.subscribe(collector.update)
        
        integrator = NumericalIntegrator(SimpsonRule())
        
        # Compute multiple integrals and observe results
        for a, b in [(0, 1), (0, 2), (1, 3)]:
            def func(x: float) -> float:
                return x
            result = integrator.integrate(func, a, b, n=100)
            observable.notify(result)
        
        assert collector.count == 3
    
    def test_factory_with_command(self):
        """Test factory pattern with command history."""
        factory = CooperativeAgentFactory()
        history = CommandHistory()
        agents: list[Any] = []
        
        class CreateAgentCommand:
            def __init__(self, factory: AgentFactory, agent_id: str, target: list):
                self.factory = factory
                self.agent_id = agent_id
                self.target = target
                self.agent = None
            
            def execute(self) -> None:
                self.agent = self.factory.create(self.agent_id)
                self.target.append(self.agent)
            
            def undo(self) -> None:
                self.target.remove(self.agent)
        
        # Create agents through commands
        history.execute(CreateAgentCommand(factory, "agent_1", agents))
        history.execute(CreateAgentCommand(factory, "agent_2", agents))
        
        assert len(agents) == 2
        
        # Undo one creation
        history.undo()
        assert len(agents) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
