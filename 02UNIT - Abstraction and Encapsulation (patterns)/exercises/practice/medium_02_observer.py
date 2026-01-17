#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
02UNIT Practice: Medium Exercise 2 — Observer Pattern
═══════════════════════════════════════════════════════════════════════════════

Difficulty: ⭐⭐ (Medium)
Estimated Time: 30 minutes

TASK
────
Implement a temperature sensor with multiple observers:
- ConsoleDisplay: prints temperature to console
- TemperatureLogger: records temperature history
- AlertSystem: triggers alert when temperature exceeds threshold

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from typing import Protocol
from dataclasses import dataclass, field


class TemperatureObserver(Protocol):
    """Protocol for temperature observers."""
    
    def on_temperature_change(self, celsius: float) -> None:
        """Called when temperature changes."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# YOUR TASK: Implement the classes below
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConsoleDisplay:
    """Displays temperature to console."""
    prefix: str = "Temperature"
    _last_displayed: float | None = field(default=None, init=False)
    
    def on_temperature_change(self, celsius: float) -> None:
        """Display the temperature."""
        # TODO: Print temperature and store in _last_displayed
        pass


@dataclass
class TemperatureLogger:
    """Records temperature history."""
    history: list[float] = field(default_factory=list)
    
    def on_temperature_change(self, celsius: float) -> None:
        """Record the temperature."""
        # TODO: Append to history
        pass
    
    def average(self) -> float:
        """Calculate average temperature."""
        # TODO: Return average of history
        pass
    
    def max_temperature(self) -> float:
        """Return maximum recorded temperature."""
        # TODO: Return max of history
        pass


@dataclass
class AlertSystem:
    """Triggers alerts when temperature exceeds threshold."""
    threshold: float
    alerts: list[str] = field(default_factory=list)
    
    def on_temperature_change(self, celsius: float) -> None:
        """Check threshold and possibly trigger alert."""
        # TODO: If celsius > threshold, append alert message
        pass


class TemperatureSensor:
    """Subject that notifies observers of temperature changes."""
    
    def __init__(self) -> None:
        self._observers: list[TemperatureObserver] = []
        self._temperature: float = 20.0  # Default 20°C
    
    def add_observer(self, observer: TemperatureObserver) -> None:
        """Subscribe an observer."""
        # TODO: Add to observers list
        pass
    
    def remove_observer(self, observer: TemperatureObserver) -> None:
        """Unsubscribe an observer."""
        # TODO: Remove from observers list
        pass
    
    def set_temperature(self, celsius: float) -> None:
        """Set temperature and notify observers."""
        # TODO: Update temperature and notify all observers
        pass
    
    @property
    def temperature(self) -> float:
        """Get current temperature."""
        return self._temperature


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_console_display() -> None:
    """Test console display observer."""
    display = ConsoleDisplay(prefix="Temp")
    sensor = TemperatureSensor()
    sensor.add_observer(display)
    sensor.set_temperature(25.5)
    assert display._last_displayed == 25.5


def test_temperature_logger() -> None:
    """Test temperature logger."""
    logger = TemperatureLogger()
    sensor = TemperatureSensor()
    sensor.add_observer(logger)
    
    sensor.set_temperature(20.0)
    sensor.set_temperature(25.0)
    sensor.set_temperature(30.0)
    
    assert len(logger.history) == 3
    assert logger.average() == 25.0
    assert logger.max_temperature() == 30.0


def test_alert_system() -> None:
    """Test alert system."""
    alert = AlertSystem(threshold=30.0)
    sensor = TemperatureSensor()
    sensor.add_observer(alert)
    
    sensor.set_temperature(25.0)  # No alert
    assert len(alert.alerts) == 0
    
    sensor.set_temperature(35.0)  # Should trigger alert
    assert len(alert.alerts) == 1


def test_multiple_observers() -> None:
    """Test multiple observers."""
    sensor = TemperatureSensor()
    logger = TemperatureLogger()
    alert = AlertSystem(threshold=50.0)
    
    sensor.add_observer(logger)
    sensor.add_observer(alert)
    
    sensor.set_temperature(40.0)
    sensor.set_temperature(60.0)
    
    assert len(logger.history) == 2
    assert len(alert.alerts) == 1  # Only 60.0 triggers


def test_remove_observer() -> None:
    """Test removing observer."""
    sensor = TemperatureSensor()
    logger = TemperatureLogger()
    
    sensor.add_observer(logger)
    sensor.set_temperature(20.0)
    sensor.remove_observer(logger)
    sensor.set_temperature(30.0)
    
    assert len(logger.history) == 1  # Only recorded before removal


if __name__ == "__main__":
    test_console_display()
    test_temperature_logger()
    test_alert_system()
    test_multiple_observers()
    test_remove_observer()
    print("All tests passed! ✓")
