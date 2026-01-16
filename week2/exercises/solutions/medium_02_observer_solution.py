#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Week 2 Practice: Medium Exercise 2 — Observer Pattern SOLUTION
═══════════════════════════════════════════════════════════════════════════════

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from typing import Protocol, Callable
from dataclasses import dataclass, field


class TemperatureObserver(Protocol):
    """Protocol for temperature observers."""
    
    def on_temperature_change(self, old_temp: float, new_temp: float) -> None:
        """Called when temperature changes."""
        ...


@dataclass
class DisplayObserver:
    """Observer that displays temperature to console."""
    name: str = "Display"
    
    def on_temperature_change(self, old_temp: float, new_temp: float) -> None:
        """Display the temperature change."""
        print(f"[{self.name}] Temperature: {old_temp:.1f}°C → {new_temp:.1f}°C")


@dataclass
class AlertObserver:
    """Observer that alerts when temperature exceeds threshold."""
    threshold: float = 30.0
    alerts: list[str] = field(default_factory=list)
    
    def on_temperature_change(self, old_temp: float, new_temp: float) -> None:
        """Check threshold and record alert if exceeded."""
        if new_temp > self.threshold >= old_temp:
            alert = f"ALERT: Temperature exceeded {self.threshold}°C (now {new_temp}°C)"
            self.alerts.append(alert)
            print(alert)
        elif new_temp <= self.threshold < old_temp:
            alert = f"NOTICE: Temperature back to normal (now {new_temp}°C)"
            self.alerts.append(alert)
            print(alert)


@dataclass
class LoggingObserver:
    """Observer that logs all temperature changes."""
    history: list[tuple[float, float]] = field(default_factory=list)
    
    def on_temperature_change(self, old_temp: float, new_temp: float) -> None:
        """Log the temperature change."""
        self.history.append((old_temp, new_temp))
    
    def get_average_change(self) -> float:
        """Calculate average temperature change magnitude."""
        if not self.history:
            return 0.0
        total = sum(abs(new - old) for old, new in self.history)
        return total / len(self.history)


class TemperatureSensor:
    """A temperature sensor that notifies observers of changes.
    
    This is the Subject in the Observer pattern.
    """
    
    def __init__(self, initial_temp: float = 20.0) -> None:
        """Initialise with starting temperature."""
        self._temperature = initial_temp
        self._observers: list[TemperatureObserver] = []
    
    @property
    def temperature(self) -> float:
        """Get current temperature."""
        return self._temperature
    
    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set temperature and notify observers."""
        if value != self._temperature:
            old_temp = self._temperature
            self._temperature = value
            self._notify_observers(old_temp, value)
    
    def subscribe(self, observer: TemperatureObserver) -> None:
        """Add an observer."""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def unsubscribe(self, observer: TemperatureObserver) -> None:
        """Remove an observer."""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def _notify_observers(self, old_temp: float, new_temp: float) -> None:
        """Notify all observers of a temperature change."""
        for observer in self._observers:
            observer.on_temperature_change(old_temp, new_temp)


# ═══════════════════════════════════════════════════════════════════════════════
# BONUS: Functional observer using callbacks
# ═══════════════════════════════════════════════════════════════════════════════

class FunctionalTemperatureSensor:
    """Temperature sensor using callback functions instead of protocols."""
    
    def __init__(self, initial_temp: float = 20.0) -> None:
        self._temperature = initial_temp
        self._callbacks: list[Callable[[float, float], None]] = []
    
    @property
    def temperature(self) -> float:
        return self._temperature
    
    @temperature.setter
    def temperature(self, value: float) -> None:
        if value != self._temperature:
            old_temp = self._temperature
            self._temperature = value
            for callback in self._callbacks:
                callback(old_temp, value)
    
    def on_change(self, callback: Callable[[float, float], None]) -> Callable[[], None]:
        """Register a callback and return an unsubscribe function."""
        self._callbacks.append(callback)
        
        def unsubscribe() -> None:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
        
        return unsubscribe


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_display_observer() -> None:
    """Test display observer receives updates."""
    sensor = TemperatureSensor(20.0)
    display = DisplayObserver("Test Display")
    
    sensor.subscribe(display)
    sensor.temperature = 25.0  # Should print update


def test_alert_observer() -> None:
    """Test alert observer threshold detection."""
    sensor = TemperatureSensor(25.0)
    alert = AlertObserver(threshold=30.0)
    
    sensor.subscribe(alert)
    
    sensor.temperature = 28.0  # No alert
    assert len(alert.alerts) == 0
    
    sensor.temperature = 32.0  # Should trigger alert
    assert len(alert.alerts) == 1
    assert "exceeded" in alert.alerts[0].lower()
    
    sensor.temperature = 29.0  # Should trigger back to normal
    assert len(alert.alerts) == 2


def test_logging_observer() -> None:
    """Test logging observer history tracking."""
    sensor = TemperatureSensor(20.0)
    logger = LoggingObserver()
    
    sensor.subscribe(logger)
    
    sensor.temperature = 25.0
    sensor.temperature = 22.0
    sensor.temperature = 30.0
    
    assert len(logger.history) == 3
    assert logger.history[0] == (20.0, 25.0)
    assert logger.history[1] == (25.0, 22.0)
    assert logger.history[2] == (22.0, 30.0)


def test_average_change() -> None:
    """Test logging observer average calculation."""
    sensor = TemperatureSensor(20.0)
    logger = LoggingObserver()
    
    sensor.subscribe(logger)
    
    sensor.temperature = 25.0  # +5
    sensor.temperature = 20.0  # -5
    sensor.temperature = 24.0  # +4
    
    # Average: (5 + 5 + 4) / 3 ≈ 4.67
    assert abs(logger.get_average_change() - 4.67) < 0.01


def test_multiple_observers() -> None:
    """Test multiple observers receive updates."""
    sensor = TemperatureSensor(20.0)
    
    display = DisplayObserver()
    alert = AlertObserver(threshold=25.0)
    logger = LoggingObserver()
    
    sensor.subscribe(display)
    sensor.subscribe(alert)
    sensor.subscribe(logger)
    
    sensor.temperature = 30.0
    
    assert len(logger.history) == 1
    assert len(alert.alerts) == 1


def test_unsubscribe() -> None:
    """Test unsubscribing an observer."""
    sensor = TemperatureSensor(20.0)
    logger = LoggingObserver()
    
    sensor.subscribe(logger)
    sensor.temperature = 25.0
    assert len(logger.history) == 1
    
    sensor.unsubscribe(logger)
    sensor.temperature = 30.0
    assert len(logger.history) == 1  # No new entry


def test_no_notification_on_same_value() -> None:
    """Test that setting same value doesn't notify."""
    sensor = TemperatureSensor(20.0)
    logger = LoggingObserver()
    
    sensor.subscribe(logger)
    sensor.temperature = 20.0  # Same value
    
    assert len(logger.history) == 0


def test_functional_observer() -> None:
    """Test functional callback-based observer."""
    sensor = FunctionalTemperatureSensor(20.0)
    changes: list[tuple[float, float]] = []
    
    def track_change(old: float, new: float) -> None:
        changes.append((old, new))
    
    unsubscribe = sensor.on_change(track_change)
    
    sensor.temperature = 25.0
    assert len(changes) == 1
    
    unsubscribe()
    sensor.temperature = 30.0
    assert len(changes) == 1  # No new entry after unsubscribe


if __name__ == "__main__":
    test_display_observer()
    test_alert_observer()
    test_logging_observer()
    test_average_change()
    test_multiple_observers()
    test_unsubscribe()
    test_no_notification_on_same_value()
    test_functional_observer()
    print("All tests passed! ✓")
