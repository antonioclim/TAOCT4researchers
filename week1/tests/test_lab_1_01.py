"""
Tests for Week 1, Lab 1: Turing Machine Simulator.

This module contains pytest tests for the Turing machine implementation.
Tests cover machine creation, simulation and example machines.

Run with: pytest tests/test_lab_1_01.py -v
"""

import pytest


class TestTuringMachineCreation:
    """Tests for Turing machine creation and validation."""
    
    def test_increment_machine_creation(self, increment_machine):
        """Test that the increment machine is created successfully."""
        assert increment_machine is not None
        assert increment_machine.initial_state == "q0"
        assert increment_machine.accept_state == "accept"
    
    def test_palindrome_machine_creation(self, palindrome_machine):
        """Test that the palindrome machine is created successfully."""
        assert palindrome_machine is not None
        assert palindrome_machine.initial_state == "q0"
        assert palindrome_machine.reject_state == "reject"
    
    def test_machine_has_transitions(self, increment_machine):
        """Test that machines have defined transitions."""
        assert len(increment_machine.transitions) > 0


class TestUnaryIncrement:
    """Tests for the unary increment Turing machine."""
    
    @pytest.mark.parametrize("input_str,expected_len", [
        ("", 1),      # 0 + 1 = 1
        ("1", 2),     # 1 + 1 = 2
        ("11", 3),    # 2 + 1 = 3
        ("111", 4),   # 3 + 1 = 4
        ("11111", 6), # 5 + 1 = 6
    ])
    def test_increment_output_length(
        self, increment_machine, simulator, input_str, expected_len
    ):
        """Test that increment produces correct output length."""
        sim = simulator(increment_machine)
        sim.load(input_str if input_str else "□")
        sim.run()
        output = sim.get_output()
        assert len(output) == expected_len
    
    def test_increment_accepts(self, increment_machine, simulator):
        """Test that increment machine always accepts."""
        sim = simulator(increment_machine)
        sim.load("111")
        result = sim.run()
        assert result is True
        assert sim.accepted is True


class TestPalindromeChecker:
    """Tests for the palindrome checker Turing machine."""
    
    @pytest.mark.parametrize("input_str,expected", [
        ("", True),
        ("0", True),
        ("1", True),
        ("00", True),
        ("11", True),
        ("010", True),
        ("0110", True),
        ("10101", True),
        ("110011", True),
    ])
    def test_palindrome_accepts(
        self, palindrome_machine, simulator, input_str, expected
    ):
        """Test that palindromes are accepted."""
        sim = simulator(palindrome_machine)
        sim.load(input_str if input_str else "□")
        result = sim.run()
        assert result is expected
    
    @pytest.mark.parametrize("input_str,expected", [
        ("01", False),
        ("10", False),
        ("0101", False),
        ("100", False),
        ("011", False),
    ])
    def test_non_palindrome_rejects(
        self, palindrome_machine, simulator, input_str, expected
    ):
        """Test that non-palindromes are rejected."""
        sim = simulator(palindrome_machine)
        sim.load(input_str)
        result = sim.run()
        assert result is expected
        assert sim.rejected is True


class TestBinarySuccessor:
    """Tests for the binary successor Turing machine."""
    
    @pytest.mark.parametrize("input_str,expected", [
        ("0", "1"),
        ("1", "10"),
        ("10", "11"),
        ("11", "100"),
        ("101", "110"),
        ("111", "1000"),
        ("1011", "1100"),
    ])
    def test_successor_output(
        self, successor_machine, simulator, input_str, expected
    ):
        """Test that successor produces correct binary output."""
        sim = simulator(successor_machine)
        sim.load(input_str)
        sim.run()
        output = sim.get_output()
        assert output == expected


class TestSimulatorBehaviour:
    """Tests for the TuringSimulator class behaviour."""
    
    def test_simulator_requires_load(self, increment_machine, simulator):
        """Test that simulator raises error if not loaded."""
        sim = simulator(increment_machine)
        with pytest.raises(RuntimeError, match="No input loaded"):
            sim.step()
    
    def test_simulator_tracks_steps(self, increment_machine, simulator):
        """Test that simulator tracks step count."""
        sim = simulator(increment_machine)
        sim.load("11")
        sim.run()
        assert sim.config.step_count > 0
    
    def test_simulator_maintains_history(self, increment_machine, simulator):
        """Test that simulator maintains execution history."""
        sim = simulator(increment_machine)
        sim.load("1")
        sim.run()
        assert len(sim.history) > 0
    
    def test_max_steps_limit(self, increment_machine, simulator):
        """Test that max_steps parameter limits execution."""
        sim = simulator(increment_machine)
        sim.load("1" * 1000)  # Long input
        sim.run(max_steps=10)
        assert sim.config.step_count <= 10


class TestConfiguration:
    """Tests for the Configuration class."""
    
    def test_configuration_read(self):
        """Test reading from configuration tape."""
        from lab_1_01_turing_machine import Configuration
        
        config = Configuration(
            tape={0: "1", 1: "0", 2: "1"},
            head_position=1,
            current_state="q0"
        )
        assert config.read() == "0"
    
    def test_configuration_write(self):
        """Test writing to configuration tape."""
        from lab_1_01_turing_machine import Configuration
        
        config = Configuration(
            tape={0: "1"},
            head_position=0,
            current_state="q0"
        )
        config.write("0")
        assert config.tape[0] == "0"
    
    def test_configuration_move(self):
        """Test moving the head."""
        from lab_1_01_turing_machine import Configuration, Direction
        
        config = Configuration(head_position=5, current_state="q0")
        
        config.move(Direction.RIGHT)
        assert config.head_position == 6
        
        config.move(Direction.LEFT)
        assert config.head_position == 5
        
        config.move(Direction.STAY)
        assert config.head_position == 5


class TestTransition:
    """Tests for the Transition class."""
    
    def test_transition_creation(self):
        """Test creating a transition."""
        from lab_1_01_turing_machine import Transition, Direction
        
        t = Transition("q1", "X", Direction.RIGHT)
        assert t.next_state == "q1"
        assert t.write_symbol == "X"
        assert t.direction == Direction.RIGHT
    
    def test_transition_is_frozen(self):
        """Test that transitions are immutable."""
        from lab_1_01_turing_machine import Transition, Direction
        
        t = Transition("q1", "X", Direction.RIGHT)
        with pytest.raises(AttributeError):
            t.next_state = "q2"


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_input_increment(self, increment_machine, simulator):
        """Test increment with conceptually empty input."""
        sim = simulator(increment_machine)
        # Empty string represents 0 in unary
        sim.load("□")
        sim.run()
        assert sim.accepted
    
    def test_long_input_palindrome(self, palindrome_machine, simulator):
        """Test palindrome checker with longer input."""
        sim = simulator(palindrome_machine)
        sim.load("1010101010101")
        result = sim.run()
        assert result is True  # This is a palindrome
    
    @pytest.mark.slow
    def test_very_long_computation(self, palindrome_machine, simulator):
        """Test that long computations complete within reasonable time."""
        sim = simulator(palindrome_machine)
        sim.load("1" * 100)
        result = sim.run(max_steps=100000)
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
