#!/usr/bin/env bash
# =============================================================================
# TAOCT4researchers — Environment Setup Script
# The Art of Computational Thinking for Researchers
# Version 5.0.0
# =============================================================================
#
# This script automates the setup of the development environment for the
# TAOCT4researchers curriculum.
#
# Usage:
#   chmod +x scripts/setup_environment.sh
#   ./scripts/setup_environment.sh [OPTIONS]
#
# Options:
#   --full          Install all dependencies for all 14 units
#   --dev           Install development dependencies (linting, testing)
#   --unit N        Install dependencies for specific unit (01-14)
#   --no-venv       Skip virtual environment creation
#   --help          Display this help message
#
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_MIN_VERSION="3.10"

# Colours for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No colour

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

print_header() {
    echo -e "${BLUE}"
    echo "============================================================================="
    echo " TAOCT4researchers — Environment Setup"
    echo " The Art of Computational Thinking for Researchers"
    echo "============================================================================="
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}→ $1${NC}"
}

show_help() {
    cat << EOF
TAOCT4researchers — Environment Setup Script

Usage:
  ./scripts/setup_environment.sh [OPTIONS]

Options:
  --full          Install all dependencies for all 14 units
  --dev           Install development dependencies (linting, testing)
  --unit N        Install dependencies for specific unit (01-14)
  --no-venv       Skip virtual environment creation
  --help          Display this help message

Examples:
  ./scripts/setup_environment.sh                  # Basic setup
  ./scripts/setup_environment.sh --full           # Full installation
  ./scripts/setup_environment.sh --unit 08        # Unit 08 only
  ./scripts/setup_environment.sh --full --dev     # Full + development

EOF
}

check_python_version() {
    print_info "Checking Python version..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    local version
    version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
        print_success "Python $version detected (minimum: $PYTHON_MIN_VERSION)"
    else
        print_error "Python $version detected, but $PYTHON_MIN_VERSION or higher is required"
        exit 1
    fi
}

check_pip() {
    print_info "Checking pip..."
    
    if ! python3 -m pip --version &> /dev/null; then
        print_error "pip is not available"
        exit 1
    fi
    
    print_success "pip is available"
}

create_venv() {
    if [ "$SKIP_VENV" = true ]; then
        print_warning "Skipping virtual environment creation (--no-venv)"
        return
    fi
    
    print_info "Creating virtual environment at $VENV_DIR..."
    
    if [ -d "$VENV_DIR" ]; then
        print_warning "Virtual environment already exists"
        read -p "Remove and recreate? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            print_info "Using existing virtual environment"
            return
        fi
    fi
    
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created"
}

activate_venv() {
    if [ "$SKIP_VENV" = true ]; then
        return
    fi
    
    print_info "Activating virtual environment..."
    
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    
    print_success "Virtual environment activated"
}

upgrade_pip() {
    print_info "Upgrading pip..."
    python3 -m pip install --upgrade pip --quiet
    print_success "pip upgraded"
}

install_base_dependencies() {
    print_info "Installing base dependencies..."
    
    if [ -f "$REPO_ROOT/requirements.txt" ]; then
        python3 -m pip install -r "$REPO_ROOT/requirements.txt" --quiet
        print_success "Base dependencies installed"
    else
        print_warning "requirements.txt not found, skipping base dependencies"
    fi
}

install_dev_dependencies() {
    print_info "Installing development dependencies..."
    
    if [ -f "$REPO_ROOT/requirements-dev.txt" ]; then
        python3 -m pip install -r "$REPO_ROOT/requirements-dev.txt" --quiet
        print_success "Development dependencies installed"
    else
        # Install common dev dependencies
        python3 -m pip install ruff mypy pytest pytest-cov pre-commit --quiet
        print_success "Development dependencies installed (default set)"
    fi
}

install_unit_dependencies() {
    local unit_num="$1"
    local unit_dir
    
    # Find unit directory
    unit_dir=$(find "$REPO_ROOT" -maxdepth 1 -type d -name "${unit_num}UNIT*" | head -1)
    
    if [ -z "$unit_dir" ]; then
        print_error "Unit $unit_num not found"
        return 1
    fi
    
    print_info "Installing dependencies for $(basename "$unit_dir")..."
    
    if [ -f "$unit_dir/requirements.txt" ]; then
        python3 -m pip install -r "$unit_dir/requirements.txt" --quiet
        print_success "Unit $unit_num dependencies installed"
    else
        print_warning "No requirements.txt for Unit $unit_num"
    fi
}

install_all_unit_dependencies() {
    print_info "Installing dependencies for all 14 units..."
    
    for unit_num in 01 02 03 04 05 06 07 08 09 10 11 12 13 14; do
        install_unit_dependencies "$unit_num" || true
    done
    
    print_success "All unit dependencies installed"
}

verify_installation() {
    print_info "Verifying installation..."
    
    local errors=0
    
    # Check core packages
    for pkg in numpy pandas matplotlib; do
        if python3 -c "import $pkg" 2>/dev/null; then
            print_success "$pkg imported successfully"
        else
            print_error "Failed to import $pkg"
            ((errors++))
        fi
    done
    
    if [ $errors -eq 0 ]; then
        print_success "Installation verified"
    else
        print_warning "Some packages failed to import"
    fi
}

print_summary() {
    echo ""
    echo -e "${GREEN}=============================================================================${NC}"
    echo -e "${GREEN} Setup Complete!${NC}"
    echo -e "${GREEN}=============================================================================${NC}"
    echo ""
    
    if [ "$SKIP_VENV" = false ]; then
        echo "To activate the virtual environment:"
        echo ""
        echo "  source $VENV_DIR/bin/activate"
        echo ""
    fi
    
    echo "To verify the installation:"
    echo ""
    echo "  make check"
    echo ""
    echo "To run the test suite:"
    echo ""
    echo "  make test"
    echo ""
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

main() {
    # Parse arguments
    INSTALL_FULL=false
    INSTALL_DEV=false
    INSTALL_UNIT=""
    SKIP_VENV=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --full)
                INSTALL_FULL=true
                shift
                ;;
            --dev)
                INSTALL_DEV=true
                shift
                ;;
            --unit)
                INSTALL_UNIT="$2"
                shift 2
                ;;
            --no-venv)
                SKIP_VENV=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Execute setup
    print_header
    
    check_python_version
    check_pip
    create_venv
    activate_venv
    upgrade_pip
    
    # Install dependencies based on options
    if [ "$INSTALL_FULL" = true ]; then
        install_base_dependencies
        install_all_unit_dependencies
    elif [ -n "$INSTALL_UNIT" ]; then
        install_base_dependencies
        install_unit_dependencies "$INSTALL_UNIT"
    else
        install_base_dependencies
    fi
    
    if [ "$INSTALL_DEV" = true ]; then
        install_dev_dependencies
    fi
    
    verify_installation
    print_summary
}

main "$@"
