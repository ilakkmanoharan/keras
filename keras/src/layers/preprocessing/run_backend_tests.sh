#!/bin/bash
# Backend Testing Script
# Runs Normalization tests across TensorFlow, JAX, and PyTorch backends

set -e

echo "================================================================================"
echo "NORMALIZATION LAYER - MULTI-BACKEND TESTING"
echo "================================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run tests
run_backend_tests() {
    local backend=$1
    echo ""
    echo "${YELLOW}Testing with ${backend} backend...${NC}"
    echo "--------------------------------------------------------------------------------"

    export KERAS_BACKEND=$backend

    if command -v pytest &> /dev/null; then
        if pytest normalization_test.py -xvs --tb=short 2>&1; then
            echo ""
            echo "${GREEN}✓ ${backend} backend tests PASSED${NC}"
            return 0
        else
            echo ""
            echo "${RED}✗ ${backend} backend tests FAILED${NC}"
            return 1
        fi
    else
        echo "${RED}pytest not found. Install with: pip install pytest${NC}"
        return 1
    fi
}

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running in Docker container"
    echo "Backend: ${KERAS_BACKEND}"
    run_backend_tests ${KERAS_BACKEND}
    exit $?
fi

# Run tests for all backends
BACKENDS=("tensorflow" "jax" "torch")
RESULTS=()

for backend in "${BACKENDS[@]}"; do
    if run_backend_tests $backend; then
        RESULTS+=("${GREEN}✓ ${backend}${NC}")
    else
        RESULTS+=("${RED}✗ ${backend}${NC}")
    fi
done

# Summary
echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
for result in "${RESULTS[@]}"; do
    echo -e "$result"
done
echo "================================================================================"
