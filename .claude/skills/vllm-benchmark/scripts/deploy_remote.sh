#!/bin/bash
# Remote deployment and verification script for p5e-cmh
#
# Usage:
#   ./deploy_remote.sh <SYMBOL_NAME> <SKIP_REMOTE>
#
# Arguments:
#   SYMBOL_NAME: Kernel name (e.g., "fused_moe_kernel")
#   SKIP_REMOTE: "true" to skip remote deployment, "false" to deploy
#
# Examples:
#   ./deploy_remote.sh fused_moe_kernel false
#   ./deploy_remote.sh paged_attention true

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SYMBOL_NAME="${1:-}"
SKIP_REMOTE="${2:-false}"
REMOTE_HOST="p5e-cmh"
REMOTE_DIR="~/bundled_benchmarks"
LOCAL_DIR="bundled_benchmarks"
ARTIFACTS_DIR="artifacts"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Validation
# ============================================================================

if [ -z "$SYMBOL_NAME" ]; then
    log_error "SYMBOL_NAME not provided"
    echo "Usage: $0 <SYMBOL_NAME> [SKIP_REMOTE]"
    exit 1
fi

if [ "$SKIP_REMOTE" = "true" ]; then
    log_warning "Skipping remote verification (--skip-remote flag set)"
    log_info "⏭ Stage 2.3 skipped"
    exit 0
fi

# Check if bundle files exist locally
if [ ! -f "${LOCAL_DIR}/${SYMBOL_NAME}.py" ]; then
    log_error "Bundle file not found: ${LOCAL_DIR}/${SYMBOL_NAME}.py"
    exit 1
fi

if [ ! -f "${LOCAL_DIR}/${SYMBOL_NAME}_bench.py" ]; then
    log_error "Benchmark file not found: ${LOCAL_DIR}/${SYMBOL_NAME}_bench.py"
    exit 1
fi

# ============================================================================
# Step 1: Check Remote Host Connectivity
# ============================================================================

log_info "Checking connectivity to ${REMOTE_HOST}..."

if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "${REMOTE_HOST}" "echo 'Connected' > /dev/null 2>&1"; then
    log_error "Cannot reach ${REMOTE_HOST}"
    log_warning "Possible solutions:"
    echo "  1. Check VPN connection"
    echo "  2. Verify SSH key authentication"
    echo "  3. Use --skip-remote flag to skip remote verification"
    exit 1
fi

log_success "Connected to ${REMOTE_HOST}"

# ============================================================================
# Step 2: Create Remote Directory
# ============================================================================

log_info "Creating remote directory: ${REMOTE_DIR}"

ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}" || {
    log_error "Failed to create remote directory"
    exit 1
}

log_success "Remote directory ready"

# ============================================================================
# Step 3: Upload Bundle Files
# ============================================================================

log_info "Uploading bundle files to ${REMOTE_HOST}:${REMOTE_DIR}/"

scp -q \
    "${LOCAL_DIR}/${SYMBOL_NAME}.py" \
    "${LOCAL_DIR}/${SYMBOL_NAME}_bench.py" \
    "${REMOTE_HOST}:${REMOTE_DIR}/" || {
    log_error "Failed to upload files"
    exit 1
}

log_success "Files uploaded successfully"

# ============================================================================
# Step 4: Verify Upload
# ============================================================================

log_info "Verifying uploaded files..."

ssh "${REMOTE_HOST}" "ls -lh ${REMOTE_DIR}/${SYMBOL_NAME}*" || {
    log_error "Files not found on remote host"
    exit 1
}

log_success "Upload verified"

# ============================================================================
# Step 5: Run Quick Validation
# ============================================================================

log_info "Running quick validation on ${REMOTE_HOST}..."

# Create artifacts directory if not exists
mkdir -p "${ARTIFACTS_DIR}"

# Run remote benchmark with validation
ssh "${REMOTE_HOST}" \
    "source ~/miniconda3/bin/activate && cd ${REMOTE_DIR} && python ${SYMBOL_NAME}_bench.py --quick --validate 2>&1" \
    | tee "${ARTIFACTS_DIR}/${SYMBOL_NAME}_remote.log" || {
    log_error "Remote validation failed"
    log_info "Check ${ARTIFACTS_DIR}/${SYMBOL_NAME}_remote.log for details"
    exit 1
}

# ============================================================================
# Step 6: Analyze Results
# ============================================================================

log_info "Analyzing results..."

# Check for success indicators in log
if grep -q "✓.*Correctness validated" "${ARTIFACTS_DIR}/${SYMBOL_NAME}_remote.log"; then
    log_success "Correctness validation passed"
else
    log_warning "Correctness validation status unclear"
fi

if grep -q "Latency:" "${ARTIFACTS_DIR}/${SYMBOL_NAME}_remote.log"; then
    log_success "Latency measurements recorded"
    # Extract and display latency numbers
    grep "Latency:" "${ARTIFACTS_DIR}/${SYMBOL_NAME}_remote.log" | head -3
else
    log_warning "No latency measurements found"
fi

# Check for errors
if grep -iq "error\|exception\|traceback" "${ARTIFACTS_DIR}/${SYMBOL_NAME}_remote.log"; then
    log_error "Errors detected in remote execution"
    log_info "Check ${ARTIFACTS_DIR}/${SYMBOL_NAME}_remote.log for details"
    exit 1
fi

# ============================================================================
# Success
# ============================================================================

echo ""
log_success "Remote verification complete!"
echo ""
echo "Summary:"
echo "  - Bundle: ${SYMBOL_NAME}.py"
echo "  - Benchmark: ${SYMBOL_NAME}_bench.py"
echo "  - Remote host: ${REMOTE_HOST}"
echo "  - Log: ${ARTIFACTS_DIR}/${SYMBOL_NAME}_remote.log"
echo ""
echo "To run full benchmark suite remotely:"
echo "  ssh ${REMOTE_HOST} \"cd ${REMOTE_DIR} && python ${SYMBOL_NAME}_bench.py --sweep full --output results.json\""
echo ""

exit 0
