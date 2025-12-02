#!/bin/bash
# Generate LaTeX tables for a dataset across metrics and sensitive attributes
# Usage: ./generate_tables.sh <dataset> [options]
#
# Examples:
#   ./generate_tables.sh heart                    # All attributes, all metrics
#   ./generate_tables.sh heart --attrs sex age    # Specific attributes, all metrics
#   ./generate_tables.sh heart --metrics loss acc # All attributes, specific metrics
#   ./generate_tables.sh heart --attrs sex_age --metrics loss acc
#   ./generate_tables.sh adult --attrs sex age sex_age

if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset> [options]"
    echo ""
    echo "Options:"
    echo "  --attrs <attr1> [attr2] ...    Sensitive attributes (default: sex age sex_age)"
    echo "  --metrics <metric1> [metric2]  Metrics to generate (default: all)"
    echo ""
    echo "Examples:"
    echo "  $0 heart                                    # All attributes, all metrics"
    echo "  $0 heart --attrs sex age                    # Specific attributes"
    echo "  $0 heart --metrics loss acc                 # Specific metrics"
    echo "  $0 adult --attrs sex_age --metrics all"
    echo ""
    echo "Available datasets: heart, adult, acs_pums"
    echo "Available sensitive attributes: sex, age, sex_age"
    echo "Available metrics: loss, acc, auroc, all"
    exit 1
fi

DATASET=$1
shift

# Default values
SENSITIVE_ATTRS=("sex" "age" "sex_age")
METRICS="all"

# Parse options
while [ $# -gt 0 ]; do
    case "$1" in
        --attrs)
            shift
            SENSITIVE_ATTRS=()
            while [ $# -gt 0 ] && [[ ! "$1" =~ ^-- ]]; do
                SENSITIVE_ATTRS+=("$1")
                shift
            done
            ;;
        --metrics)
            shift
            METRICS=""
            while [ $# -gt 0 ] && [[ ! "$1" =~ ^-- ]]; do
                if [ -z "$METRICS" ]; then
                    METRICS="$1"
                else
                    METRICS="$METRICS $1"
                fi
                shift
            done
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Generating tables for dataset: $DATASET"
echo "Sensitive attributes: ${SENSITIVE_ATTRS[@]}"
echo "Metrics: $METRICS"
echo ""

for attr in "${SENSITIVE_ATTRS[@]}"; do
    echo "Generating tables for sensitive attribute: $attr"
    python aggregate_results.py --dataset "$DATASET" --sensitive "$attr" --metrics $METRICS
    if [ $? -ne 0 ]; then
        echo "Error generating tables for $attr"
        exit 1
    fi
done

echo ""
echo "✓ All tables generated successfully"
echo "Output location: results/tables/"
echo ""
echo "Generated files:"
ls -lh results/tables/${DATASET}_*.tex 2>/dev/null | awk '{print "  " $9}' | sort
