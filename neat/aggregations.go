package neat

import (
	"fmt"
	"math"
)

// AggregationType defines the type for aggregation functions.
type AggregationType func(inputs []float64) float64

// AggregationFunctions maps function names to the actual aggregation functions.
var AggregationFunctions = map[string]AggregationType{
	"sum":     AggregateSum,
	"product": AggregateProduct,
	"min":     AggregateMin,
	"max":     AggregateMax,
	"mean":    AggregateMean,
	"median":  AggregateMedian,
	// Add aliases or other functions if needed
	"average": AggregateMean, // Alias for mean
}

// GetAggregation retrieves an aggregation function by name.
func GetAggregation(name string) (AggregationType, error) {
	if fn, ok := AggregationFunctions[name]; ok {
		return fn, nil
	}
	return nil, fmt.Errorf("unknown aggregation function: %s", name)
}

// --- Standard Aggregation Function Implementations ---

// AggregateSum calculates the sum of the inputs.
func AggregateSum(inputs []float64) float64 {
	// Use the Sum function from math_util
	return Sum(inputs)
}

// AggregateProduct calculates the product of the inputs.
func AggregateProduct(inputs []float64) float64 {
	if len(inputs) == 0 {
		return 0.0 // Or 1.0? Python returns 1.0
	}
	product := 1.0
	for _, v := range inputs {
		product *= v
	}
	return product
}

// AggregateMin finds the minimum value among the inputs.
func AggregateMin(inputs []float64) float64 {
	// Use the MinFloat function from math_util
	return MinFloat(inputs)
}

// AggregateMax finds the maximum value among the inputs.
func AggregateMax(inputs []float64) float64 {
	// Use the MaxFloat function from math_util
	return MaxFloat(inputs)
}

// AggregateMean calculates the average of the inputs.
func AggregateMean(inputs []float64) float64 {
	// Use the Mean function from math_util
	return Mean(inputs)
}

// AggregateMedian calculates the median of the inputs.
func AggregateMedian(inputs []float64) float64 {
	// Use the Median function from math_util
	return Median(inputs)
}

// --- Specific Aggregations from neat-python (might be less common) ---

// NOTE: The `maxabs`, `median`, `meanabs` functions in neat-python's aggregations.py
// seem slightly different from the standard stats functions. Re-implement if exact
// behavior is crucial. For now, the standard Mean/Median/Max cover the main cases.

// Example: MaxAbs (if needed)
func AggregateMaxAbs(inputs []float64) float64 {
	if len(inputs) == 0 {
		return 0.0
	}
	maxAbsVal := math.Abs(inputs[0])
	for i := 1; i < len(inputs); i++ {
		absVal := math.Abs(inputs[i])
		if absVal > maxAbsVal {
			maxAbsVal = absVal
		}
	}
	return maxAbsVal
}
