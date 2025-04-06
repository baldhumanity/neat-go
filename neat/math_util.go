package neat

import (
	"math"
	"math/rand"
	"sort"
	"strings"
)

// clamp restricts a value to a given range [minVal, maxVal].
func clamp(value, minVal, maxVal float64) float64 {
	return math.Max(minVal, math.Min(value, maxVal))
}

// parseBoolAttribute parses common string representations of booleans.
// Handles true/false, yes/no, on/off, 1/0, and random.
func parseBoolAttribute(valStr string) bool {
	valStr = strings.ToLower(strings.TrimSpace(valStr))
	if valStr == "true" || valStr == "yes" || valStr == "on" || valStr == "1" {
		return true
	}
	if valStr == "random" || valStr == "none" {
		return rand.Float64() < 0.5 // Randomize at initialization time if config says 'random'
	}
	return false
}

// --- Statistical Functions ---

// Mean calculates the average of a slice of float64 values.
func Mean(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

// Stdev calculates the standard deviation of a slice of float64 values.
func Stdev(values []float64) float64 {
	if len(values) < 2 {
		return 0.0 // Standard deviation is undefined for less than 2 values
	}
	mean := Mean(values)
	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	// Use sample standard deviation (divide by n-1)
	return math.Sqrt(variance / float64(len(values)-1))
}

// Sum calculates the sum of a slice of float64 values.
func Sum(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum
}

// MaxFloat calculates the maximum value in a slice of float64 values.
// Returns negative infinity if the slice is empty.
func MaxFloat(values []float64) float64 {
	if len(values) == 0 {
		return math.Inf(-1)
	}
	maxVal := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] > maxVal {
			maxVal = values[i]
		}
	}
	return maxVal
}

// MinFloat calculates the minimum value in a slice of float64 values.
// Returns positive infinity if the slice is empty.
func MinFloat(values []float64) float64 {
	if len(values) == 0 {
		return math.Inf(1)
	}
	minVal := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] < minVal {
			minVal = values[i]
		}
	}
	return minVal
}

// Median calculates the median of a slice of float64 values.
// Returns NaN if the slice is empty.
func Median(values []float64) float64 {
	n := len(values)
	if n == 0 {
		return math.NaN()
	}

	// Create a copy to avoid modifying the original slice
	sortedValues := make([]float64, n)
	copy(sortedValues, values)

	// Sort the slice
	// Use Float64s for sort.Sort
	sort.Float64s(sortedValues)

	mid := n / 2
	if n%2 == 1 {
		// Odd number of elements
		return sortedValues[mid]
	} else {
		// Even number of elements
		return (sortedValues[mid-1] + sortedValues[mid]) / 2.0
	}
}

// StatFunctions maps function names to the actual statistical functions.
// Used by Stagnation config.
var StatFunctions = map[string]func([]float64) float64{
	"mean":   Mean,
	"stdev":  Stdev,
	"sum":    Sum,
	"max":    MaxFloat,
	"min":    MinFloat,
	"median": Median,
}
