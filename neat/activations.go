package neat

import (
	"fmt"
	"math"
)

// ActivationType defines the type for activation functions.
type ActivationType func(input float64, params ...float64) float64

// ActivationFunctions maps function names to the actual activation functions.
// This allows configuration to specify activations by name.
var ActivationFunctions = map[string]ActivationType{
	"sigmoid":  Sigmoid,
	"tanh":     Tanh,
	"relu":     ReLU,
	"identity": Identity,
	"clamped":  Clamped,
	"gaussian": Gaussian,
	"absolute": Absolute,
	"sine":     Sine,
	"cosine":   Cosine,
	// Add more functions as needed, matching neat-python's options
	"inv":    Inv,
	"log":    Log,
	"exp":    Exp,
	"abs":    Absolute, // Alias for absolute
	"hat":    Hat,
	"square": Square,
	"cube":   Cube,
	// Custom/advanced ones (like Softplus, ELU) could be added if required.
}

// GetActivation retrieves an activation function by name.
func GetActivation(name string) (ActivationType, error) {
	if fn, ok := ActivationFunctions[name]; ok {
		return fn, nil
	}
	return nil, fmt.Errorf("unknown activation function: %s", name)
}

// --- Standard Activation Function Implementations ---

// Sigmoid activation function.
func Sigmoid(x float64, params ...float64) float64 {
	// Use the logistic sigmoid formula: 1 / (1 + exp(-k * x))
	// Default k = 4.9 based on neat-python's config defaults (bias_mutate_power, response_mutate_power)
	// However, the activation function itself in neat-python doesn't seem to use node's response directly here.
	// It seems the response attribute scales the *input* to the activation function during network activation.
	// Let's stick to the standard sigmoid definition for the function itself.
	// The 'response' parameter from NodeGene will be applied *before* calling this.
	k := 4.9 // Default steepness, make configurable if needed?
	return 1.0 / (1.0 + math.Exp(-k*x))
}

// Tanh activation function.
func Tanh(x float64, params ...float64) float64 {
	return math.Tanh(x)
}

// ReLU (Rectified Linear Unit) activation function.
func ReLU(x float64, params ...float64) float64 {
	return math.Max(0, x)
}

// Identity activation function (linear).
func Identity(x float64, params ...float64) float64 {
	return x
}

// Clamped activation function (clamps output between -1 and 1).
func Clamped(x float64, params ...float64) float64 {
	return clamp(x, -1.0, 1.0) // Use the helper from math_util
}

// Gaussian activation function.
func Gaussian(x float64, params ...float64) float64 {
	return math.Exp(-x * x / 2.0)
}

// Absolute value activation function.
func Absolute(x float64, params ...float64) float64 {
	return math.Abs(x)
}

// Sine activation function.
func Sine(x float64, params ...float64) float64 {
	return math.Sin(x)
}

// Cosine activation function.
func Cosine(x float64, params ...float64) float64 {
	return math.Cos(x)
}

// Inv (Inverse) activation function.
func Inv(x float64, params ...float64) float64 {
	if x == 0.0 {
		// Handle division by zero - neat-python returns 0.0
		return 0.0
	}
	return 1.0 / x
}

// Log activation function (natural logarithm).
func Log(x float64, params ...float64) float64 {
	if x <= 0.0 {
		// Handle invalid input for log - neat-python uses log(max(eps, x))
		// Let's use a small epsilon or return 0
		epsilon := 1e-9
		return math.Log(math.Max(epsilon, x))
	}
	return math.Log(x)
}

// Exp activation function (e^x).
func Exp(x float64, params ...float64) float64 {
	// Clamp input to prevent overflow, similar to neat-python
	clampedX := clamp(x, -60.0, 60.0)
	return math.Exp(clampedX)
}

// Hat activation function (triangular pulse centered at 0).
func Hat(x float64, params ...float64) float64 {
	return math.Max(0.0, 1.0-math.Abs(x))
}

// Square activation function (x^2).
func Square(x float64, params ...float64) float64 {
	return x * x
}

// Cube activation function (x^3).
func Cube(x float64, params ...float64) float64 {
	return x * x * x
}
