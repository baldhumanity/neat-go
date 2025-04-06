package neat

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
)

// GeneType defines the type of gene (Node or Connection)
type GeneType int

const (
	NodeGeneType GeneType = iota
	ConnectionGeneType
)

// BaseGene defines common properties and methods for genes.
// In Go, we'll use composition or interfaces rather than direct inheritance.
// For now, we'll put common fields/methods directly in NodeGene/ConnectionGene
// or use helper functions.

// --------------------------- NodeGene ---------------------------

// NodeGene represents a node (neuron) in the neural network genome.
type NodeGene struct {
	Key         int // Unique identifier for this node gene (negative for inputs, >=0 for outputs/hidden)
	Bias        float64
	Response    float64
	Activation  string // Name of the activation function
	Aggregation string // Name of the aggregation function
}

// NewNodeGene creates a new NodeGene with attributes initialized according to the config.
func NewNodeGene(key int, config *GenomeConfig) *NodeGene {
	ng := &NodeGene{
		Key:         key,
		Activation:  initStringAttribute(config.ActivationDefault, config.ActivationOptions),
		Aggregation: initStringAttribute(config.AggregationDefault, config.AggregationOptions),
	}
	ng.Bias = initFloatAttribute(config.BiasInitMean, config.BiasInitStdev, config.BiasInitType, config.BiasMinValue, config.BiasMaxValue)
	ng.Response = initFloatAttribute(config.ResponseInitMean, config.ResponseInitStdev, config.ResponseInitType, config.ResponseMinValue, config.ResponseMaxValue)
	return ng
}

// String returns a string representation of the NodeGene.
func (ng *NodeGene) String() string {
	return fmt.Sprintf("NodeGene(Key: %d, Bias: %.3f, Response: %.3f, Activation: %s, Aggregation: %s)",
		ng.Key, ng.Bias, ng.Response, ng.Activation, ng.Aggregation)
}

// Copy creates a deep copy of the NodeGene.
func (ng *NodeGene) Copy() *NodeGene {
	return &NodeGene{
		Key:         ng.Key,
		Bias:        ng.Bias,
		Response:    ng.Response,
		Activation:  ng.Activation,
		Aggregation: ng.Aggregation,
	}
}

// Mutate adjusts the attributes of the NodeGene based on mutation rates in the config.
func (ng *NodeGene) Mutate(config *GenomeConfig) {
	ng.Bias = mutateFloatAttribute(ng.Bias, config.BiasMutateRate, config.BiasReplaceRate, config.BiasMutatePower, config.BiasInitMean, config.BiasInitStdev, config.BiasInitType, config.BiasMinValue, config.BiasMaxValue)
	ng.Response = mutateFloatAttribute(ng.Response, config.ResponseMutateRate, config.ResponseReplaceRate, config.ResponseMutatePower, config.ResponseInitMean, config.ResponseInitStdev, config.ResponseInitType, config.ResponseMinValue, config.ResponseMaxValue)
	ng.Activation = mutateStringAttribute(ng.Activation, config.ActivationMutateRate, config.ActivationOptions)
	ng.Aggregation = mutateStringAttribute(ng.Aggregation, config.AggregationMutateRate, config.AggregationOptions)
}

// Distance calculates the genetic distance between two NodeGenes based on their attributes.
func (ng *NodeGene) Distance(other *NodeGene, config *GenomeConfig) float64 {
	d := math.Abs(ng.Bias-other.Bias) + math.Abs(ng.Response-other.Response)
	if ng.Activation != other.Activation {
		d += 1.0
	}
	if ng.Aggregation != other.Aggregation {
		d += 1.0
	}
	return d * config.CompatibilityWeightCoefficient // Using the same coefficient as weights for now
}

// Crossover creates a new NodeGene by randomly inheriting attributes from two parent NodeGenes.
func (ng *NodeGene) Crossover(other *NodeGene) *NodeGene {
	// Assume ng is the primary parent (e.g., the more fit one if applicable)
	child := ng.Copy() // Start with a copy of the primary parent

	if rand.Float64() < 0.5 {
		child.Bias = other.Bias
	}
	if rand.Float64() < 0.5 {
		child.Response = other.Response
	}
	if rand.Float64() < 0.5 {
		child.Activation = other.Activation
	}
	if rand.Float64() < 0.5 {
		child.Aggregation = other.Aggregation
	}

	return child
}

// --------------------------- ConnectionGene ---------------------------

// ConnectionGene represents a connection between two nodes in the genome.
// The Key is a tuple (in Python), represented here as ConnectionKey struct.
type ConnectionGene struct {
	Key     ConnectionKey // Represents the (in_node_id, out_node_id) tuple
	Weight  float64
	Enabled bool
	// InnovationNumber is handled implicitly by using the Key (ConnectionKey) as the map key in Genome.
}

// ConnectionKey uniquely identifies a connection gene (innovation).
type ConnectionKey struct {
	InNodeID  int
	OutNodeID int
}

// NewConnectionGene creates a new ConnectionGene with attributes initialized according to the config.
func NewConnectionGene(key ConnectionKey, config *GenomeConfig) *ConnectionGene {
	cg := &ConnectionGene{
		Key:     key,
		Enabled: initBoolAttribute(config.EnabledDefault),
	}
	cg.Weight = initFloatAttribute(config.WeightInitMean, config.WeightInitStdev, config.WeightInitType, config.WeightMinValue, config.WeightMaxValue)
	return cg
}

// String returns a string representation of the ConnectionGene.
func (cg *ConnectionGene) String() string {
	return fmt.Sprintf("ConnGene(Key: %d->%d, Weight: %.3f, Enabled: %t)",
		cg.Key.InNodeID, cg.Key.OutNodeID, cg.Weight, cg.Enabled)
}

// Copy creates a deep copy of the ConnectionGene.
func (cg *ConnectionGene) Copy() *ConnectionGene {
	return &ConnectionGene{
		Key:     cg.Key,
		Weight:  cg.Weight,
		Enabled: cg.Enabled,
	}
}

// Mutate adjusts the attributes of the ConnectionGene based on mutation rates in the config.
// It now accepts the genome to check for cycles when enabling connections in feedforward mode.
func (cg *ConnectionGene) Mutate(genome *Genome, config *GenomeConfig) {
	cg.Weight = mutateFloatAttribute(cg.Weight, config.WeightMutateRate, config.WeightReplaceRate, config.WeightMutatePower, config.WeightInitMean, config.WeightInitStdev, config.WeightInitType, config.WeightMinValue, config.WeightMaxValue)
	// Pass necessary context to mutateBoolAttribute for potential cycle check
	cg.Enabled = mutateBoolAttribute(cg.Enabled, config.EnabledMutateRate, config.EnabledRateToTrueAdd, config.EnabledRateToFalseAdd, genome, cg)
}

// Distance calculates the genetic distance between two ConnectionGenes.
func (cg *ConnectionGene) Distance(other *ConnectionGene, config *GenomeConfig) float64 {
	d := math.Abs(cg.Weight - other.Weight)
	if cg.Enabled != other.Enabled {
		d += 1.0
	}
	return d * config.CompatibilityWeightCoefficient
}

// Crossover creates a new ConnectionGene by randomly inheriting attributes from two parent ConnectionGenes.
func (cg *ConnectionGene) Crossover(other *ConnectionGene) *ConnectionGene {
	// Assume cg is the primary parent
	child := cg.Copy()

	if rand.Float64() < 0.5 {
		child.Weight = other.Weight
	}
	// For enabled gene, prefer enabled if either parent has it enabled (as per original NEAT paper, C5, p116)
	// However, neat-python just randomly chooses one parent's value. We'll follow neat-python here.
	if rand.Float64() < 0.5 {
		child.Enabled = other.Enabled
	}

	return child
}

// --------------------------- Attribute Helpers ---------------------------
// These functions mimic the behavior of the Python Attribute classes for initialization and mutation.

func initFloatAttribute(mean, stdev float64, initType string, minVal, maxVal float64) float64 {
	var val float64
	switch strings.ToLower(initType) {
	case "gaussian", "normal", "": // Default to gaussian
		val = rand.NormFloat64()*stdev + mean
	case "uniform":
		// Estimate uniform range from mean/stdev assuming approx 2 std devs covers most range
		rangeMin := math.Max(minVal, mean-(2*stdev))
		rangeMax := math.Min(maxVal, mean+(2*stdev))
		if rangeMax < rangeMin {
			rangeMax = rangeMin
		} // Prevent issues if stdev is huge
		val = rand.Float64()*(rangeMax-rangeMin) + rangeMin
	default:
		// Consider returning an error or panicking for unknown type
		fmt.Printf("Warning: Unknown float init_type '%s', using gaussian\n", initType)
		val = rand.NormFloat64()*stdev + mean
	}
	return clamp(val, minVal, maxVal)
}

func mutateFloatAttribute(value, mutateRate, replaceRate, mutatePower, initMean, initStdev float64, initType string, minVal, maxVal float64) float64 {
	r := rand.Float64()
	if r < mutateRate {
		// Perturb value
		perturbation := rand.NormFloat64() * mutatePower
		value += perturbation
		return clamp(value, minVal, maxVal)
	}
	if r < mutateRate+replaceRate {
		// Replace value with a new one
		return initFloatAttribute(initMean, initStdev, initType, minVal, maxVal)
	}
	// No mutation
	return value
}

func initBoolAttribute(defaultValStr string) bool {
	return parseBoolAttribute(defaultValStr) // Use helper from config.go (assuming it's accessible or moved)
}

func mutateBoolAttribute(value bool, mutateRate, rateToTrueAdd, rateToFalseAdd float64, genome *Genome, cg *ConnectionGene) bool {
	effectiveMutateRate := mutateRate
	if value { // Currently true, might mutate to false
		effectiveMutateRate += rateToFalseAdd
	} else { // Currently false, might mutate to true
		effectiveMutateRate += rateToTrueAdd
	}

	if effectiveMutateRate > 0 && rand.Float64() < effectiveMutateRate {
		// Instead of just flipping, decide the new state (true or false).
		newState := rand.Float64() < 0.5

		// Cycle Check: Only allow enabling if it doesn't create a cycle in feedforward mode
		if !value && newState && genome.Config.FeedForward {
			// Trying to enable the connection (value=false, newState=true)
			if createsCycle(genome, cg.Key.InNodeID, cg.Key.OutNodeID) {
				return false // Prevent enabling if it creates a cycle
			}
		}
		return newState // Return the randomly chosen state (true/false) if no cycle issue
	}
	// No mutation
	return value
}

func initStringAttribute(defaultVal string, options []string) string {
	if len(options) == 0 {
		// This should ideally be caught during config validation
		fmt.Println("Warning: Attempting to initialize string attribute with no options.")
		return ""
	}
	defaultValLower := strings.ToLower(defaultVal)
	if defaultValLower == "random" || defaultValLower == "none" || defaultValLower == "" {
		return options[rand.Intn(len(options))]
	}
	// Check if the default value is actually in the options list
	for _, opt := range options {
		if opt == defaultVal {
			return defaultVal
		}
	}
	// If default is not 'random'/'none' and not in options, issue warning and pick random
	fmt.Printf("Warning: Default string value '%s' not in options %v. Choosing random.\n", defaultVal, options)
	return options[rand.Intn(len(options))]
}

func mutateStringAttribute(value string, mutateRate float64, options []string) string {
	if len(options) <= 1 { // Can't mutate if only one or zero options
		return value
	}
	if mutateRate > 0 && rand.Float64() < mutateRate {
		// Choose a random option *different* from the current value if possible
		var newValue string
		for {
			newValue = options[rand.Intn(len(options))]
			if newValue != value {
				break
			}
			// If by chance all options are the same, break the loop
			allSame := true
			for _, opt := range options {
				if opt != value {
					allSame = false
					break
				}
			}
			if allSame {
				break
			}
		}
		return newValue
	}
	// No mutation
	return value
}

// Note: The `parseBoolAttribute` and `clamp` helper functions are duplicated from config.go.
// Ideally, these would be moved to a shared utility file (e.g., `math_util.go`) later.
// For now, they are included here to make genes.go runnable standalone.
/*
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
func clamp(value, minVal, maxVal float64) float64 {
	return math.Max(minVal, math.Min(value, maxVal))
}
*/
