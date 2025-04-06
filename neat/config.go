package neat

import (
	"fmt"
	"strings"

	"gopkg.in/ini.v1"
)

// Config stores the configuration parameters for the NEAT algorithm.
type Config struct {
	Neat         NeatConfig
	Genome       GenomeConfig
	Reproduction ReproductionConfig
	SpeciesSet   SpeciesSetConfig
	Stagnation   StagnationConfig
}

// NeatConfig holds parameters specific to the NEAT algorithm itself.
type NeatConfig struct {
	PopSize              int     `ini:"pop_size"`
	FitnessCriterion     string  `ini:"fitness_criterion"` // e.g., "max", "min", "mean"
	FitnessThreshold     float64 `ini:"fitness_threshold"`
	ResetOnExtinction    bool    `ini:"reset_on_extinction"`
	NoFitnessTermination bool    `ini:"no_fitness_termination"`
}

// GenomeConfig holds parameters specific to the structure and mutation of genomes.
type GenomeConfig struct {
	// --- Top-level Genome parameters ---
	NumInputs                        int     `ini:"num_inputs"`
	NumOutputs                       int     `ini:"num_outputs"`
	NumHidden                        int     `ini:"num_hidden"`
	FeedForward                      bool    `ini:"feed_forward"` // If true, recurrent connections are disallowed
	CompatibilityDisjointCoefficient float64 `ini:"compatibility_disjoint_coefficient"`
	CompatibilityWeightCoefficient   float64 `ini:"compatibility_weight_coefficient"`
	ConnAddProb                      float64 `ini:"conn_add_prob"`
	ConnDeleteProb                   float64 `ini:"conn_delete_prob"`
	NodeAddProb                      float64 `ini:"node_add_prob"`
	NodeDeleteProb                   float64 `ini:"node_delete_prob"`
	SingleStructuralMutation         bool    `ini:"single_structural_mutation"` // Python default: false
	StructuralMutationSurer          string  `ini:"structural_mutation_surer"`  // Python default: 'default'
	InitialConnection                string  `ini:"initial_connection"`         // Python default: 'unconnected'

	// --- Node Gene parameters ---
	BiasInitMean    float64 `ini:"bias_init_mean"`
	BiasInitStdev   float64 `ini:"bias_init_stdev"`
	BiasInitType    string  `ini:"bias_init_type"` // Default: 'gaussian'
	BiasReplaceRate float64 `ini:"bias_replace_rate"`
	BiasMutateRate  float64 `ini:"bias_mutate_rate"`
	BiasMutatePower float64 `ini:"bias_mutate_power"`
	BiasMaxValue    float64 `ini:"bias_max_value"`
	BiasMinValue    float64 `ini:"bias_min_value"`

	ResponseInitMean    float64 `ini:"response_init_mean"`
	ResponseInitStdev   float64 `ini:"response_init_stdev"`
	ResponseInitType    string  `ini:"response_init_type"` // Default: 'gaussian'
	ResponseReplaceRate float64 `ini:"response_replace_rate"`
	ResponseMutateRate  float64 `ini:"response_mutate_rate"`
	ResponseMutatePower float64 `ini:"response_mutate_power"`
	ResponseMaxValue    float64 `ini:"response_max_value"`
	ResponseMinValue    float64 `ini:"response_min_value"`

	ActivationDefault    string   `ini:"activation_default"`           // Default: 'random'
	ActivationOptions    []string `ini:"activation_options" delim:" "` // Space-separated list
	ActivationMutateRate float64  `ini:"activation_mutate_rate"`

	AggregationDefault    string   `ini:"aggregation_default"`           // Default: 'random'
	AggregationOptions    []string `ini:"aggregation_options" delim:" "` // Space-separated list
	AggregationMutateRate float64  `ini:"aggregation_mutate_rate"`

	// --- Connection Gene parameters ---
	WeightInitMean    float64 `ini:"weight_init_mean"`
	WeightInitStdev   float64 `ini:"weight_init_stdev"`
	WeightInitType    string  `ini:"weight_init_type"` // Default: 'gaussian'
	WeightReplaceRate float64 `ini:"weight_replace_rate"`
	WeightMutateRate  float64 `ini:"weight_mutate_rate"`
	WeightMutatePower float64 `ini:"weight_mutate_power"`
	WeightMaxValue    float64 `ini:"weight_max_value"`
	WeightMinValue    float64 `ini:"weight_min_value"`

	EnabledDefault        string  `ini:"enabled_default"` // Default: 'True'
	EnabledMutateRate     float64 `ini:"enabled_mutate_rate"`
	EnabledRateToTrueAdd  float64 `ini:"enabled_rate_to_true_add"`  // Python default: 0.0
	EnabledRateToFalseAdd float64 `ini:"enabled_rate_to_false_add"` // Python default: 0.0

	// --- Calculated/Derived ---
	InputKeys    []int // Derived
	OutputKeys   []int // Derived
	NodeKeyIndex int   // Derived, used for assigning new node keys
}

// ReproductionConfig holds parameters related to reproduction.
type ReproductionConfig struct {
	Elitism           int     `ini:"elitism"`            // Python default: 0
	SurvivalThreshold float64 `ini:"survival_threshold"` // Python default: 0.2
	MinSpeciesSize    int     `ini:"min_species_size"`   // Python default: 1
}

// SpeciesSetConfig holds parameters related to speciation.
type SpeciesSetConfig struct {
	CompatibilityThreshold float64 `ini:"compatibility_threshold"`
}

// StagnationConfig holds parameters related to species stagnation.
type StagnationConfig struct {
	SpeciesFitnessFunc string `ini:"species_fitness_func"` // Python default: 'mean'
	MaxStagnation      int    `ini:"max_stagnation"`       // Python default: 15
	SpeciesElitism     int    `ini:"species_elitism"`      // Python default: 0
}

// LoadConfig loads configuration parameters from an INI file.
func LoadConfig(filePath string) (*Config, error) {
	cfg, err := ini.LoadSources(ini.LoadOptions{
		IgnoreInlineComment:         true, // Allow # comments starting with # or ;
		UnescapeValueCommentSymbols: true, // If # or ; appear in value, treat as value
	}, filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config file '%s': %w", filePath, err)
	}

	config := &Config{}

	// Map sections to structs
	if err := cfg.Section("NEAT").MapTo(&config.Neat); err != nil {
		return nil, fmt.Errorf("failed to map [NEAT] section: %w", err)
	}
	if err := cfg.Section("DefaultGenome").MapTo(&config.Genome); err != nil {
		// Try mapping individual params for more specific error messages if section map fails
		return nil, fmt.Errorf("failed to map [DefaultGenome] section: %w", err)
	}
	if err := cfg.Section("DefaultReproduction").MapTo(&config.Reproduction); err != nil {
		return nil, fmt.Errorf("failed to map [DefaultReproduction] section: %w", err)
	}
	if err := cfg.Section("DefaultSpeciesSet").MapTo(&config.SpeciesSet); err != nil {
		return nil, fmt.Errorf("failed to map [DefaultSpeciesSet] section: %w", err)
	}
	if err := cfg.Section("DefaultStagnation").MapTo(&config.Stagnation); err != nil {
		return nil, fmt.Errorf("failed to map [DefaultStagnation] section: %w", err)
	}

	// --- Manually reload potentially problematic bool/float values ---
	// This is a workaround in case MapTo has issues with comments or specific formats
	neatSection := cfg.Section("NEAT")
	ffKey, err := neatSection.GetKey("fitness_threshold")
	if err == nil { // Only overwrite if key exists
		config.Neat.FitnessThreshold, _ = ffKey.Float64() // Ignore error on re-parse? Or handle?
	}
	ffKey, err = neatSection.GetKey("no_fitness_termination")
	if err == nil {
		config.Neat.NoFitnessTermination, _ = ffKey.Bool()
	}
	ffKey, err = neatSection.GetKey("reset_on_extinction")
	if err == nil {
		config.Neat.ResetOnExtinction, _ = ffKey.Bool()
	}

	genomeSection := cfg.Section("DefaultGenome")
	ffKey, err = genomeSection.GetKey("feed_forward")
	if err == nil {
		config.Genome.FeedForward, _ = ffKey.Bool()
	}
	ffKey, err = genomeSection.GetKey("single_structural_mutation")
	if err == nil {
		config.Genome.SingleStructuralMutation, _ = ffKey.Bool()
	}

	// --- Explicitly clean potentially problematic string values ---
	config.Genome.BiasInitType = cleanIniString(config.Genome.BiasInitType)
	config.Genome.ResponseInitType = cleanIniString(config.Genome.ResponseInitType)
	config.Genome.ActivationDefault = cleanIniString(config.Genome.ActivationDefault)
	config.Genome.AggregationDefault = cleanIniString(config.Genome.AggregationDefault)
	config.Genome.WeightInitType = cleanIniString(config.Genome.WeightInitType)
	config.Genome.EnabledDefault = cleanIniString(config.Genome.EnabledDefault)
	config.Genome.InitialConnection = cleanIniString(config.Genome.InitialConnection)
	config.Genome.StructuralMutationSurer = cleanIniString(config.Genome.StructuralMutationSurer)
	config.Neat.FitnessCriterion = cleanIniString(config.Neat.FitnessCriterion)
	config.Stagnation.SpeciesFitnessFunc = cleanIniString(config.Stagnation.SpeciesFitnessFunc)
	// Clean list options (trim spaces from each element)
	for i, opt := range config.Genome.ActivationOptions {
		config.Genome.ActivationOptions[i] = strings.TrimSpace(opt)
	}
	for i, opt := range config.Genome.AggregationOptions {
		config.Genome.AggregationOptions[i] = strings.TrimSpace(opt)
	}

	// Set Defaults (where Python version had them hardcoded or implied)
	// Note: The ini library handles defaults if specified in the struct tag (e.g. `default:"value"`),
	// but many Python defaults were implicit or set programmatically.
	if config.Genome.BiasInitType == "" {
		config.Genome.BiasInitType = "gaussian"
	}
	if config.Genome.ResponseInitType == "" {
		config.Genome.ResponseInitType = "gaussian"
	}
	if config.Genome.ActivationDefault == "" {
		config.Genome.ActivationDefault = "random"
	}
	if config.Genome.AggregationDefault == "" {
		config.Genome.AggregationDefault = "random"
	}
	if config.Genome.WeightInitType == "" {
		config.Genome.WeightInitType = "gaussian"
	}
	if config.Genome.EnabledDefault == "" {
		config.Genome.EnabledDefault = "True"
	} // Python bool attribute parses this
	// single_structural_mutation, structural_mutation_surer have Python defaults handled by tag/parsing logic
	if config.Reproduction.MinSpeciesSize == 0 {
		config.Reproduction.MinSpeciesSize = 1
	} // Default from Python Class
	if config.Reproduction.SurvivalThreshold == 0 {
		config.Reproduction.SurvivalThreshold = 0.2
	} // Default from Python Class
	if config.Stagnation.SpeciesFitnessFunc == "" {
		config.Stagnation.SpeciesFitnessFunc = "mean"
	} // Default from Python Class
	if config.Stagnation.MaxStagnation == 0 {
		config.Stagnation.MaxStagnation = 15
	} // Default from Python Class

	// --- Post-processing and Validation ---

	// Derive Input/Output Keys
	config.Genome.InputKeys = make([]int, config.Genome.NumInputs)
	for i := 0; i < config.Genome.NumInputs; i++ {
		config.Genome.InputKeys[i] = -(i + 1)
	}
	config.Genome.OutputKeys = make([]int, config.Genome.NumOutputs)
	for i := 0; i < config.Genome.NumOutputs; i++ {
		config.Genome.OutputKeys[i] = i
	}
	// Initialize NodeKeyIndex (used for creating hidden nodes)
	// Start indexing after output nodes (0..NumOutputs-1)
	config.Genome.NodeKeyIndex = config.Genome.NumOutputs

	// Validate activation/aggregation options
	if len(config.Genome.ActivationOptions) == 0 {
		return nil, fmt.Errorf("config error: activation_options must be specified")
	}
	if len(config.Genome.AggregationOptions) == 0 {
		return nil, fmt.Errorf("config error: aggregation_options must be specified")
	}

	// Basic value validation (could be more extensive)
	if config.Genome.NumInputs <= 0 {
		return nil, fmt.Errorf("config error: num_inputs must be positive")
	}
	if config.Genome.NumOutputs <= 0 {
		return nil, fmt.Errorf("config error: num_outputs must be positive")
	}
	if config.Genome.CompatibilityDisjointCoefficient < 0 {
		return nil, fmt.Errorf("config error: compatibility_disjoint_coefficient cannot be negative")
	}
	if config.Genome.CompatibilityWeightCoefficient < 0 {
		return nil, fmt.Errorf("config error: compatibility_weight_coefficient cannot be negative")
	}
	if config.Genome.ConnAddProb < 0 || config.Genome.ConnAddProb > 1 {
		return nil, fmt.Errorf("config error: conn_add_prob must be between 0 and 1")
	}
	if config.Genome.ConnDeleteProb < 0 || config.Genome.ConnDeleteProb > 1 {
		return nil, fmt.Errorf("config error: conn_delete_prob must be between 0 and 1")
	}
	if config.Genome.NodeAddProb < 0 || config.Genome.NodeAddProb > 1 {
		return nil, fmt.Errorf("config error: node_add_prob must be between 0 and 1")
	}
	if config.Genome.NodeDeleteProb < 0 || config.Genome.NodeDeleteProb > 1 {
		return nil, fmt.Errorf("config error: node_delete_prob must be between 0 and 1")
	}
	// Check min/max values
	if config.Genome.BiasMaxValue < config.Genome.BiasMinValue {
		return nil, fmt.Errorf("config error: bias_max_value cannot be less than bias_min_value")
	}
	if config.Genome.ResponseMaxValue < config.Genome.ResponseMinValue {
		return nil, fmt.Errorf("config error: response_max_value cannot be less than response_min_value")
	}
	if config.Genome.WeightMaxValue < config.Genome.WeightMinValue {
		return nil, fmt.Errorf("config error: weight_max_value cannot be less than weight_min_value")
	}
	if config.Reproduction.SurvivalThreshold < 0 || config.Reproduction.SurvivalThreshold > 1 {
		return nil, fmt.Errorf("config error: survival_threshold must be between 0 and 1")
	}
	if config.Reproduction.MinSpeciesSize <= 0 {
		return nil, fmt.Errorf("config error: min_species_size must be positive")
	}
	if config.SpeciesSet.CompatibilityThreshold < 0 {
		return nil, fmt.Errorf("config error: compatibility_threshold cannot be negative")
	}
	if config.Stagnation.MaxStagnation <= 0 {
		return nil, fmt.Errorf("config error: max_stagnation must be positive")
	}

	// Validate fitness criterion
	validCriteria := map[string]bool{"max": true, "min": true, "mean": true}
	if !validCriteria[strings.ToLower(config.Neat.FitnessCriterion)] {
		return nil, fmt.Errorf("config error: invalid fitness_criterion '%s', must be one of 'max', 'min', 'mean'", config.Neat.FitnessCriterion)
	}

	// Validate initial connection type (more complex types like 'partial N' require further parsing later)
	validConnections := map[string]bool{
		"unconnected": true, "fs_neat_nohidden": true, "fs_neat": true, "fs_neat_hidden": true,
		"full_nodirect": true, "full": true, "full_direct": true,
		"partial_nodirect": true, "partial": true, "partial_direct": true,
	}
	baseConnection := strings.Fields(config.Genome.InitialConnection)[0]
	if !validConnections[baseConnection] {
		return nil, fmt.Errorf("config error: invalid initial_connection type '%s'", baseConnection)
	}

	// Validate stagnation fitness function
	validStagnationFuncs := map[string]bool{"max": true, "min": true, "mean": true, "median": true, "sum": true} // Based on Python math_util
	if !validStagnationFuncs[strings.ToLower(config.Stagnation.SpeciesFitnessFunc)] {
		return nil, fmt.Errorf("config error: invalid species_fitness_func '%s'", config.Stagnation.SpeciesFitnessFunc)
	}

	return config, nil
}

// Helper to get next node key - ensures unique positive integers >= NumOutputs
func (gc *GenomeConfig) GetNewNodeKey() int {
	key := gc.NodeKeyIndex
	gc.NodeKeyIndex++
	return key
}

// cleanIniString removes inline comments and trims whitespace from a string read from INI.
func cleanIniString(s string) string {
	// Remove comments starting with # or ;
	if idx := strings.IndexAny(s, "#;"); idx != -1 {
		s = s[:idx]
	}
	return strings.TrimSpace(s)
}
