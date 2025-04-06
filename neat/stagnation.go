package neat

import (
	"fmt"
	"math"
	"sort"
)

// Stagnation manages the detection of stagnant species.
type Stagnation struct {
	Config             *StagnationConfig
	SpeciesFitnessFunc func([]float64) float64
	// Reporters         *reporting.ReporterSet // TODO: Add reporters later
}

// NewStagnation creates a new stagnation manager.
func NewStagnation(config *StagnationConfig) (*Stagnation, error) {
	fn, ok := StatFunctions[config.SpeciesFitnessFunc]
	if !ok {
		return nil, fmt.Errorf("invalid species_fitness_func in config: %s", config.SpeciesFitnessFunc)
	}

	return &Stagnation{
		Config:             config,
		SpeciesFitnessFunc: fn,
	}, nil
}

// StagnationInfo holds the results of the stagnation update for a single species.
type StagnationInfo struct {
	SpeciesID  int
	Species    *Species
	IsStagnant bool
}

// Update checks for stagnant species within the species set.
// It updates species fitness history and marks species for removal based on stagnation criteria.
func (s *Stagnation) Update(speciesSet *SpeciesSet, generation int) ([]StagnationInfo, error) {
	if len(speciesSet.Species) == 0 {
		return []StagnationInfo{}, nil
	}

	speciesData := []struct {
		ID      int
		Species *Species
	}{}

	// Calculate fitness for each species and update history
	for sid, sp := range speciesSet.Species {
		previousMaxFitness := math.Inf(-1)
		if len(sp.FitnessHistory) > 0 {
			previousMaxFitness = MaxFloat(sp.FitnessHistory) // Use MaxFloat from math_util
		}

		memberFitnesses := sp.GetFitnesses()
		if len(memberFitnesses) == 0 {
			// Handle species with no members (should ideally not happen after speciation)
			sp.Fitness = math.Inf(-1) // Assign lowest possible fitness
		} else {
			sp.Fitness = s.SpeciesFitnessFunc(memberFitnesses)
		}

		sp.FitnessHistory = append(sp.FitnessHistory, sp.Fitness)
		sp.AdjustedFitness = 0 // Reset adjusted fitness, will be calculated later in reproduction

		if sp.Fitness > previousMaxFitness {
			sp.LastImproved = generation
		}

		speciesData = append(speciesData, struct {
			ID      int
			Species *Species
		}{sid, sp})
	}

	// Sort species by fitness (ascending - least fit first)
	sort.Slice(speciesData, func(i, j int) bool {
		return speciesData[i].Species.Fitness < speciesData[j].Species.Fitness
	})

	result := make([]StagnationInfo, len(speciesData))
	numSpecies := len(speciesData)
	numNonStagnant := numSpecies

	// Determine stagnation, applying species elitism
	for i, data := range speciesData {
		sp := data.Species
		stagnantTime := generation - sp.LastImproved
		isStagnant := false

		// Check if basic stagnation criteria is met
		if stagnantTime >= s.Config.MaxStagnation {
			// Check if elitism spares this species
			// Elitism protects the top `species_elitism` fittest species.
			// Since we sorted ascending, the last `species_elitism` are the fittest.
			if (numSpecies - i) > s.Config.SpeciesElitism {
				// This species is not among the elite, mark as stagnant
				isStagnant = true
				numNonStagnant--
			}
		}

		// Additional check from neat-python (seems redundant given the above but included for parity):
		// Override stagnant state if marking this species as stagnant would
		// result in the total number of species dropping below the limit.
		if numNonStagnant <= s.Config.SpeciesElitism && isStagnant {
			// This logic seems complex and potentially conflicts with the above check.
			// Let's stick to the simpler logic: only mark stagnant if not in the elite group.
			// The python code structure is slightly different here.
			// Re-evaluating the python code: it seems `num_non_stagnant` is used *before* deciding
			// `is_stagnant` based on elitism rank. Let's try that.

			/* Revised Logic Attempt (closer to Python order): */
			isStagnantStandard := stagnantTime >= s.Config.MaxStagnation
			isStagnant = false
			if numNonStagnant > s.Config.SpeciesElitism && isStagnantStandard {
				// Only consider it stagnant if removing it wouldn't drop below elite count
				isStagnant = true
			}
			// Double-check: Ensure elite species are never marked stagnant, regardless of time.
			if (numSpecies - i) <= s.Config.SpeciesElitism {
				isStagnant = false
			}

			if isStagnantStandard && !isStagnant {
				// We are sparing this species due to elitism
				fmt.Printf("Info: Species %d spared from stagnation due to elitism (Fitness: %.3f, Stagnant for: %d gen)\n", sp.Key, sp.Fitness, stagnantTime)
			} else if isStagnant {
				numNonStagnant-- // Decrement count only if truly marked stagnant
			}
			/* --- */
		}

		result[i] = StagnationInfo{
			SpeciesID:  data.ID,
			Species:    sp,
			IsStagnant: isStagnant,
		}
	}

	return result, nil
}
