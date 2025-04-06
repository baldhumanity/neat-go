package neat

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// Reproduction handles the creation of new genomes, either from scratch or through crossover and mutation.
type Reproduction struct {
	Config *ReproductionConfig
	// GenomeIndexer func() int // Function removed, state stored in NextGenomeKey
	NextGenomeKey int           // State for the next genome key
	Ancestors     map[int][]int // Map genome key -> parent keys (for tracking lineage)
	// Reporters   *reporting.ReporterSet // TODO: Add reporters later
	Stagnation *Stagnation // Reference to stagnation info for filtering
}

// nextGenomeKeyGenerator returns a function that generates sequential genome keys starting from 1.
/* // Generator function removed, use r.getNextKey() instead
func nextGenomeKeyGenerator() func() int {
	key := 1
	return func() int {
		currentKey := key
		key++
		return currentKey
	}
}
*/

// getNextKey gets the next available genome key and increments the internal counter.
func (r *Reproduction) getNextKey() int {
	key := r.NextGenomeKey
	r.NextGenomeKey++
	return key
}

// NewReproduction creates a new reproduction manager.
func NewReproduction(config *ReproductionConfig, stagnation *Stagnation) *Reproduction {
	return &Reproduction{
		Config: config,
		// GenomeIndexer: nextGenomeKeyGenerator(), // Removed
		NextGenomeKey: 1, // Start genome keys at 1
		Ancestors:     make(map[int][]int),
		Stagnation:    stagnation,
	}
}

// CreateNewPopulation creates an initial population of genomes.
func (r *Reproduction) CreateNewPopulation(genomeConfig *GenomeConfig, popSize int) map[int]*Genome {
	newGenomes := make(map[int]*Genome, popSize)
	for i := 0; i < popSize; i++ {
		key := r.getNextKey() // Use method now
		g := NewGenome(key, genomeConfig)
		g.ConfigureNew() // Initialize nodes and connections based on config
		newGenomes[key] = g
		r.Ancestors[key] = []int{} // No parents for initial population
	}
	return newGenomes
}

// Reproduce creates the next generation of genomes based on the current species and their fitness.
func (r *Reproduction) Reproduce(overallConfig *Config, speciesSet *SpeciesSet, popSize int, generation int) (map[int]*Genome, error) {

	// --- Step 1: Evaluate Stagnation ---
	stagnationInfo, err := r.Stagnation.Update(speciesSet, generation)
	if err != nil {
		return nil, fmt.Errorf("failed to update stagnation: %w", err)
	}

	// --- Step 2: Filter Species & Calculate Adjusted Fitness ---
	allFitnesses := []float64{}
	remainingSpecies := []*Species{}
	for _, info := range stagnationInfo {
		if info.IsStagnant {
			// TODO: Report species stagnant (using reporter system later)
			fmt.Printf("Info: Species %d removed due to stagnation.\n", info.SpeciesID)
		} else {
			sp := info.Species
			memberFitnesses := sp.GetFitnesses()
			if len(memberFitnesses) > 0 {
				allFitnesses = append(allFitnesses, memberFitnesses...)
				remainingSpecies = append(remainingSpecies, sp)
			} else {
				// Species has no members, even if not stagnant - cannot reproduce
				fmt.Printf("Info: Species %d removed as it has no members.\n", info.SpeciesID)
			}
		}
	}

	if len(remainingSpecies) == 0 {
		// TODO: Handle extinction (reset population?)
		fmt.Println("Error: All species became extinct!")
		// Based on config.Neat.ResetOnExtinction, might need to create a new population here.
		// For now, return empty.
		return make(map[int]*Genome), nil
	}

	// Calculate adjusted fitness based on fitness sharing
	minFitness := MinFloat(allFitnesses)
	maxFitness := MaxFloat(allFitnesses)
	fitnessRange := math.Max(1.0, maxFitness-minFitness) // Avoid division by zero, ensure range >= 1.0

	adjustedFitnessSum := 0.0
	for _, sp := range remainingSpecies {
		// Use the species fitness calculated during stagnation update
		meanSpeciesFitness := sp.Fitness
		adjustedFitness := (meanSpeciesFitness - minFitness) / fitnessRange
		sp.AdjustedFitness = adjustedFitness
		adjustedFitnessSum += adjustedFitness
	}

	// --- Step 3: Calculate Spawn Amounts ---
	previousSizes := make([]int, len(remainingSpecies))
	adjustedFitnesses := make([]float64, len(remainingSpecies))
	for i, sp := range remainingSpecies {
		previousSizes[i] = len(sp.Members)
		adjustedFitnesses[i] = sp.AdjustedFitness
	}

	minSpeciesSize := r.Config.MinSpeciesSize
	// Consider elitism when calculating minimum size for spawning logic
	// (ensures elite slots don't artificially inflate perceived spawn capacity)
	spawnMinSize := max(minSpeciesSize, r.Config.Elitism)

	spawnAmounts := computeSpawnAmounts(adjustedFitnesses, adjustedFitnessSum, previousSizes, popSize, spawnMinSize)

	// --- Step 4: Create New Population ---
	newPopulation := make(map[int]*Genome)
	newAncestors := make(map[int][]int)

	for i, sp := range remainingSpecies {
		spawn := spawnAmounts[i]
		spawn = max(spawn, r.Config.Elitism) // Ensure elitism minimum

		if spawn <= 0 {
			continue // Should not happen if spawnMinSize >= 1, but safety check
		}

		// Sort old members by fitness (descending) for elitism and parent selection.
		oldMembers := make([]*Genome, 0, len(sp.Members))
		for _, g := range sp.Members {
			oldMembers = append(oldMembers, g)
		}
		sort.Slice(oldMembers, func(i, j int) bool {
			return oldMembers[i].Fitness > oldMembers[j].Fitness
		})

		// Transfer elites.
		elitesTaken := 0
		if r.Config.Elitism > 0 {
			for j := 0; j < r.Config.Elitism && j < len(oldMembers); j++ {
				eliteGenome := oldMembers[j]
				newPopulation[eliteGenome.Key] = eliteGenome           // Transfer directly
				newAncestors[eliteGenome.Key] = []int{eliteGenome.Key} // Mark as its own ancestor for tracking
				elitesTaken++
			}
		}
		spawn -= elitesTaken
		if spawn <= 0 {
			continue
		}

		// Determine parents for remaining spawn.
		survivalCutoff := int(math.Ceil(r.Config.SurvivalThreshold * float64(len(oldMembers))))
		survivalCutoff = max(survivalCutoff, 2) // Need at least two parents
		if survivalCutoff > len(oldMembers) {
			survivalCutoff = len(oldMembers)
		}
		if survivalCutoff < 1 && len(oldMembers) > 0 {
			survivalCutoff = 1
		} // Handle edge case where threshold is 0 but members exist

		parents := oldMembers[:survivalCutoff]

		if len(parents) == 0 {
			// This should only happen if a species survives stagnation/filtering but has 0 members
			// or if survival threshold is extremely low. Skip spawning for this species.
			fmt.Printf("Warning: No parents available for species %d despite spawn > 0.\n", sp.Key)
			continue
		}

		// Produce offspring.
		for j := 0; j < spawn; j++ {
			// Select parents randomly from the surviving pool.
			parent1 := parents[rand.Intn(len(parents))]
			parent2 := parents[rand.Intn(len(parents))]

			// Create child genome.
			childKey := r.getNextKey() // Use method now
			child := NewGenome(childKey, &overallConfig.Genome)
			child.ConfigureCrossover(parent1, parent2)
			child.Mutate()

			newPopulation[childKey] = child
			newAncestors[childKey] = []int{parent1.Key, parent2.Key}
		}
	}
	r.Ancestors = newAncestors // Update ancestor tracking for the new generation

	// Final check: if population size is drastically different from target, log warning?
	if len(newPopulation) != popSize {
		fmt.Printf("Warning: New population size (%d) differs from target (%d).\n", len(newPopulation), popSize)
	}

	return newPopulation, nil
}

// computeSpawnAmounts calculates the number of offspring each species should produce.
func computeSpawnAmounts(adjustedFitnesses []float64, adjustedFitnessSum float64, previousSizes []int, popSize int, minSpeciesSize int) []int {
	spawnAmounts := make([]int, len(adjustedFitnesses))

	for i, af := range adjustedFitnesses {
		ps := previousSizes[i]
		var s float64
		if adjustedFitnessSum > 0 {
			// Proportional spawn based on adjusted fitness
			s = af / adjustedFitnessSum * float64(popSize)
		} else {
			// If total adjusted fitness is zero (e.g., all have min fitness),
			// distribute remaining slots evenly or just give minimum.
			s = float64(minSpeciesSize)
		}
		s = math.Max(float64(minSpeciesSize), s) // Ensure minimum size

		// Adjustment based on previous size (dampening effect from neat-python)
		d := (s - float64(ps)) * 0.5
		c := int(math.Round(d)) // Round to nearest integer
		spawn := ps
		if math.Abs(float64(c)) > 0 {
			spawn += c
		} else if d > 0 {
			spawn++
		} else if d < 0 {
			spawn--
		}
		spawnAmounts[i] = max(minSpeciesSize, spawn) // Ensure minimum size again after adjustment
	}

	// Normalize spawn amounts to match the target population size.
	totalSpawn := 0
	for _, sa := range spawnAmounts {
		totalSpawn += sa
	}

	if totalSpawn == 0 {
		// Avoid division by zero if somehow totalSpawn is 0
		// Assign minimum to all species? This case shouldn't happen if minSpeciesSize >= 1.
		fmt.Println("Warning: Total spawn calculated as 0. Assigning minimum size to all species.")
		for i := range spawnAmounts {
			spawnAmounts[i] = minSpeciesSize
		}
		totalSpawn = len(spawnAmounts) * minSpeciesSize
		if totalSpawn == 0 {
			return spawnAmounts
		} // Still 0, return as is.
	}

	norm := float64(popSize) / float64(totalSpawn)
	finalSpawnAmounts := make([]int, len(spawnAmounts))
	currentTotal := 0
	for i, sa := range spawnAmounts {
		// Calculate normalized spawn, ensuring minimum size.
		normalizedSpawn := int(math.Round(float64(sa) * norm))
		finalSpawnAmounts[i] = max(minSpeciesSize, normalizedSpawn)
		currentTotal += finalSpawnAmounts[i]
	}

	// Adjust final amounts if the total doesn't match popSize due to rounding/minimums.
	diff := popSize - currentTotal
	if diff != 0 {
		// Add/remove individuals one by one from species, prioritizing larger ones?
		// Or simply add/remove from random species? Let's do random.
		indices := make([]int, len(finalSpawnAmounts))
		for i := range indices {
			indices[i] = i
		}
		rand.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })

		for _, idx := range indices {
			if diff == 0 {
				break
			}
			if diff > 0 {
				finalSpawnAmounts[idx]++
				diff--
			} else { // diff < 0
				if finalSpawnAmounts[idx] > minSpeciesSize { // Don't reduce below minimum
					finalSpawnAmounts[idx]--
					diff++
				}
			}
		}
		// If diff still not zero (e.g., couldn't reduce enough due to min size), log warning.
		if diff != 0 {
			fmt.Printf("Warning: Could not exactly match pop_size after spawn normalization. Final size may differ slightly.\n")
		}
	}

	return finalSpawnAmounts
}
