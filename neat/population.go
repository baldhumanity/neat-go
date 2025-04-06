package neat

import (
	// "compress/gzip" // Moved to checkpoint.go
	// "encoding/gob" // Moved to checkpoint.go
	"fmt"
	// "math/rand" // Moved to checkpoint.go
	// "os" // Moved to checkpoint.go
	"math"
	"time" // Added import
	// Added missing sort import
)

// FitnessFunc is the type for the function provided by the user to evaluate genome fitness.
// It takes the current generation of genomes and should update their Fitness field.
// The genomes map maps genome key to the Genome object.
type FitnessFunc func(genomes map[int]*Genome) error

// Population holds the state of the NEAT evolutionary process.
type Population struct {
	Config       *Config
	Population   map[int]*Genome // Current generation of genomes (maps genome key -> genome)
	SpeciesSet   *SpeciesSet
	Reproduction *Reproduction
	Stagnation   *Stagnation
	Generation   int
	BestGenome   *Genome // Best genome found so far
	// TODO: Add Reporters
}

// NewPopulation creates a new Population instance.
// It initializes the first generation of genomes based on the config.
func NewPopulation(config *Config) (*Population, error) {
	stagnation, err := NewStagnation(&config.Stagnation)
	if err != nil {
		return nil, fmt.Errorf("failed to create stagnation manager: %w", err)
	}

	reproduction := NewReproduction(&config.Reproduction, stagnation)
	initialPopulation := reproduction.CreateNewPopulation(&config.Genome, config.Neat.PopSize)
	speciesSet := NewSpeciesSet(&config.SpeciesSet)

	p := &Population{
		Config:       config,
		Population:   initialPopulation,
		SpeciesSet:   speciesSet,
		Reproduction: reproduction,
		Stagnation:   stagnation,
		Generation:   0,
		BestGenome:   nil,
	}
	return p, nil
}

// RunGeneration executes a single generation of the NEAT algorithm.
// Returns the winning genome if the fitness threshold is met this generation, otherwise nil.
func (p *Population) RunGeneration(fitnessFunc FitnessFunc) (*Genome, error) {
	p.Generation++
	genStartTime := time.Now() // Need to import "time"
	fmt.Printf("****** Generation %d ******\n", p.Generation)

	// 1. Evaluate Fitness
	fmt.Println(" Evaluating fitness...")
	if err := fitnessFunc(p.Population); err != nil {
		return nil, fmt.Errorf("fitness evaluation failed in generation %d: %w", p.Generation, err)
	}

	// 2. Track Best Genome & Check Termination Condition
	currentBest := p.findBestGenome()
	bestUpdated := false
	if p.BestGenome == nil || (currentBest != nil && currentBest.Fitness > p.BestGenome.Fitness) {
		p.BestGenome = currentBest
		bestUpdated = true
		// Print only if it's truly a new overall best
		if bestUpdated && p.BestGenome != nil {
			fmt.Printf(" New best genome found! Key: %d, Fitness: %.4f\n", p.BestGenome.Key, p.BestGenome.Fitness)
		}
	}

	if currentBest != nil {
		fmt.Printf(" Best of generation %d: Key: %d, Fitness: %.4f\n", p.Generation, currentBest.Key, currentBest.Fitness)
	}

	// Check fitness threshold termination
	if !p.Config.Neat.NoFitnessTermination && p.BestGenome != nil {
		if p.BestGenome.Fitness >= p.Config.Neat.FitnessThreshold {
			// Don't print threshold met here, let the main loop handle it.
			return p.BestGenome, nil // Return winner
		}
	}

	// Check for empty population (extinction before reproduction)
	if len(p.Population) == 0 {
		fmt.Println("Population extinct before speciation/reproduction.")
		if p.Config.Neat.ResetOnExtinction {
			fmt.Println("Resetting population due to extinction.")
			p.Population = p.Reproduction.CreateNewPopulation(&p.Config.Genome, p.Config.Neat.PopSize)
			p.SpeciesSet = NewSpeciesSet(&p.Config.SpeciesSet) // Reset species too
			// Continue to next generation is handled by the main loop structure
			return nil, nil // No winner yet, but continue
		} else {
			// Return current best (which might be nil or from previous gen) + error
			return p.BestGenome, fmt.Errorf("population extinct in generation %d", p.Generation)
		}
	}

	// 3. Speciate
	fmt.Println(" Speciating...")
	if err := p.SpeciesSet.Speciate(p.Config, p.Population, p.Generation); err != nil {
		// Return current best + error
		return p.BestGenome, fmt.Errorf("speciation failed in generation %d: %w", p.Generation, err)
	}
	fmt.Printf(" Population divided into %d species.\n", len(p.SpeciesSet.Species))

	// 4. Reproduce
	fmt.Println(" Reproducing...")
	newPopulation, err := p.Reproduction.Reproduce(p.Config, p.SpeciesSet, p.Config.Neat.PopSize, p.Generation)
	if err != nil {
		// Return current best + error
		return p.BestGenome, fmt.Errorf("reproduction failed in generation %d: %w", p.Generation, err)
	}

	// Check for extinction after reproduction
	if len(newPopulation) == 0 {
		fmt.Println("Population extinct after reproduction.")
		if p.Config.Neat.ResetOnExtinction {
			fmt.Println("Resetting population due to extinction.")
			p.Population = p.Reproduction.CreateNewPopulation(&p.Config.Genome, p.Config.Neat.PopSize)
			p.SpeciesSet = NewSpeciesSet(&p.Config.SpeciesSet) // Reset species too
			return nil, nil                                    // No winner yet, but continue
		} else {
			// Return current best + error
			return p.BestGenome, fmt.Errorf("population extinct in generation %d", p.Generation)
		}
	} else {
		p.Population = newPopulation
	}

	// TODO: Add Reporting Calls Here

	genEndTime := time.Now()
	fmt.Printf("Generation %d finished in %s\n\n", p.Generation, genEndTime.Sub(genStartTime))

	return nil, nil // No winner found this generation
}

// findBestGenome finds the genome with the highest fitness in the current population.
func (p *Population) findBestGenome() *Genome {
	var best *Genome = nil
	maxFitness := math.Inf(-1)

	for _, g := range p.Population {
		if g.Fitness > maxFitness {
			maxFitness = g.Fitness
			best = g
		}
	}
	return best
}
