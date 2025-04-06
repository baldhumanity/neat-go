package main

import (
	"errors"
	"fmt"
	"log"
	"os"

	"github.com/baldhumanity/neat-go/neat"
	"github.com/baldhumanity/neat-go/neat/nn"
)

// XOR inputs and expected outputs.
var xorInputs = [][]float64{
	{0.0, 0.0},
	{0.0, 1.0},
	{1.0, 0.0},
	{1.0, 1.0},
}
var xorOutputs = [][]float64{
	{0.0},
	{1.0},
	{1.0},
	{0.0},
}

// evalGenomes calculates the fitness for each genome in the population
// based on how well it performs on the XOR task.
func evalGenomes(genomes map[int]*neat.Genome) error {
	if len(genomes) == 0 {
		return errors.New("cannot evaluate fitness for empty population")
	}

	for _, g := range genomes {
		// Create a neural network from the genome.
		// Assuming genome has Config field populated (set during NewGenome or Crossover)
		if g.Config == nil {
			// This should ideally not happen if population management is correct
			g.Fitness = 0.0 // Assign minimal fitness
			fmt.Printf("Warning: Genome %d missing config reference during fitness evaluation.\n", g.Key)
			continue
		}

		// Need the nn subpackage
		net, err := nn.CreateFeedForwardNetwork(g)
		if err != nil {
			// Handle error: Could not create network (e.g., cycle detected improperly)
			// Assign low fitness or log the error.
			fmt.Printf("Warning: Failed to create network for genome %d: %v. Assigning fitness 0.\n", g.Key, err)
			g.Fitness = 0.0
			continue
		}

		// Evaluate the network on XOR inputs.
		sumSquaredError := 0.0
		for i, inputs := range xorInputs {
			outputs, err := net.Activate(inputs)
			if err != nil {
				// Handle activation error
				fmt.Printf("Warning: Network activation failed for genome %d: %v. Assigning fitness 0.\n", g.Key, err)
				g.Fitness = 0.0
				sumSquaredError = 4.0 * 4.0 // Max possible error to ensure low fitness
				break                       // Stop evaluating this genome
			}
			if len(outputs) == 0 {
				// Handle case where network produces no output (shouldn't happen with valid config)
				fmt.Printf("Warning: Network for genome %d produced no output. Assigning fitness 0.\n", g.Key)
				g.Fitness = 0.0
				sumSquaredError = 4.0 * 4.0
				break
			}

			error := outputs[0] - xorOutputs[i][0]
			sumSquaredError += error * error
		}

		// Calculate fitness using the standard formula for XOR, clamping to avoid rewarding large errors.
		// Fitness = max(0, 4.0 - sum_squared_error) ^ 2
		baseFitness := 4.0 - sumSquaredError
		if baseFitness < 0 {
			baseFitness = 0
		}
		fitness := baseFitness * baseFitness // Or math.Pow(baseFitness, 2)
		g.Fitness = fitness
	}
	return nil
}

func main() {
	// Config and Checkpoint file paths
	configPath := "./configs/xor-config"
	checkpointPrefix := "xor_checkpoint"
	checkpointFile := checkpointPrefix + ".gz"
	fmt.Printf("Loading configuration from: %s\n", configPath)

	// Load configuration.
	config, err := neat.LoadConfig(configPath)
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	var pop *neat.Population
	// Attempt to load from checkpoint.
	if _, err := os.Stat(checkpointFile); err == nil {
		fmt.Printf("Attempting to load population state from %s\n", checkpointFile)
		pop, err = neat.LoadCheckpoint(checkpointFile, configPath)
		if err != nil {
			log.Printf("WARN: Failed to load checkpoint: %v. Starting new evolution.\n", err)
			pop = nil // Ensure pop is nil if loading failed
		}
	} else {
		fmt.Println("No checkpoint file found. Starting new evolution.")
	}

	// If loading failed or no checkpoint existed, create a new population.
	if pop == nil {
		pop, err = neat.NewPopulation(config)
		if err != nil {
			log.Fatalf("Failed to create new population: %v", err)
		}
	}

	// Run the evolution.
	numGenerations := 300
	// Determine remaining generations if loaded from checkpoint
	startGen := pop.Generation + 1 // Start from the generation *after* the loaded one
	remGenerations := numGenerations - startGen + 1
	if remGenerations <= 0 {
		fmt.Println("Loaded checkpoint is already at or beyond the target number of generations.")
		// Display winner from loaded population
	} else {
		fmt.Printf("Running for %d generations (%d to %d)...\n", remGenerations, startGen, numGenerations)
		// Modify the Run function to accept starting generation and checkpoint interval?
		// For now, let's modify the loop here.

		winnerFound := false
		for g := 0; g < remGenerations; g++ {
			winner, err := pop.RunGeneration(evalGenomes) // Need a RunGeneration function
			if err != nil {
				log.Fatalf("Generation %d failed: %v", pop.Generation, err)
			}

			if winner != nil {
				fmt.Println("\nFitness threshold met!")
				pop.BestGenome = winner // Update best before final save
				winnerFound = true
				break

			}

			// Save checkpoint periodically
			if pop.Generation%2 == 0 {
				checkpointFilename := fmt.Sprintf("%s_gen%d.gz", checkpointPrefix, pop.Generation)
				err := pop.SaveCheckpoint(checkpointFilename)
				if err != nil {
					log.Printf("WARN: Failed to save checkpoint for generation %d: %v", pop.Generation, err)
				}
			}
		}
		if !winnerFound {
			fmt.Printf("\nReached maximum generations (%d).\n", numGenerations)
		}
	}

	// Save final checkpoint
	finalCheckpointFile := fmt.Sprintf("%s_final.gz", checkpointPrefix)
	err = pop.SaveCheckpoint(finalCheckpointFile)
	if err != nil {
		log.Printf("WARN: Failed to save final checkpoint: %v", err)
	}

	// Evolution finished - Use pop.BestGenome which was updated during the run
	winner := pop.BestGenome
	fmt.Println("\n--- Evolution Complete ---")
	if winner != nil {
		fmt.Printf("Best genome found (Key: %d, Fitness: %.4f, Gen: %d):\n", winner.Key, winner.Fitness, pop.Generation) // Show final generation
		fmt.Printf(" Nodes: %d, Connections: %d\n", len(winner.Nodes), len(winner.Connections))

		// Show the performance of the winner.
		winnerNet, err := nn.CreateFeedForwardNetwork(winner)
		if err != nil {
			log.Fatalf("Failed to create network from winner genome: %v", err)
		}

		fmt.Println("\nWinner network output:")
		fmt.Println(" Input | Expected | Output")
		fmt.Println("-----------------------------")
		for i, inputs := range xorInputs {
			output, err := winnerNet.Activate(inputs)
			if err != nil {
				fmt.Printf(" %v |   %.1f    | Error: %v\n", inputs, xorOutputs[i][0], err)
			} else {
				fmt.Printf(" %v |   %.1f    | %.4f\n", inputs, xorOutputs[i][0], output[0])
			}
		}

		// Inspect the winner genome's output node(s)
		fmt.Println("\nWinner Genome Output Node Details:")
		for _, key := range winner.Config.OutputKeys {
			if node, ok := winner.Nodes[key]; ok {
				fmt.Printf("  Node %d: Activation='%s', Response=%.3f, Bias=%.3f\n",
					key, node.Activation, node.Response, node.Bias)
			} else {
				fmt.Printf("  Output node key %d not found in winner genome!\n", key)
			}
		}

	} else {
		fmt.Println("No winner found within the given generations.")
	}
}
