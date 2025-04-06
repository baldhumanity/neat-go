// Package neat provides a Go implementation of the NeuroEvolution of Augmenting Topologies (NEAT) algorithm.
//
// NEAT is a genetic algorithm for the generation of evolving artificial neural networks.
// It alters both the weighting parameters and structures of networks, attempting to find
// a balance between the fitness of evolved solutions and their diversity.
//
// This implementation is based on the original paper by Kenneth O. Stanley and Risto Miikkulainen
// and the neat-python implementation (https://github.com/CodeReclaimers/neat-python).
//
// Basic usage:
//
//	// Load configuration
//	config, err := neat.LoadConfig("path/to/config")
//	if err != nil {
//		log.Fatalf("Error loading config: %v", err)
//	}
//
//	// Create a new population
//	pop, err := neat.NewPopulation(config)
//	if err != nil {
//		log.Fatalf("Error creating population: %v", err)
//	}
//
//	// Run for 100 generations with your fitness function
//	for i := 0; i < 100; i++ {
//		winner, err := pop.RunGeneration(evalGenomes)
//		if err != nil {
//			log.Fatalf("Error running generation: %v", err)
//		}
//
//		if winner != nil {
//			fmt.Println("Solution found!")
//			break
//		}
//	}
package neat
