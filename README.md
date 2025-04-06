# NEAT-Go

NEAT-Go is a Go implementation of the NeuroEvolution of Augmenting Topologies (NEAT) algorithm, developed by Kenneth O. Stanley and Risto Miikkulainen. This implementation is based on the original Python implementation [neat-python](https://github.com/CodeReclaimers/neat-python).

## Overview

NEAT is a genetic algorithm for the generation of evolving artificial neural networks. It alters both the weighting parameters and structures of networks, attempting to find a balance between the fitness of evolved solutions and their diversity.

## Features

- Core NEAT algorithm implementation
- XOR problem example
- Genome operations including mutation and crossover
- Species management and stagnation handling
- Checkpointing system
- Reporting system

## Installation

```go
go get github.com/baldhumanity/neat-go
```

## Quick Start

Here's a simple example of how to use NEAT-Go to solve the XOR problem:

```go
package main

import (
	"fmt"
	"log"

	"github.com/baldhumanity/neat-go/neat"
)

func main() {
	// Load configuration
	config, err := neat.LoadConfig("path/to/config")
	if err != nil {
		log.Fatalf("Error loading config: %v", err)
	}

	// Create a new population
	pop, err := neat.NewPopulation(config)
	if err != nil {
		log.Fatalf("Error creating population: %v", err)
	}

	// Run for 100 generations
	for i := 0; i < 100; i++ {
		winner, err := pop.RunGeneration(evalGenomes)
		if err != nil {
			log.Fatalf("Error running generation: %v", err)
		}
		
		if winner != nil {
			fmt.Println("Solution found!")
			break
		}
	}
}

// Define your fitness function
func evalGenomes(genomes map[int]*neat.Genome) error {
	// Evaluate each genome and set its fitness
	// ...
	return nil
}
```

## Documentation

For full documentation, see the [GoDoc](https://godoc.org/github.com/yourusername/neat-go).

## Example Configuration

NEAT-Go uses configuration files to set parameters for the evolution. A sample configuration file can be found in the examples directory.

## Running the XOR Example

```
cd examples/xor
go run main.go
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Kenneth O. Stanley and Risto Miikkulainen for developing the NEAT algorithm
- The neat-python project which served as a reference for this implementation 