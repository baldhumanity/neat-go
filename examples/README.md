# NEAT-Go Examples

This directory contains examples of how to use the NEAT-Go library.

## XOR Example

The XOR example demonstrates how to use NEAT-Go to solve the XOR problem. The XOR (exclusive OR) problem is a classical problem in machine learning that requires a network to learn a non-linear function.

### Running the XOR Example

```
cd xor
go run main.go
```

### Configuration

The XOR example uses a configuration file `evolve-config` that specifies the parameters for the NEAT algorithm. The file includes settings for:

- Population size
- Compatibility threshold for species
- Connection add/delete rates
- Node add/delete rates
- Weight mutation rates
- Activation functions
- And more

### Understanding the Results

The example will run for the specified number of generations or until a solution is found. A solution is found when a genome's fitness exceeds the fitness threshold specified in the configuration.

When a solution is found, the example will output:
- The genome's key and fitness
- The number of nodes and connections in the solution
- The network's output for each input combination

### Checkpoint Files

The example saves checkpoint files periodically. These files contain the state of the population and can be used to resume the evolution from a specific generation. 