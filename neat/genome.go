package neat

import (
	"fmt"
	"math/rand"
	"sort"
	"strings"
)

// Genome represents an individual organism in the population.
// It consists of NodeGenes and ConnectionGenes.
type Genome struct {
	Key         int                               // Unique identifier for this genome.
	Nodes       map[int]*NodeGene                 // Map node ID -> NodeGene
	Connections map[ConnectionKey]*ConnectionGene // Map connection key -> ConnectionGene
	Fitness     float64                           // Fitness score of the genome.
	// Config holds a reference to the configuration for easy access to parameters.
	// Note: Storing the whole config might be overkill; maybe just GenomeConfig?
	// Let's start with GenomeConfig.
	Config *GenomeConfig
}

// NewGenome creates a new Genome instance with the specified key and config reference.
func NewGenome(key int, config *GenomeConfig) *Genome {
	return &Genome{
		Key:         key,
		Nodes:       make(map[int]*NodeGene),
		Connections: make(map[ConnectionKey]*ConnectionGene),
		Fitness:     0.0,
		Config:      config,
	}
}

// ConfigureNew initializes a new genome based on the configuration.
// It creates input, output, and potentially hidden nodes, and sets up initial connections.
func (g *Genome) ConfigureNew() {
	// Create node genes for the output nodes first.
	for _, nodeKey := range g.Config.OutputKeys {
		g.Nodes[nodeKey] = NewNodeGene(nodeKey, g.Config)
	}

	// Create node genes for the hidden nodes, if any.
	if g.Config.NumHidden > 0 {
		for i := 0; i < g.Config.NumHidden; i++ {
			// Get a unique key for the new hidden node.
			// We use the NodeKeyIndex from the config, which should be initialized >= NumOutputs
			nodeKey := g.Config.GetNewNodeKey() // This increments the index
			// Ensure the key isn't already somehow used (shouldn't happen with proper indexing)
			if _, exists := g.Nodes[nodeKey]; exists {
				// This indicates a potential issue with NodeKeyIndex management
				panic(fmt.Sprintf("Attempted to create duplicate node key: %d", nodeKey))
			}
			g.Nodes[nodeKey] = NewNodeGene(nodeKey, g.Config)
		}
	}

	// Add connections based on the initial_connection configuration.
	// This part is complex and depends on the specific connection scheme.
	g.setupInitialConnections()
}

// setupInitialConnections creates the initial connections based on the config string.
func (g *Genome) setupInitialConnections() {
	connType := g.Config.InitialConnection
	// Handle potential 'partial N' format
	parts := strings.Fields(connType)
	baseConnType := parts[0]
	connectionFraction := 1.0 // Default for non-partial types
	if strings.HasPrefix(baseConnType, "partial") && len(parts) > 1 {
		// Try to parse the fraction (error handling might be needed here)
		// connectionFraction = strconv.ParseFloat(parts[1], 64)
		// For now, assume valid config means it's handled, maybe add later.
		// Let's just focus on the type for the switch statement.
	}

	// Collect input, output, and hidden node keys for easier iteration
	inputKeys := g.Config.InputKeys
	outputKeys := g.Config.OutputKeys
	hiddenKeys := []int{}
	for nk := range g.Nodes {
		isOutput := false
		for _, ok := range outputKeys {
			if nk == ok {
				isOutput = true
				break
			}
		}
		if !isOutput {
			hiddenKeys = append(hiddenKeys, nk)
		}
	}
	// Sort hidden keys for deterministic order if needed (though map iteration isn't guaranteed order)
	sort.Ints(hiddenKeys)

	// Based on Python neat/genome.py initial connection logic:
	switch baseConnType {
	case "unconnected":
		// No connections are made.
	case "fs_neat_nohidden", "fs_neat":
		// Connect all inputs to all outputs (FS-NEAT without hidden).
		// Python `fs_neat` also defaults to this if num_hidden > 0, with a warning.
		for _, ik := range inputKeys {
			for _, ok := range outputKeys {
				connKey := ConnectionKey{InNodeID: ik, OutNodeID: ok}
				g.Connections[connKey] = NewConnectionGene(connKey, g.Config)
			}
		}
	case "fs_neat_hidden":
		// Connect all inputs to all hidden nodes, and all hidden nodes to all outputs.
		for _, ik := range inputKeys {
			for _, hk := range hiddenKeys {
				connKey := ConnectionKey{InNodeID: ik, OutNodeID: hk}
				g.Connections[connKey] = NewConnectionGene(connKey, g.Config)
			}
		}
		for _, hk := range hiddenKeys {
			for _, ok := range outputKeys {
				connKey := ConnectionKey{InNodeID: hk, OutNodeID: ok}
				g.Connections[connKey] = NewConnectionGene(connKey, g.Config)
			}
		}
	case "full_nodirect", "full":
		// Connect inputs to hidden, hidden to hidden, and hidden to outputs.
		// No direct input-to-output connections.
		// Python `full` defaults to this if num_hidden > 0, with a warning.
		outputNodes := make(map[int]bool)
		for _, ok := range outputKeys {
			outputNodes[ok] = true
		}

		for _, ik := range inputKeys {
			for _, hk := range hiddenKeys {
				connKey := ConnectionKey{InNodeID: ik, OutNodeID: hk}
				g.Connections[connKey] = NewConnectionGene(connKey, g.Config)
			}
		}
		for _, hk1 := range hiddenKeys {
			for _, hk2 := range hiddenKeys {
				connKey := ConnectionKey{InNodeID: hk1, OutNodeID: hk2}
				g.Connections[connKey] = NewConnectionGene(connKey, g.Config)
			}
			for _, ok := range outputKeys {
				connKey := ConnectionKey{InNodeID: hk1, OutNodeID: ok}
				g.Connections[connKey] = NewConnectionGene(connKey, g.Config)
			}
		}
	case "full_direct":
		// Fully connect, including direct input-output, input-hidden, hidden-hidden, hidden-output.
		for _, ik := range inputKeys {
			for _, hk := range hiddenKeys {
				connKey := ConnectionKey{InNodeID: ik, OutNodeID: hk}
				g.Connections[connKey] = NewConnectionGene(connKey, g.Config)
			}
			for _, ok := range outputKeys {
				connKey := ConnectionKey{InNodeID: ik, OutNodeID: ok}
				g.Connections[connKey] = NewConnectionGene(connKey, g.Config)
			}
		}
		for _, hk1 := range hiddenKeys {
			for _, hk2 := range hiddenKeys {
				connKey := ConnectionKey{InNodeID: hk1, OutNodeID: hk2}
				g.Connections[connKey] = NewConnectionGene(connKey, g.Config)
			}
			for _, ok := range outputKeys {
				connKey := ConnectionKey{InNodeID: hk1, OutNodeID: ok}
				g.Connections[connKey] = NewConnectionGene(connKey, g.Config)
			}
		}
	case "partial_nodirect", "partial":
		// Partially connect (probabilistically) like full_nodirect.
		// Python `partial` defaults to this if num_hidden > 0, with a warning.
		// TODO: Implement probabilistic connection based on connectionFraction.
		fmt.Println("Warning: initial_connection 'partial_nodirect'/'partial' not fully implemented yet (using full_nodirect logic).")
		// Fallback to full_nodirect logic for now
		outputNodes := make(map[int]bool)
		for _, ok := range outputKeys {
			outputNodes[ok] = true
		}
		for _, ik := range inputKeys {
			for _, hk := range hiddenKeys {
				if rand.Float64() < connectionFraction { // Apply probability
					connKey := ConnectionKey{InNodeID: ik, OutNodeID: hk}
					g.Connections[connKey] = NewConnectionGene(connKey, g.Config)
				}
			}
		}
		for _, hk1 := range hiddenKeys {
			for _, hk2 := range hiddenKeys {
				if rand.Float64() < connectionFraction {
					connKey := ConnectionKey{InNodeID: hk1, OutNodeID: hk2}
					g.Connections[connKey] = NewConnectionGene(connKey, g.Config)
				}
			}
			for _, ok := range outputKeys {
				if rand.Float64() < connectionFraction {
					connKey := ConnectionKey{InNodeID: hk1, OutNodeID: ok}
					g.Connections[connKey] = NewConnectionGene(connKey, g.Config)
				}
			}
		}
	case "partial_direct":
		// Partially connect (probabilistically) like full_direct.
		fmt.Println("Warning: initial_connection 'partial_direct' not fully implemented yet (using full_direct logic).")
		// Fallback to full_direct logic for now
		for _, ik := range inputKeys {
			for _, hk := range hiddenKeys {
				if rand.Float64() < connectionFraction {
					connKey := ConnectionKey{InNodeID: ik, OutNodeID: hk}
					g.Connections[connKey] = NewConnectionGene(connKey, g.Config)
				}
			}
			for _, ok := range outputKeys {
				if rand.Float64() < connectionFraction {
					connKey := ConnectionKey{InNodeID: ik, OutNodeID: ok}
					g.Connections[connKey] = NewConnectionGene(connKey, g.Config)
				}
			}
		}
		for _, hk1 := range hiddenKeys {
			for _, hk2 := range hiddenKeys {
				if rand.Float64() < connectionFraction {
					connKey := ConnectionKey{InNodeID: hk1, OutNodeID: hk2}
					g.Connections[connKey] = NewConnectionGene(connKey, g.Config)
				}
			}
			for _, ok := range outputKeys {
				if rand.Float64() < connectionFraction {
					connKey := ConnectionKey{InNodeID: hk1, OutNodeID: ok}
					g.Connections[connKey] = NewConnectionGene(connKey, g.Config)
				}
			}
		}
	default:
		// This should be caught by config validation ideally
		panic(fmt.Sprintf("Invalid initial_connection type in genome configuration: %s", connType))
	}
}

// ConfigureCrossover creates a new genome by combining genes from two parent genomes.
func (g *Genome) ConfigureCrossover(parent1, parent2 *Genome) {
	// Assume parent1 is the more fit parent (convention from neat-python)
	// This matters for deciding which disjoint/excess genes to inherit.
	if parent1.Fitness < parent2.Fitness {
		parent1, parent2 = parent2, parent1 // Ensure parent1 is the fitter one
	}

	g.Config = parent1.Config // Child inherits config from parent

	// Inherit nodes: All nodes from the fitter parent are inherited.
	// Node attributes are handled during connection gene crossover if nodes match.
	// In neat-python, node crossover isn't explicitly done, nodes are just copied
	// from the primary parent, and the attributes only matter if the connection exists.
	// Let's follow that - copy all nodes from parent1.
	for key, node1 := range parent1.Nodes {
		g.Nodes[key] = node1.Copy() // Must copy to avoid modifying parent
	}

	// Inherit connection genes:
	for key, conn1 := range parent1.Connections {
		conn2, exists := parent2.Connections[key]
		if exists {
			// Homologous gene: crossover attributes.
			g.Connections[key] = conn1.Crossover(conn2)
		} else {
			// Disjoint or excess gene (from fitter parent): copy directly.
			g.Connections[key] = conn1.Copy()
		}
	}

	// Note: We don't explicitly inherit disjoint/excess genes from the less fit parent (parent2)
	// following the standard NEAT algorithm and neat-python's implementation.
}

// Mutate applies mutations to the genome, including structural and attribute mutations.
func (g *Genome) Mutate() {
	// Determine if only a single structural mutation should occur (if configured)
	singleMutation := g.Config.SingleStructuralMutation
	// Python's 'structural_mutation_surer' is complex, mapping 'default' -> single_structural_mutation
	// We'll simplify here: if singleMutation is true, we *might* do one structural change.
	structureMutated := false

	// --- Structural Mutations ---

	// Mutate: Add Node
	if rand.Float64() < g.Config.NodeAddProb {
		g.mutateAddNode()
		structureMutated = true
	}

	// Mutate: Add Connection
	if !singleMutation || !structureMutated {
		if rand.Float64() < g.Config.ConnAddProb {
			g.mutateAddConnection()
			structureMutated = true
		}
	}

	// Mutate: Delete Node (Optional, often less critical than adding)
	// Need careful implementation to handle associated connections.
	if !singleMutation || !structureMutated {
		if rand.Float64() < g.Config.NodeDeleteProb {
			// g.mutateDeleteNode() // Placeholder - implement if needed
			// structureMutated = true
		}
	}

	// Mutate: Delete Connection (Optional)
	if !singleMutation || !structureMutated {
		if rand.Float64() < g.Config.ConnDeleteProb {
			// g.mutateDeleteConnection() // Placeholder - implement if needed
			// structureMutated = true
		}
	}

	// --- Non-Structural Mutations (Attribute Mutations) ---
	// Mutate attributes of existing nodes.
	for _, node := range g.Nodes {
		node.Mutate(g.Config)
	}

	// Mutate attributes of existing connections.
	for _, conn := range g.Connections {
		conn.Mutate(g.Config)
	}
}

// mutateAddNode attempts to add a new node by splitting an existing connection.
func (g *Genome) mutateAddNode() {
	if len(g.Connections) == 0 {
		return // Cannot split if no connections exist.
	}

	// Choose a random connection to split.
	// Need a way to pick one randomly from the map.
	keys := make([]ConnectionKey, 0, len(g.Connections))
	for k := range g.Connections {
		keys = append(keys, k)
	}
	connToSplitKey := keys[rand.Intn(len(keys))]
	connToSplit := g.Connections[connToSplitKey]

	// If the chosen connection is already disabled, do nothing (or maybe re-enable?).
	// neat-python appears to allow splitting disabled connections.
	// if !connToSplit.Enabled {
	// 	return
	// }

	// Disable the original connection.
	connToSplit.Enabled = false

	// Create the new node.
	newNodeKey := g.Config.GetNewNodeKey()
	newNode := NewNodeGene(newNodeKey, g.Config)
	g.Nodes[newNodeKey] = newNode

	// Create the two new connections.
	// Connection from original input node to the new node.
	conn1Key := ConnectionKey{InNodeID: connToSplit.Key.InNodeID, OutNodeID: newNodeKey}
	conn1 := NewConnectionGene(conn1Key, g.Config)
	conn1.Weight = 1.0 // Weight of the input connection is 1.0 (standard NEAT)
	conn1.Enabled = true
	g.Connections[conn1Key] = conn1

	// Connection from the new node to the original output node.
	conn2Key := ConnectionKey{InNodeID: newNodeKey, OutNodeID: connToSplit.Key.OutNodeID}
	conn2 := NewConnectionGene(conn2Key, g.Config)
	conn2.Weight = connToSplit.Weight // Weight of the output connection is the original weight
	conn2.Enabled = true
	g.Connections[conn2Key] = conn2
}

// mutateAddConnection attempts to add a new connection between two previously unconnected nodes.
func (g *Genome) mutateAddConnection() {
	// Collect possible input and output nodes for the new connection.
	possibleInputs := make([]int, 0, len(g.Config.InputKeys)+len(g.Nodes))
	possibleInputs = append(possibleInputs, g.Config.InputKeys...)
	for nk := range g.Nodes {
		// Check if nk is already in InputKeys (it shouldn't be, but safety check)
		isInput := false
		for _, ik := range g.Config.InputKeys {
			if nk == ik {
				isInput = true
				break
			}
		}
		if !isInput {
			possibleInputs = append(possibleInputs, nk)
		}
	}

	possibleOutputs := make([]int, 0, len(g.Nodes))
	for nk := range g.Nodes { // Only output/hidden nodes can be outputs of a connection
		possibleOutputs = append(possibleOutputs, nk)
	}

	if len(possibleInputs) == 0 || len(possibleOutputs) == 0 {
		return // Cannot add connection if no possible start or end nodes.
	}

	// Attempt to find a valid pair of nodes that are not already connected.
	// Limit attempts to prevent infinite loops in densely connected genomes.
	maxAttempts := 20 // Arbitrary limit
	for i := 0; i < maxAttempts; i++ {
		inNodeKey := possibleInputs[rand.Intn(len(possibleInputs))]
		outNodeKey := possibleOutputs[rand.Intn(len(possibleOutputs))]

		// Check if the chosen output node is an input node (disallowed).
		isOutputAnInput := false
		for _, ik := range g.Config.InputKeys {
			if outNodeKey == ik {
				isOutputAnInput = true
				break
			}
		}
		if isOutputAnInput {
			continue // Output cannot be an input node
		}

		connKey := ConnectionKey{InNodeID: inNodeKey, OutNodeID: outNodeKey}

		// Check if this connection already exists.
		if _, exists := g.Connections[connKey]; exists {
			continue // Connection already exists
		}

		// Check for recurrent connection if feedforward is required.
		if g.Config.FeedForward {
			// Need a function to check if adding this connection creates a cycle.
			// This requires building a graph representation or traversal.
			if createsCycle(g, inNodeKey, outNodeKey) { // Placeholder function
				continue // Recurrent connection disallowed
			}
		}

		// Found a valid new connection.
		newConn := NewConnectionGene(connKey, g.Config)
		g.Connections[connKey] = newConn
		return // Successfully added a connection
	}

	// Failed to find a valid connection after multiple attempts.
	// fmt.Println("Warning: Failed to find a valid new connection to add.")
}

// Distance calculates the genetic distance between this genome and another.
// It considers disjoint/excess genes and differences in matching gene attributes.
func (g *Genome) Distance(other *Genome) float64 {
	// Ensure configs are compatible for distance calculation?
	// Assume they share the same basic config for now.
	disjointCount := 0
	// excessCount := 0 // Not explicitly counted in neat-python, handled by disjoint loop
	weightDiffSum := 0.0
	matchingGeneCount := 0

	// Use node keys to align nodes - assumes keys are consistent identifiers
	// Node distance calculation (optional, neat-python focuses on connections)
	// nodeDiffSum := 0.0
	// matchingNodes := 0
	// nodes1 := g.Nodes
	// nodes2 := other.Nodes
	// maxNodeKey := max(maxKey(nodes1), maxKey(nodes2))

	// Iterate over connections of the first genome.
	for key, conn1 := range g.Connections {
		if conn2, exists := other.Connections[key]; exists {
			// Matching connection gene.
			weightDiffSum += conn1.Distance(conn2, g.Config) // Distance includes weight and enabled status
			matchingGeneCount++
		} else {
			// Disjoint or excess gene in genome 1.
			disjointCount++ // Simplified: treat all non-matching as disjoint for now
		}
	}

	// Iterate over connections of the second genome to find its disjoint/excess genes.
	for key := range other.Connections {
		if _, exists := g.Connections[key]; !exists {
			// Disjoint or excess gene in genome 2.
			disjointCount++
		}
	}

	// Normalize N (number of genes in the larger genome)
	N := float64(max(len(g.Connections), len(other.Connections)))
	if N < 1.0 {
		N = 1.0
	} // Avoid division by zero if both genomes are empty

	// Calculate distance using the NEAT formula.
	// d = (c1 * E / N) + (c2 * D / N) + (c3 * W)
	// Where E=Excess, D=Disjoint, W=Avg Weight Diff
	// neat-python combines E and D.
	compatibility := (g.Config.CompatibilityDisjointCoefficient * float64(disjointCount)) / N
	if matchingGeneCount > 0 {
		averageWeightDiff := weightDiffSum / float64(matchingGeneCount)
		compatibility += g.Config.CompatibilityWeightCoefficient * averageWeightDiff
	}

	return compatibility
}

// Helper function: max returns the greater of two integers.
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Placeholder for cycle detection needed in mutateAddConnection
func createsCycle(genome *Genome, inNode, outNode int) bool {
	// Simple case: direct cycle
	if inNode == outNode {
		return true
	}

	// Check if outNode can reach inNode through existing enabled connections.
	visited := make(map[int]bool)
	queue := []int{outNode}

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		if current == inNode {
			return true // Found a path back
		}

		if visited[current] {
			continue
		}
		visited[current] = true

		// Find nodes reachable from current
		for connKey, conn := range genome.Connections {
			if conn.Enabled && connKey.InNodeID == current {
				queue = append(queue, connKey.OutNodeID)
			}
		}
	}

	return false // No path found
}

// TODO: Implement mutateDeleteNode and mutateDeleteConnection if needed.
// TODO: Consider more sophisticated handling of disjoint/excess genes in Distance.
