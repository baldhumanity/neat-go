package nn

import (
	"fmt"
	"sort"

	"github.com/baldhumanity/neat-go/neat" // Import the parent neat package
)

// neuralNode represents a node during network activation.
// It stores pre-fetched activation/aggregation functions and node properties.
type neuralNode struct {
	Key           int
	Bias          float64
	Response      float64
	ActivationFn  neat.ActivationType
	AggregationFn neat.AggregationType
	InputKeys     []neat.ConnectionKey // Incoming connections relevant to this node
}

// FeedForwardNetwork represents a phenotype network that can be activated.
// It assumes a feed-forward structure (no cycles).
type FeedForwardNetwork struct {
	InputKeys     []int                                      // List of input node keys (negative)
	OutputKeys    []int                                      // List of output node keys (0 to N-1)
	NodeEvalOrder []int                                      // Topologically sorted list of node keys for evaluation
	Nodes         map[int]neuralNode                         // Map of node key -> processed node data
	Connections   map[neat.ConnectionKey]neat.ConnectionGene // Map connection key -> connection gene (only enabled)
}

// CreateFeedForwardNetwork builds a runnable feed-forward network from a genome.
// It performs a topological sort to determine the activation order.
func CreateFeedForwardNetwork(g *neat.Genome) (*FeedForwardNetwork, error) {
	if !g.Config.FeedForward {
		// This function assumes FeedForward = true in config.
		// A separate creator would be needed for recurrent networks.
		return nil, fmt.Errorf("cannot create FeedForwardNetwork for a genome configured with FeedForward=false")
	}

	// Gather active nodes and connections
	nodes := make(map[int]neuralNode)
	connections := make(map[neat.ConnectionKey]neat.ConnectionGene)
	incomingConnections := make(map[int][]neat.ConnectionKey) // nodeKey -> list of incoming connKeys
	nodeKeys := make(map[int]bool)                            // Set of all node keys present (inputs implicitly included later)

	// Process genome nodes
	for key, gn := range g.Nodes {
		actFn, err := neat.GetActivation(gn.Activation)
		if err != nil {
			return nil, fmt.Errorf("failed to get activation function '%s' for node %d: %w", gn.Activation, key, err)
		}
		aggFn, err := neat.GetAggregation(gn.Aggregation)
		if err != nil {
			return nil, fmt.Errorf("failed to get aggregation function '%s' for node %d: %w", gn.Aggregation, key, err)
		}
		nodes[key] = neuralNode{
			Key:           key,
			Bias:          gn.Bias,
			Response:      gn.Response,
			ActivationFn:  actFn,
			AggregationFn: aggFn,
			InputKeys:     []neat.ConnectionKey{}, // Initialize empty, populate later
		}
		nodeKeys[key] = true
	}

	// Process genome connections (only enabled ones)
	for key, gc := range g.Connections {
		if !gc.Enabled {
			continue
		}
		connections[key] = *gc.Copy() // Store a copy

		// Record incoming connection for the target node
		outNodeKey := key.OutNodeID
		if _, exists := incomingConnections[outNodeKey]; !exists {
			incomingConnections[outNodeKey] = []neat.ConnectionKey{}
		}
		incomingConnections[outNodeKey] = append(incomingConnections[outNodeKey], key)

		// Ensure connected nodes are considered in our node set
		nodeKeys[key.InNodeID] = true
		nodeKeys[key.OutNodeID] = true
	}

	// Populate InputKeys for each neuralNode
	for key, node := range nodes {
		if inputs, ok := incomingConnections[key]; ok {
			node.InputKeys = inputs
			nodes[key] = node // Update map entry
		}
	}

	// Topological sort of nodes (Kahn's algorithm)
	inDegree := make(map[int]int) // nodeKey -> count of incoming connections
	graph := make(map[int][]int)  // nodeKey -> list of outgoing node keys
	allNodeKeysList := []int{}

	// Initialize graph and in-degrees
	for nk := range nodeKeys {
		allNodeKeysList = append(allNodeKeysList, nk)
		inDegree[nk] = 0 // Initialize
		if _, exists := graph[nk]; !exists {
			graph[nk] = []int{}
		}
	}
	// Add input keys explicitly if they weren't part of g.Nodes (which they shouldn't be)
	for _, ik := range g.Config.InputKeys {
		if _, exists := nodeKeys[ik]; !exists {
			allNodeKeysList = append(allNodeKeysList, ik)
			inDegree[ik] = 0
			graph[ik] = []int{}
			nodeKeys[ik] = true // Add to the overall set
		}
	}
	sort.Ints(allNodeKeysList) // Sort for deterministic processing (optional)

	for connKey := range connections {
		inNode := connKey.InNodeID
		outNode := connKey.OutNodeID
		graph[inNode] = append(graph[inNode], outNode)
		inDegree[outNode]++
	}

	// Kahn's algorithm queue
	queue := []int{}
	for _, nk := range allNodeKeysList {
		if inDegree[nk] == 0 {
			queue = append(queue, nk)
		}
	}
	sort.Ints(queue) // Sort initial queue for deterministic order

	evalOrder := []int{}
	for len(queue) > 0 {
		// Dequeue node
		u := queue[0]
		queue = queue[1:]
		evalOrder = append(evalOrder, u)

		// Process neighbors
		neighbors := graph[u]
		sort.Ints(neighbors) // Process neighbors deterministically
		for _, v := range neighbors {
			inDegree[v]--
			if inDegree[v] == 0 {
				queue = append(queue, v)
			}
		}
		sort.Ints(queue) // Keep queue sorted for determinism
	}

	// Check if sort was successful (cycle detection)
	if len(evalOrder) != len(nodeKeys) {
		// Cycle detected or disconnected nodes not reachable from inputs?
		// This should ideally not happen in a feed-forward configured genome
		// that passed validation, but check anyway.
		return nil, fmt.Errorf("failed topological sort: cycle detected or graph issue (expected %d nodes, got %d)", len(nodeKeys), len(evalOrder))
	}

	// Filter evalOrder to only include non-input nodes needed for activation loop
	filteredEvalOrder := []int{}
	inputKeySet := make(map[int]bool)
	for _, ik := range g.Config.InputKeys {
		inputKeySet[ik] = true
	}
	for _, nk := range evalOrder {
		if !inputKeySet[nk] { // Exclude explicit input nodes
			filteredEvalOrder = append(filteredEvalOrder, nk)
		}
	}

	net := &FeedForwardNetwork{
		InputKeys:     g.Config.InputKeys,
		OutputKeys:    g.Config.OutputKeys,
		NodeEvalOrder: filteredEvalOrder, // Use the order excluding inputs
		Nodes:         nodes,
		Connections:   connections,
	}

	return net, nil
}

// Activate computes the network's output for a given slice of input values.
// The input slice must match the number of input nodes.
func (net *FeedForwardNetwork) Activate(inputs []float64) ([]float64, error) {
	if len(inputs) != len(net.InputKeys) {
		return nil, fmt.Errorf("mismatch between input count (%d) and network input nodes (%d)", len(inputs), len(net.InputKeys))
	}

	// nodeValues stores the computed output of each node during activation.
	nodeValues := make(map[int]float64)

	// Initialize input node values.
	for i, ik := range net.InputKeys {
		nodeValues[ik] = inputs[i]
	}

	// Reusable buffer for incoming connection values to reduce allocations.
	var incInputsBuffer []float64

	// Activate nodes in topological order.
	for _, nodeKey := range net.NodeEvalOrder {
		node := net.Nodes[nodeKey]

		// Gather inputs for this node based on incoming connections.
		// Reuse the buffer, ensuring it's reset (sliced to zero length).
		// Ensure capacity is sufficient, grow if needed (less frequent than alloc).
		if cap(incInputsBuffer) < len(node.InputKeys) {
			incInputsBuffer = make([]float64, 0, len(node.InputKeys))
		}
		incInputs := incInputsBuffer[:0] // Reset slice length to 0, keep capacity

		for _, connKey := range node.InputKeys {
			conn := net.Connections[connKey]        // Assumes connection exists (validated during creation)
			inValue := nodeValues[connKey.InNodeID] // Value from the source node
			incInputs = append(incInputs, inValue*conn.Weight)
		}
		incInputsBuffer = incInputs // Update buffer reference in case append reallocated

		// Aggregate inputs.
		aggregated := node.AggregationFn(incInputs)

		// Apply bias and response scaling, then activation function.
		activationInput := aggregated + node.Bias
		activationInput *= node.Response
		outputValue := node.ActivationFn(activationInput)

		// Store the computed value for this node.
		nodeValues[nodeKey] = outputValue
	}

	// Collect outputs from the designated output nodes.
	outputs := make([]float64, len(net.OutputKeys))
	for i, ok := range net.OutputKeys {
		// Output nodes might not have been activated if they had no incoming enabled connections.
		// Default value should be 0 in that case, which is the default for map lookups.
		outputs[i] = nodeValues[ok]
	}

	return outputs, nil
}
