package nn

import (
	"fmt"
	"sort"

	"github.com/baldhumanity/neat-go/neat" // Import the parent neat package
)

// InputConnection stores pre-calculated information for an incoming connection to a node.
type InputConnection struct {
	InputNodeIndex int     // The slice index of the node providing input
	Weight         float64 // The weight of the connection
}

// neuralNode represents a node during network activation, optimized for slice access.
// It stores pre-fetched activation/aggregation functions and pre-processed input connection info.
type neuralNode struct {
	OriginalKey   int // Original node key (useful for debugging/reference)
	Bias          float64
	Response      float64
	ActivationFn  neat.ActivationType
	AggregationFn neat.AggregationType
	Inputs        []InputConnection // Optimized incoming connections
}

// FeedForwardNetwork represents a phenotype network optimized for feed-forward activation using slice indexing.
type FeedForwardNetwork struct {
	InputIndices  []int        // Slice indices for input nodes
	OutputIndices []int        // Slice indices for output nodes
	NodeEvalOrder []int        // Topologically sorted list of node slice indices for evaluation (excluding inputs)
	Nodes         []neuralNode // Slice of all nodes (indexed 0..N-1), includes inputs
	NumNodes      int          // Total number of nodes (inputs + hidden + outputs)
}

// CreateFeedForwardNetwork builds a runnable, optimized feed-forward network from a genome.
// It assigns unique slice indices to each node and performs a topological sort on these indices.
func CreateFeedForwardNetwork(g *neat.Genome) (*FeedForwardNetwork, error) {
	if !g.Config.FeedForward {
		return nil, fmt.Errorf("cannot create FeedForwardNetwork for a genome configured with FeedForward=false")
	}

	// 1. Gather all unique node keys and create index mapping
	allNodeKeysMap := make(map[int]struct{}) // Use a map as a set for uniqueness
	inputKeysMap := make(map[int]struct{})
	outputKeysMap := make(map[int]struct{})

	for _, k := range g.Config.InputKeys {
		allNodeKeysMap[k] = struct{}{}
		inputKeysMap[k] = struct{}{}
	}
	for _, k := range g.Config.OutputKeys {
		allNodeKeysMap[k] = struct{}{}
		outputKeysMap[k] = struct{}{}
	}
	for k := range g.Nodes {
		allNodeKeysMap[k] = struct{}{}
	}
	enabledConnections := make(map[neat.ConnectionKey]neat.ConnectionGene)
	for key, gc := range g.Connections {
		if !gc.Enabled {
			continue
		}
		enabledConnections[key] = *gc.Copy()
		// Ensure connected nodes are included, even if not in input/output/defined nodes (shouldn't happen with valid genome)
		allNodeKeysMap[key.InNodeID] = struct{}{}
		allNodeKeysMap[key.OutNodeID] = struct{}{}
	}

	// Sort keys for deterministic index assignment
	allNodeKeysList := make([]int, 0, len(allNodeKeysMap))
	for k := range allNodeKeysMap {
		allNodeKeysList = append(allNodeKeysList, k)
	}
	sort.Ints(allNodeKeysList)

	nodeKeyToIndex := make(map[int]int, len(allNodeKeysList))
	indexToNodeKey := make([]int, len(allNodeKeysList)) // Optional, for debugging/reference
	for i, key := range allNodeKeysList {
		nodeKeyToIndex[key] = i
		indexToNodeKey[i] = key
	}
	numNodes := len(allNodeKeysList)

	// 2. Initialize the Nodes slice ensuring all nodes are covered
	nodesSlice := make([]neuralNode, numNodes)
	identityFn, err := neat.GetActivation("identity") // Lookup defaults once
	if err != nil {
		return nil, fmt.Errorf("failed to get default 'identity' activation function: %w", err)
	}
	sumAggFn, err := neat.GetAggregation("sum") // Lookup defaults once
	if err != nil {
		return nil, fmt.Errorf("failed to get default 'sum' aggregation function: %w", err)
	}

	for idx, key := range indexToNodeKey { // Iterate through ALL nodes by index/key
		nodesSlice[idx].OriginalKey = key            // Set original key first
		nodesSlice[idx].Inputs = []InputConnection{} // Ensure Inputs slice exists

		if gn, isInGenome := g.Nodes[key]; isInGenome {
			// Node is defined in the genome (could be hidden, output, or even input)
			actFn, err := neat.GetActivation(gn.Activation)
			if err != nil {
				return nil, fmt.Errorf("failed to get activation function '%s' for node %d: %w", gn.Activation, key, err)
			}
			aggFn, err := neat.GetAggregation(gn.Aggregation)
			if err != nil {
				return nil, fmt.Errorf("failed to get aggregation function '%s' for node %d: %w", gn.Aggregation, key, err)
			}
			nodesSlice[idx].Bias = gn.Bias
			nodesSlice[idx].Response = gn.Response
			nodesSlice[idx].ActivationFn = actFn
			nodesSlice[idx].AggregationFn = aggFn
		} else if _, isInput := inputKeysMap[key]; isInput {
			// Node is an input node NOT defined in the genome (pure input)
			nodesSlice[idx].Bias = 0.0
			nodesSlice[idx].Response = 1.0
			nodesSlice[idx].ActivationFn = identityFn
			nodesSlice[idx].AggregationFn = sumAggFn
		} else {
			// Node is NOT in genome and NOT a pure input.
			// This might be an output node listed in Config.OutputKeys but missing from g.Nodes,
			// or a node referenced only by a connection (potentially invalid genome).
			// Assigning defaults is a lenient approach; erroring might be stricter.
			if _, isOutput := outputKeysMap[key]; isOutput {
				// fmt.Printf("Warning: Output node %d not found in g.Nodes, using default functions/bias/response\n", key)
			} else {
				// This node isn't explicitly an output either - definitely unexpected.
				// Consider returning an error here for stricter validation.
				// fmt.Printf("Warning: Node %d used in network but not defined in g.Nodes, inputs, or outputs. Using defaults.\n", key)
			}
			nodesSlice[idx].Bias = 0.0                // Default
			nodesSlice[idx].Response = 1.0            // Default
			nodesSlice[idx].ActivationFn = identityFn // Default
			nodesSlice[idx].AggregationFn = sumAggFn  // Default
		}
	}

	// 3. Populate Inputs for each node in the slice (this adds to the initialized nodesSlice)
	for connKey, gc := range enabledConnections {
		inNodeIndex, okIn := nodeKeyToIndex[connKey.InNodeID]
		outNodeIndex, okOut := nodeKeyToIndex[connKey.OutNodeID]
		if !okIn || !okOut {
			// Should not happen
			return nil, fmt.Errorf("internal error: connection key node (%d or %d) not found in index map", connKey.InNodeID, connKey.OutNodeID)
		}

		inputConn := InputConnection{
			InputNodeIndex: inNodeIndex,
			Weight:         gc.Weight,
		}
		nodesSlice[outNodeIndex].Inputs = append(nodesSlice[outNodeIndex].Inputs, inputConn)
	}

	// 4. Topological sort using indices (Kahn's algorithm)
	inDegree := make([]int, numNodes)
	graph := make([][]int, numNodes) // Adjacency list: graph[i] lists nodes that node i outputs to

	for targetNodeIndex := range nodesSlice {
		for _, inputConn := range nodesSlice[targetNodeIndex].Inputs {
			sourceNodeIndex := inputConn.InputNodeIndex
			inDegree[targetNodeIndex]++
			if graph[sourceNodeIndex] == nil {
				graph[sourceNodeIndex] = []int{}
			}
			graph[sourceNodeIndex] = append(graph[sourceNodeIndex], targetNodeIndex)
		}
	}

	// Kahn's algorithm queue (indices)
	queue := []int{}
	for i := 0; i < numNodes; i++ {
		if inDegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	sort.Ints(queue) // Sort initial queue for deterministic order

	fullEvalOrderIndices := []int{} // Stores the full order including inputs
	for len(queue) > 0 {
		// Dequeue node index
		u := queue[0]
		queue = queue[1:]
		fullEvalOrderIndices = append(fullEvalOrderIndices, u)

		// Process neighbors (indices)
		neighbors := graph[u] // Nodes that 'u' outputs to
		sort.Ints(neighbors)  // Process neighbors deterministically
		for _, v := range neighbors {
			inDegree[v]--
			if inDegree[v] == 0 {
				queue = append(queue, v)
			}
		}
		sort.Ints(queue) // Keep queue sorted for determinism
	}

	// Check if sort was successful (cycle detection)
	if len(fullEvalOrderIndices) != numNodes {
		// Cycle detected or graph issue
		return nil, fmt.Errorf("failed topological sort: cycle detected or graph issue (expected %d nodes, got %d)", numNodes, len(fullEvalOrderIndices))
	}

	// 5. Filter evalOrder to exclude input node indices
	finalEvalOrder := make([]int, 0, numNodes) // Capacity estimate
	inputIndexSet := make(map[int]struct{})
	for _, ik := range g.Config.InputKeys {
		inputIndexSet[nodeKeyToIndex[ik]] = struct{}{}
	}
	for _, nodeIndex := range fullEvalOrderIndices {
		if _, isInput := inputIndexSet[nodeIndex]; !isInput {
			finalEvalOrder = append(finalEvalOrder, nodeIndex)
		}
	}

	// 6. Prepare InputIndices and OutputIndices
	inputIndices := make([]int, len(g.Config.InputKeys))
	for i, key := range g.Config.InputKeys {
		inputIndices[i] = nodeKeyToIndex[key]
	}
	outputIndices := make([]int, len(g.Config.OutputKeys))
	for i, key := range g.Config.OutputKeys {
		outputIndices[i] = nodeKeyToIndex[key]
	}

	// 7. Construct the network
	net := &FeedForwardNetwork{
		InputIndices:  inputIndices,
		OutputIndices: outputIndices,
		NodeEvalOrder: finalEvalOrder, // Use the order excluding inputs
		Nodes:         nodesSlice,
		NumNodes:      numNodes,
	}

	return net, nil
}

// Activate computes the network's output for a given slice of input values.
// The input slice must match the number of input nodes configured.
// This version uses slice indexing for potentially faster activation.
func (net *FeedForwardNetwork) Activate(inputs []float64) ([]float64, error) {
	if len(inputs) != len(net.InputIndices) {
		return nil, fmt.Errorf("mismatch between input count (%d) and network input nodes (%d)", len(inputs), len(net.InputIndices))
	}

	// nodeValues stores the computed output of each node (indexed 0..NumNodes-1).
	// Consider passing a pre-allocated buffer if Activate is called very frequently.
	nodeValues := make([]float64, net.NumNodes)

	// Initialize input node values using their slice indices.
	for i, inputIndex := range net.InputIndices {
		nodeValues[inputIndex] = inputs[i]
	}

	// Reusable buffer for incoming connection values to reduce allocations.
	var incInputsBuffer []float64

	// Activate nodes in topological order (indices, excluding inputs).
	for _, nodeIndex := range net.NodeEvalOrder {
		node := net.Nodes[nodeIndex] // Fast slice access

		// Gather weighted inputs for this node.
		// Reuse the buffer.
		requiredCapacity := len(node.Inputs)
		if cap(incInputsBuffer) < requiredCapacity {
			incInputsBuffer = make([]float64, 0, requiredCapacity)
		}
		// Reset slice length to 0, keep capacity.
		incInputs := incInputsBuffer[:0]

		for _, conn := range node.Inputs { // Iterate over pre-processed InputConnection slice
			inValue := nodeValues[conn.InputNodeIndex] // Fast slice access for input value
			incInputs = append(incInputs, inValue*conn.Weight)
		}
		incInputsBuffer = incInputs // Update buffer reference in case append reallocated

		// Aggregate inputs.
		aggregated := node.AggregationFn(incInputs)

		// Apply bias and response scaling, then activation function.
		// Using direct float arithmetic is generally fast.
		activationInput := aggregated + node.Bias
		activationInput *= node.Response // Apply response scaling
		outputValue := node.ActivationFn(activationInput)

		// Store the computed value for this node (fast slice assignment).
		nodeValues[nodeIndex] = outputValue
	}

	// Collect outputs from the designated output nodes using their indices.
	outputs := make([]float64, len(net.OutputIndices))
	for i, outputIndex := range net.OutputIndices {
		// Output nodes might have 0 activation if they weren't reachable or had no inputs.
		// Default float64 value (0.0) from nodeValues slice is correct here.
		outputs[i] = nodeValues[outputIndex]
	}

	return outputs, nil
}
