package neat

import (
	"compress/gzip"
	"encoding/gob"
	"fmt" // Needed for Gob encoding/decoding of math/rand state
	"os"
)

// PopulationSaveData is a helper struct to hold only the parts of Population needed for saving.
// We don't save the full Config, as it's reloaded from the original file.
// We also need to explicitly save the random number generator state.
type PopulationSaveData struct {
	Population   map[int]*Genome
	SpeciesSet   *SpeciesSet
	Reproduction *Reproduction // Includes NextGenomeKey and Ancestors
	Generation   int
	BestGenome   *Genome
	// RandState    []byte // Marshaled state of the default math/rand source (REMOVED for simplicity)
}

// SaveCheckpoint saves the current state of the Population to a file.
// Uses gzip compression for smaller file size.
func (p *Population) SaveCheckpoint(filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create checkpoint file '%s': %w", filePath, err)
	}
	defer file.Close()

	// Use gzip for compression
	gzWriter := gzip.NewWriter(file)
	defer gzWriter.Close()

	// --- Prepare data for saving ---
	/* // Removed Rand state saving
	// Get the state of the default random number generator.
	// Note: This only saves the state of the *default* source (math/rand).
	// If other RNGs are used, their state needs separate handling.
	randBytes, err := rand.Source(0).(gob.GobEncoder).GobEncode() // Needs Go 1.18+ Source(0)
	if err != nil {
	    return fmt.Errorf("failed to marshal random state: %w", err)
	}
	*/
	saveData := PopulationSaveData{
		Population:   p.Population,
		SpeciesSet:   p.SpeciesSet,
		Reproduction: p.Reproduction, // Includes NextGenomeKey
		Generation:   p.Generation,
		BestGenome:   p.BestGenome, // Might be nil
		// RandState:    randBytes, // Removed
	}

	// --- Register types needed for Gob encoding ---
	// Gob needs to know about the concrete types being encoded, especially for interfaces
	// or structs containing unexported fields (though ours should be okay here).
	// Explicitly registering is good practice.
	gob.Register(map[int]*Genome{})
	gob.Register(map[ConnectionKey]*ConnectionGene{})
	gob.Register(map[int]*NodeGene{})
	gob.Register(map[int]*Species{})
	gob.Register(map[int]int{})
	gob.Register([]int{})
	// Add other complex types used within Population, SpeciesSet, Reproduction if needed

	// --- Encode the data ---
	encoder := gob.NewEncoder(gzWriter)
	err = encoder.Encode(saveData)
	if err != nil {
		return fmt.Errorf("failed to encode population data: %w", err)
	}

	fmt.Printf("Checkpoint saved to %s\n", filePath)
	return nil
}

// LoadCheckpoint loads a Population state from a checkpoint file.
// It requires the original configuration file path to reconstruct the Config object.
func LoadCheckpoint(checkpointPath string, configPath string) (*Population, error) {
	// 1. Load the configuration first.
	config, err := LoadConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config '%s' for checkpoint: %w", configPath, err)
	}

	// 2. Open the checkpoint file.
	file, err := os.Open(checkpointPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open checkpoint file '%s': %w", checkpointPath, err)
	}
	defer file.Close()

	// Use gzip for decompression
	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return nil, fmt.Errorf("failed to create gzip reader for checkpoint: %w", err)
	}
	defer gzReader.Close()

	// 3. Decode the saved data.
	saveData := PopulationSaveData{}
	decoder := gob.NewDecoder(gzReader)

	// Register types for decoding (must match encoding)
	gob.Register(map[int]*Genome{})
	gob.Register(map[ConnectionKey]*ConnectionGene{})
	gob.Register(map[int]*NodeGene{})
	gob.Register(map[int]*Species{})
	gob.Register(map[int]int{})
	gob.Register([]int{})

	err = decoder.Decode(&saveData)
	if err != nil {
		return nil, fmt.Errorf("failed to decode population data from checkpoint: %w", err)
	}

	/* // Removed Rand state loading
	// 4. Restore the random number generator state.
	// Note: This restores the *default* source (math/rand).
	err = rand.Source(0).(gob.GobDecoder).GobDecode(saveData.RandState)
	if err != nil {
	    return nil, fmt.Errorf("failed to unmarshal random state: %w", err)
	}
	*/

	// 5. Reconstruct the Population object.
	// Need to re-initialize Stagnation based on the loaded config.
	stagnation, err := NewStagnation(&config.Stagnation)
	if err != nil {
		return nil, fmt.Errorf("failed to re-initialize stagnation from loaded config: %w", err)
	}

	// Set the stagnation reference in the loaded Reproduction object
	if saveData.Reproduction != nil {
		saveData.Reproduction.Stagnation = stagnation
	}

	// Assign loaded config to genomes (Gob doesn't save/restore unexported or complex fields like pointers well by default)
	// We need to re-link the config to each loaded genome.
	if saveData.Population != nil {
		for _, genome := range saveData.Population {
			genome.Config = &config.Genome // Re-link the GenomeConfig part
		}
	}
	if saveData.BestGenome != nil {
		saveData.BestGenome.Config = &config.Genome // Re-link config for best genome too
	}
	// Also need to re-link GenomeConfig to the DistanceCache within SpeciesSet if it was saved
	if saveData.SpeciesSet != nil {
		// SpeciesSet wasn't part of PopulationSaveData initially, let's assume it's loaded correctly for now
		// If distance cache needs config, it should be re-initialized or re-linked here.
	}

	p := &Population{
		Config:       config, // Use the newly loaded config
		Population:   saveData.Population,
		SpeciesSet:   saveData.SpeciesSet,
		Reproduction: saveData.Reproduction,
		Stagnation:   stagnation, // Use the re-initialized stagnation manager
		Generation:   saveData.Generation,
		BestGenome:   saveData.BestGenome,
	}

	fmt.Printf("Checkpoint loaded from %s (Generation %d)\n", checkpointPath, p.Generation)
	return p, nil
}
