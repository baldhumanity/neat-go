package neat

import (
	"fmt"
	"math"
	"sort"
)

// Species represents a group of genetically similar genomes.
type Species struct {
	Key             int             // Unique identifier for the species.
	Created         int             // Generation number when the species was created.
	LastImproved    int             // Last generation where fitness improved.
	Representative  *Genome         // The representative genome for this species.
	Members         map[int]*Genome // Genomes belonging to this species (maps genome key -> genome).
	Fitness         float64         // Calculated fitness for the species (e.g., mean fitness of members).
	AdjustedFitness float64         // Fitness adjusted by sharing.
	FitnessHistory  []float64       // History of fitness values for stagnation detection.
}

// NewSpecies creates a new species.
func NewSpecies(key, generation int) *Species {
	return &Species{
		Key:            key,
		Created:        generation,
		LastImproved:   generation,
		Members:        make(map[int]*Genome),
		FitnessHistory: []float64{},
	}
}

// Update adjusts the species' representative and members.
func (s *Species) Update(representative *Genome, members map[int]*Genome) {
	s.Representative = representative
	s.Members = members
}

// GetFitnesses returns a slice containing the fitness values of all members.
func (s *Species) GetFitnesses() []float64 {
	fitnesses := make([]float64, 0, len(s.Members))
	for _, g := range s.Members {
		fitnesses = append(fitnesses, g.Fitness)
	}
	return fitnesses
}

// --------------------------- GenomeDistanceCache ---------------------------

// GenomeDistanceCache stores calculated distances between genomes to avoid redundant computations.
type GenomeDistanceCache struct {
	Distances map[ConnectionKey]float64 // Using ConnectionKey as a proxy for genome pair (g1.Key, g2.Key)
	Hits      int
	Misses    int
	Config    *GenomeConfig // Needed for the Distance function
}

// NewGenomeDistanceCache creates a new distance cache.
func NewGenomeDistanceCache(config *GenomeConfig) *GenomeDistanceCache {
	return &GenomeDistanceCache{
		Distances: make(map[ConnectionKey]float64),
		Config:    config,
	}
}

// Distance calculates or retrieves the distance between two genomes.
func (dc *GenomeDistanceCache) Distance(genome1, genome2 *Genome) float64 {
	g1Key := genome1.Key
	g2Key := genome2.Key

	// Ensure order for cache key (g1 < g2)
	if g1Key > g2Key {
		g1Key, g2Key = g2Key, g1Key
	}

	cacheKey := ConnectionKey{InNodeID: g1Key, OutNodeID: g2Key}

	d, exists := dc.Distances[cacheKey]
	if exists {
		dc.Hits++
		return d
	}

	// Distance not in cache, compute it.
	dc.Misses++
	d = genome1.Distance(genome2) // Use the Genome.Distance method
	dc.Distances[cacheKey] = d
	return d
}

// --------------------------- SpeciesSet ---------------------------

// SpeciesSet manages the collection of species within a population.
type SpeciesSet struct {
	Species         map[int]*Species  // Map species key -> Species
	GenomeToSpecies map[int]int       // Map genome key -> species key
	Indexer         int               // Counter for assigning new species keys (start at 1)
	Config          *SpeciesSetConfig // Reference to speciation config
	// Reporters      *reporting.ReporterSet // TODO: Add reporters later
}

// NewSpeciesSet creates a new species set manager.
func NewSpeciesSet(config *SpeciesSetConfig) *SpeciesSet {
	return &SpeciesSet{
		Species:         make(map[int]*Species),
		GenomeToSpecies: make(map[int]int),
		Indexer:         1, // Start species IDs at 1
		Config:          config,
	}
}

// Speciate partitions the population into species based on genetic distance.
func (ss *SpeciesSet) Speciate(config *Config, population map[int]*Genome, generation int) error {
	if len(population) == 0 {
		ss.Species = make(map[int]*Species) // Reset if population is empty
		ss.GenomeToSpecies = make(map[int]int)
		return nil
	}

	compatibilityThreshold := ss.Config.CompatibilityThreshold
	distanceCache := NewGenomeDistanceCache(&config.Genome) // Need GenomeConfig for distance calcs

	// --- Step 1: Prepare ---
	unspeciated := make(map[int]*Genome, len(population))
	for k, v := range population {
		unspeciated[k] = v
	}
	newRepresentatives := make(map[int]*Genome) // species key -> new representative genome
	newMembers := make(map[int][]int)           // species key -> list of member genome keys

	// --- Step 2: Assign Representatives for Existing Species ---
	// Find the genome in the current population closest to the *old* representative.
	// This genome becomes the new representative for the next generation.
	// Note: This differs slightly from neat-python v0.92 which keeps old reps until after speciation.
	// Let's try the approach of picking the best new rep first.
	for sid, s := range ss.Species {
		if len(unspeciated) == 0 {
			break
		}

		candidates := []struct {
			Genome *Genome
			Dist   float64
		}{}

		// If the old representative is still in the population, consider it.
		// Otherwise, the species might die out if no members are close enough.
		if s.Representative == nil {
			// This shouldn't happen if species are managed correctly
			fmt.Printf("Warning: Species %d has no representative. Skipping.\n", sid)
			continue
		}

		for _, g := range unspeciated {
			d := distanceCache.Distance(s.Representative, g)
			candidates = append(candidates, struct {
				Genome *Genome
				Dist   float64
			}{g, d})
		}

		if len(candidates) == 0 {
			// No unspeciated genomes left to check against this species' rep
			continue
		}

		// Sort candidates by distance to the old representative.
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].Dist < candidates[j].Dist
		})

		// The closest genome becomes the new representative.
		newRep := candidates[0].Genome
		newRepresentatives[sid] = newRep
		newMembers[sid] = []int{newRep.Key}
		delete(unspeciated, newRep.Key)
	}

	// --- Step 3: Assign Remaining Genomes to Species ---
	// Convert remaining unspeciated map to a slice for predictable iteration order
	remainingGenomes := make([]*Genome, 0, len(unspeciated))
	for _, g := range unspeciated {
		remainingGenomes = append(remainingGenomes, g)
	}
	// Sort remaining genomes by key for deterministic assignment
	sort.Slice(remainingGenomes, func(i, j int) bool {
		return remainingGenomes[i].Key < remainingGenomes[j].Key
	})

	for _, g := range remainingGenomes {
		gid := g.Key

		bestSpecies := -1
		minDist := math.Inf(1)

		// Find the existing species (based on *new* representatives) this genome is closest to.
		for sid, rep := range newRepresentatives {
			d := distanceCache.Distance(rep, g)
			if d < compatibilityThreshold && d < minDist {
				minDist = d
				bestSpecies = sid
			}
		}

		if bestSpecies != -1 {
			// Assign to the best-matching existing species.
			newMembers[bestSpecies] = append(newMembers[bestSpecies], gid)
		} else {
			// No suitable species found, create a new one.
			newSID := ss.Indexer
			ss.Indexer++
			newRepresentatives[newSID] = g
			newMembers[newSID] = []int{gid}
		}
	}

	// --- Step 4: Update SpeciesSet ---
	newSpeciesMap := make(map[int]*Species)
	newGenomeToSpeciesMap := make(map[int]int)

	for sid, representative := range newRepresentatives {
		membersList := newMembers[sid]
		if len(membersList) == 0 {
			// This species died out (no representative assigned or members found)
			fmt.Printf("Info: Species %d died out.\n", sid)
			continue
		}

		s := ss.Species[sid] // Get existing species data if available
		if s == nil {
			// It's a newly created species
			s = NewSpecies(sid, generation)
			fmt.Printf("Info: Created new species %d represented by genome %d\n", sid, representative.Key)
		}

		memberMap := make(map[int]*Genome)
		for _, gid := range membersList {
			memberMap[gid] = population[gid] // Get pointer from original population map
			newGenomeToSpeciesMap[gid] = sid
		}

		s.Update(representative, memberMap)
		newSpeciesMap[sid] = s
	}

	ss.Species = newSpeciesMap
	ss.GenomeToSpecies = newGenomeToSpeciesMap

	// Report distance cache performance (optional)
	// fmt.Printf("Distance Cache: Hits=%d, Misses=%d\n", distanceCache.Hits, distanceCache.Misses)

	// Report mean/stdev genetic distance (optional)
	if len(distanceCache.Distances) > 0 {
		allDistances := make([]float64, 0, len(distanceCache.Distances))
		for _, d := range distanceCache.Distances {
			allDistances = append(allDistances, d)
		}
		meanDist := Mean(allDistances)
		stdevDist := Stdev(allDistances)
		fmt.Printf("Mean genetic distance: %.3f, Stdev: %.3f\n", meanDist, stdevDist)
	}

	return nil
}

// GetSpeciesID returns the species ID for a given genome ID.
func (ss *SpeciesSet) GetSpeciesID(genomeID int) (int, bool) {
	sid, exists := ss.GenomeToSpecies[genomeID]
	return sid, exists
}

// GetSpecies returns the Species object for a given genome ID.
func (ss *SpeciesSet) GetSpecies(genomeID int) (*Species, bool) {
	sid, exists := ss.GenomeToSpecies[genomeID]
	if !exists {
		return nil, false
	}
	s, exists := ss.Species[sid]
	return s, exists
}
