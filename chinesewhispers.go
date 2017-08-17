/*
 * Copyright 2017 Onchere Bironga
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package gocw

import (
	"math"
	"math/rand"
	"sort"
)

// Pair represents an edge in a directed graph
type Pair struct {
	// identifies indices of the samples at the ends of an edge.
	Idx1, Idx2 uint64
	// can be used for any purpose
	Distance float64
}

type Edges []Pair

func (e Edges) Len() int { return len(e) }

func (e Edges) Less(i, j int) bool {
	return e[i].Idx1 < e[j].Idx1 || (e[i].Idx1 == e[j].Idx1 && e[i].Idx2 < e[j].Idx2)
}

func (e Edges) Swap(i, j int) { e[i], e[j] = e[j], e[i] }

// ChineseWhispers implements the chinese whispers
// graph clustering algorithm
type ChineseWhispers struct {
	numIterations uint64
	edges         Edges
	labels        []uint64
}

// NewChineseWhispers gives a new ChineseWhispers instance
func NewChineseWhispers(numIterations uint64) *ChineseWhispers {
	return &ChineseWhispers{
		numIterations: numIterations,
	}
}

// AddEdge adds graph edges
func (c *ChineseWhispers) AddEdge(pair Pair) {
	c.edges = append(c.edges, pair)
}

func (c *ChineseWhispers) ensureOrdered() {
	if sort.IsSorted(c.edges) {
		return
	}
	ordered := make(Edges, len(c.edges)*2)
	for i := 0; i < len(c.edges); i++ {
		ordered = append(ordered, Pair{
			Idx1:     c.edges[i].Idx1,
			Idx2:     c.edges[i].Idx2,
			Distance: c.edges[i].Distance,
		})
		if c.edges[i].Idx1 != c.edges[i].Idx2 {
			ordered = append(ordered, Pair{
				Idx1:     c.edges[i].Idx2,
				Idx2:     c.edges[i].Idx1,
				Distance: c.edges[i].Distance,
			})
		}
	}
	sort.Sort(ordered)
}

func (c *ChineseWhispers) findNeighbourRanges(neighbours *[][2]uint64) {
	// setup neighbors so that [neighbors[i][0], neighbors[i][1]) is the range
	// within edges that contains all node i's edges.
	numNodes := func() uint64 {
		if len(c.edges) == 0 {
			return 0
		}
		var maxIdx uint64
		for i := 0; i < c.edges.Len(); i++ {
			if c.edges[i].Idx1 > maxIdx {
				maxIdx = c.edges[i].Idx1
			}
			if c.edges[i].Idx2 > maxIdx {
				maxIdx = c.edges[i].Idx2
			}
		}
		return maxIdx + 1
	}()
	for i := 0; i < int(numNodes); i++ {
		(*neighbours) = append((*neighbours), [2]uint64{})
	}
	var curNode, startIdx uint64
	for i := 0; i < c.edges.Len(); i++ {
		if c.edges[i].Idx1 != curNode {
			(*neighbours)[curNode] = [2]uint64{startIdx, uint64(i)}
			startIdx = uint64(i)
			curNode = c.edges[i].Idx1
		}
	}
	if len(*neighbours) != 0 {
		(*neighbours)[curNode] = [2]uint64{startIdx, uint64(len(c.edges))}
	}
}

// Run runs the algorithm returning number of labels
func (c *ChineseWhispers) Run() int {
	c.ensureOrdered()
	c.labels = []uint64{}
	if c.edges.Len() == 0 {
		return 0
	}
	var neighbours [][2]uint64
	c.findNeighbourRanges(&neighbours)
	// Initialize the labels, each node gets a different label.
	c.labels = make([]uint64, len(neighbours))
	for i := 0; i < len(c.labels); i++ {
		c.labels[i] = uint64(i)
	}
	for i := 0; i < len(neighbours)*int(c.numIterations); i++ {
		// Pick a random node.
		idx := rand.Int63() % int64(len(neighbours))
		// Count how many times each label happens amongst our neighbors.
		labelsToCounts := make(map[uint64]float64)
		end := neighbours[idx][1]
		for n := neighbours[idx][0]; n != end; n++ {
			labelsToCounts[c.labels[c.edges[n].Idx2]] += c.edges[n].Distance
		}
		// find the most common label
		bestScore := math.Inf(-1)
		bestLabel := c.labels[idx]
		for k, v := range labelsToCounts {
			if v > bestScore {
				bestScore = v
				bestLabel = k
			}
		}
		c.labels[idx] = bestLabel
	}
	// Remap the labels into a contiguous range.  First we find the
	// mapping.
	labelRemap := make(map[uint64]uint64)
	for i := 0; i < len(c.labels); i++ {
		nextIdx := len(labelRemap)
		labelRemap[c.labels[i]] = uint64(nextIdx)
	}
	// now apply the mapping to all the labels.
	for i := 0; i < len(c.labels); i++ {
		c.labels[i] = labelRemap[c.labels[i]]
	}
	return len(labelRemap)
}

// GetLabel returns the label at the index idx
func (c *ChineseWhispers) GetLabel(idx uint64) uint64 {
	return c.labels[idx]
}

// GetLabels returns the labels
func (c *ChineseWhispers) GetLabels() []uint64 {
	return c.labels
}
