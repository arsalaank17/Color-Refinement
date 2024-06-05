## Graph Isomorphism Project running instructions 

---

### Graphs Directory

Put the graph files you would like to run the algorithm with in the directory:
`src/graphs`

### Running the algrotithm

To run an algorithm with a specific filename, use the `run(filename, mode, find_one)` function.

#### Parameters explanation
* `filename` - name of the graph file
* `mode` - has three possible values:
  * `GI` - Graph Isomorphism mode
  * `AUT` - Number of Automorphisms mode
  * `GIAUT` - Combination of the above (runs graph isomorphisms, 
            and counts the total number of isomorphisms per equivalence class)
* `find_one` - boolean indicating whether we terminate after only finding one isomorphism