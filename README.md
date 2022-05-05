## Mobility Taxonomy
Tools for analysing the changes in importance hierarchies of time evolving complex networks

# Basic setup
- Clone this repo
- Create a ``dataset`` folder for all network data files
- Check data files consist of three columns seperated by 1 space
  1. ``n1`` _First node in edge_
  2. ``n2`` _Second node in edge_
  3. ``creation_time`` _Timestamp of edge creation_
- Create ``dataset_type_lookup.json`` in line with ``{"dataset_name": ["collection_type", "structure_type"]}`` (subsitituing for all strings)
- Run either ``plot_taxonomy_over_time.py`` or ``plot_taxonomy_per_timestep.py``
  - NB: comment out any plots you do not want to create

Any questions, message me or open an issue.
