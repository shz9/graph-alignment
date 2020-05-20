

# ARXIV NETSCIENCE MULTIPLEX

###### Last update: 18 March 2015

### Reference and Acknowledgments

This README file accompanies the dataset representing the multiplex coauthorship network of the free scientific repository "arXiv". 
If you use this dataset in your work either for analysis or for visualization, you should acknowledge/cite the following paper:
	
	“Identifying Modular Flows on Multilayer Networks Reveals Highly Overlapping Organization in Interconnected Systems”
	Manlio De Domenico, Andrea Lancichinetti, Alex Arenas, and Martin Rosvall
	Physical Review X 5, 011027 (2015)
	
that can be found at the following URLs:

<http://journals.aps.org/prx/abstract/10.1103/PhysRevX.5.011027>


This work has been supported by European Commission FET-Proactive project PLEXMATH (Grant No. 317614), the European project devoted to the investigation of multi-level complex systems and has been developed at the Alephsys Lab. 

Visit

PLEXMATH: <http://www.plexmath.eu/>

ALEPHSYS: <http://deim.urv.cat/~alephsys/>

for further details.



### Description of the dataset

The multiplex consists of layers corresponding to different arXiv categories. To restrict the analysis to a well-defined topic of research, we only included papers with "networks" in the title or abstract up to May 2014.

The multiplex network used in the paper makes use of 13 layers corresponding to:

1. physics.soc-ph
2. physics.data-an
3. physics.bio-ph
4. math-ph
5. math.OC
6. cond-mat.dis-nn
7. cond-mat.stat-mech
8. q-bio.MN
9. q-bio
10. q-bio.BM
11. nlin.AO
12. cs.SI
13. cs.CV

There are 14,489 nodes, labelled with integer ID between 0 and 14,488, and 59,026 coauthorship connections.
The multiplex is undirected (with only one direction specified) and weighted, stored as edges list in the file
    
    arxiv_netscience_multiplex.edges

with format

    layerID nodeID nodeID weight

The IDs of all layers are stored in 

    arxiv_netscience_layers.txt

The IDs of nodes are not provided.



### License

This ARXIV NETSCIENCE MULTIPLEX DATASET is made available under the Open Database License: <http://opendatacommons.org/licenses/odbl/1.0/>. Any rights in individual contents of the database are licensed under the Database Contents License: <http://opendatacommons.org/licenses/dbcl/1.0/>

You should find a copy of the above licenses accompanying this dataset. If it is not the case, please contact us (see below).

A friendly summary of this license can be found here:

<http://opendatacommons.org/licenses/odbl/summary/>

and is reported in the following.

======================================================
ODC Open Database License (ODbL) Summary

This is a human-readable summary of the ODbL 1.0 license. Please see the disclaimer below.

You are free:

*    To Share: To copy, distribute and use the database.
*    To Create: To produce works from the database.
*    To Adapt: To modify, transform and build upon the database.

As long as you:
    
*	Attribute: You must attribute any public use of the database, or works produced from the database, in the manner specified in the ODbL. For any use or redistribution of the database, or works produced from it, you must make clear to others the license of the database and keep intact any notices on the original database.
    
*	Share-Alike: If you publicly use any adapted version of this database, or works produced from an adapted database, you must also offer that adapted database under the ODbL.
    
*	Keep open: If you redistribute the database, or an adapted version of it, then you may use technological measures that restrict the work (such as DRM) as long as you also redistribute a version without such measures.

======================================================


### Contacts

If you find any error in the dataset or you have questions, please contact

	Manlio De Domenico
	Universitat Rovira i Virgili 
	Tarragona (Spain)

email: <manlio.dedomenico@urv.cat>web: <http://deim.urv.cat/~manlio.dedomenico/>