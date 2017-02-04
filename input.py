from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from gensim.corpora import Dictionary, MmCorpus
import gensim
import pyLDAvis.gensim as gensimvis
import pyLDAvis
import pandas as pd
import numpy as np
import operator
from scipy.stats import entropy
from sklearn.cluster import KMeans
import networkx
import itertools as itt
import matplotlib.pyplot as plt
import json
from networkx.readwrite import json_graph
import flask
"""
def JSD_dist(LDA_a, LDA_b):
  LDA_a = np.array(LDA_a)
  LDA_b = np.array(LDA_b)
  eps = 1e-32
  
  M = 0.5*(LDA_a+LDA_b+eps)
  distance = np.sqrt(0.5*(np.sum(LDA_a*np.log(LDA_a/M+eps)) + np.sum(LDA_b*np.log(LDA_b/M+eps))))
  
  return distance
"""
def JSD_dist(LDA_a, LDA_b):
	_P = np.array(LDA_a)
	_Q = np.array(LDA_b)
	_M = 0.5 * (_P + _Q)
	return 0.5 * (entropy(_P, _M) + entropy(_Q + _M))

d = []
with open("input2") as f:
	for line in f:
        	y = line.rstrip()

        	x = y.split(",")
		d.append(x[0])
		ipstr = "LDA/" + str(x[0])
		with open(ipstr, "a") as myfile:
			for y in (0, int(x[2])):
				str1 = x[1] + " "
    				myfile.write(str1)
myset = set(d)
sortedset = sorted(myset)
d = list(sortedset)
print d
texts = []
for i in d:
	docstr = "LDA/" + i
	print docstr
	with open(docstr) as f:
		for line in f:
			y = line.rstrip()
			x = y.split(" ")
			texts.append(x)

for text in texts:
	print text

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
#print dictionary.token2id
corpus = [dictionary.doc2bow(text) for text in texts]
#print "******** Corpus *********"
#print corpus
MmCorpus.serialize('LDA/enterprise_traffic.mm', corpus)
dictionary.save('LDA/enterprise_traffic.dict')
ldamodel = gensim.models.ldamodel.LdaModel(corpus, id2word = dictionary, num_topics=7, passes=30)
result =  ldamodel.get_document_topics(corpus, minimum_probability=0.0)

#print result

#print "Done"

document_key = d
document_topic_matrix = []
for doc_id in range(len(corpus)):
    docbok = corpus[doc_id]
    #print docbok
    doc_topics = ldamodel.get_document_topics(docbok, 0)
    tmp = []
    for topic_id, topic_prob in doc_topics:
    	#print topic_id
    	#print topic_prob
        tmp.append(topic_prob)
    document_topic_matrix.append(tmp)

#print document_topic_matrix

document_key = d
document_topic = {}
for doc_id in range(len(corpus)):
    docbok = corpus[doc_id]
    #print docbok
    doc_topics = ldamodel.get_document_topics(docbok, 0)
    tmp = []
    for topic_id, topic_prob in doc_topics:
    	#print topic_id
    	#print topic_prob
        tmp.append(topic_prob)
    document_topic[document_key[doc_id]] = tmp


document_similarity = {}
for doc in range(len(document_topic)):
	
	l = []
	for doc_other in range(len(document_topic)):
		tmp = ()
		if doc == doc_other:
			similarity = 0
		else:
			similarity = JSD_dist(document_topic[document_key[doc]], document_topic[document_key[doc_other]])
		tmp = (doc_other, similarity)
		l.append(tmp)
	l.sort(key=operator.itemgetter(1), reverse=True)
	document_similarity[document_key[doc]] = l[:10]

document_similarity_matrix = []

print len(d)
print len(document_topic)
for doc in range(len(document_topic)):
	
	l = []
	for doc_other in range(len(document_topic)):
		if doc == doc_other:
			similarity = 0
		else:
			if doc < doc_other:
				similarity = JSD_dist(document_topic[document_key[doc]], document_topic[document_key[doc_other]])
			else:
				similarity = JSD_dist(document_topic[document_key[doc_other]], document_topic[document_key[doc]])
		l.append(similarity)
		print similarity
	document_similarity_matrix.append(l)

for i in range(len(document_similarity_matrix)):
	print document_similarity_matrix[i]
	print ""
#print "DOC Similarity"
#for doc in range(len(document_topic)):
#	print ""
#	print document_similarity[document_key[doc]]



#print "Doc Topic"
#print document_topic

df = pd.DataFrame.from_dict(document_topic, orient='index')
df.to_csv('csv')
ldamodel.save('LDA/enterprise_traffic.model')
vis_data = gensimvis.prepare(ldamodel, corpus, dictionary)
pyLDAvis.display(vis_data)


num_clusters = 13

km = KMeans(n_clusters=num_clusters)

km.fit(document_topic_matrix)

clusters = km.labels_.tolist()

print clusters
print len(clusters)
print d
print len(d)


documents_all = { 'doc' : d }
frame = pd.DataFrame(documents_all, index = [clusters], columns = ['doc']) 
frame.sort_index(inplace=True)
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    print frame

order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print "Cluster" + str(i)

    #for doc in frame.ix[i]['doc'].values.tolist():
    #	print doc
print ""
print ""

edges = [(i, j, {'weight': int(document_similarity_matrix[i][j] * 100)})
         for i, j in itt.combinations(range(len(d)), 2)]
"""
#************
edges = [(i, j, document_similarity_matrix[i][j])
         for i, j in itt.combinations(range(len(d)), 2)]
#print edges
documents_all = { 'doc' : d }
docframe = pd.DataFrame(documents_all, columns = ['doc']) 
#print docframe.reset_index().to_json(orient='records')
edgeframe = pd.DataFrame(edges, columns=['Source', 'Target', 'Weight'])
graph_json = {}
graph_json["nodes"] = docframe.reset_index().to_json(orient='records')
graph_json["edges"] = edgeframe.reset_index().to_json(orient='records')
#print json.dump(graph_json)
with open('LDA/network.json', 'w') as outfile:
    json.dump(graph_json, outfile)

#print edgeframe.reset_index().to_json(orient='records')
"""

k = np.percentile(np.array([e[2]['weight'] for e in edges]), 90)

G = networkx.Graph()
for i in range(len(d)):
	G.add_node(i, group=clusters[i])
#G.add_nodes_from(range(len(d)))
G.add_edges_from([e for e in edges if e[2]['weight'] > k])

#print G.edges(data=True)
dg = json_graph.node_link_data(G) # node-link format to serialize
print dg
# write json
json.dump(dg, open('force/force.json','w'))
print('Wrote node-link JSON data to force/force.json')

# Serve the file over http to allow for cross origin requests
app = flask.Flask(__name__, static_folder="force")

@app.route('/<path:path>')
def static_proxy(path):
  return app.send_static_file(path)
print('\nGo to http://localhost:8000/force.html to see the example\n')
app.run(port=8000)

"""
elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.8]
esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=0.8]

pos=networkx.spring_layout(G)

networkx.draw_networkx_nodes(G,pos,node_size=700)


# edges
networkx.draw_networkx_edges(G,pos,edgelist=elarge, edge_color='y',
                    width=6)
networkx.draw_networkx_edges(G,pos,edgelist=esmall,
                    width=1,alpha=0.5,edge_color='b',style='dashed')

# labels
networkx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')

plt.axis('off')
plt.savefig("weighted_graph.png") # save as png
plt.show() # display
"""