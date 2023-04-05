# Clustering of domain names generated from DGA

## Introduction
### Context

The great potential offered by the Internet is not without
drawbacks and dangers. As the number of people
connected to the network, the number of targets that an
attacker can go to threaten. In fact, the number of attacks
in the last period has significantly increased, not only to machines
private machines, but also to systems belonging to public and governmental bodies.
This phenomenon has given hackers the opportunity to easily create
networks of zombie computers, which could be exploited to carry out
criminal actions, and disrupt crucial services. Moreover, it is very
difficult to detect these networks, since the owners of the machines
realise too late that they have been the victim of an attack.

Nowadays, each server, or service, on the Internet is identified by a
unique address (the IP address) that allows the user to locate it on the
web, make requests to it, and obtain information in response.
However, requiring the Internet user to use IP addresses directly
would be far too complex. To make browsing a more user-friendly process
process, the _Domain Name System_, or DNS, was devised.
is a system used to assign names to nodes on the network, which
become alternative identifiers to the IP. More specifically, the DNS is a
distributed database in which each IP address is associated with a domain name.
domain, in the simplest case. Domain names are generally
strings separated by full stops and organised in levels. Of these, the most important
most important is the string that is encountered starting from the
right, which is called the _Top Level Domain_ or TLD for this. This is followed by the
second level domain, which must comply with formal rules
imposed, such as having a certain length and containing only characters belonging to a certain set.
characters belonging to a certain set. The third level in turn is called
third-level domain and in the general case may also be followed by
higher levels. When a user types in the domain name assigned to
a node on the network that he wants to reach to make a request,
a process called DNS Resolution takes place, through which the user
obtains the IP address associated with that domain name, to which he can then
connect to.

With the ever-increasing use of the Internet, over the years it has spread
malicious codes capable of damaging PCs used mostly by non
experts. If all these actions remain confined to a single PC, in the
worst cases, they only harm the victim, leading to the loss
or partial loss of their data. In addition to this type of attack, however,
there is also another technique: instead of infecting a single computer, an attempt is made
to infect several computers simultaneously. This type of attack
This type of attack leads the hacker to set up networks of 'zombie computers', called _botnets'.
zombies', called _botnets_, which can be exploited to carry out illegal activities even against third
against third parties, and not directly against the owners of the infected computers.
infected computers. The person in charge of managing the _botnet_ , who is
called _botmaster_ , sends commands via the so-called C&C
( _Command and Control) Channel_. This component of the _botnet_ is
fundamental, as it allows the attacker to communicate with the
infected machines and receive data from them. For this reason, without this
component, the _botnet_ becomes useless, as the _botmaster_ loses contact
with the controlled machines, which are no longer able to carry out
attacks independently. Therefore, closing the control channel of the
_botnet_ essentially means neutralising it. Another
relevant characteristic of botnets is their structure, on which the way the
which the _botmaster_ communicates with the other nodes. It may be:

- Centralised, when the command and control server is unique, and
    all the machines belonging to the _botnet_ are connected to this
    single node from which they receive commands.
- Decentralised, when each _bot_ that is part of the _botnet_ can
    be used both as a client and as a command and
    control.
- Hybrid, when it combines the two approaches.

The domain to contact the C&C server of the _botnet_ can be inserted in
hard-coded_ within the code of the _malware_ or in configuration files
configuration files that are downloaded into the _bot_ together with the _malware_.
This, however, ensures that if that contact node drops, it is not
possible for the _botnet_ to update itself to contact a new node. To
prevent this inconvenience, the idea was to make
were the _bot_ themselves to periodically generate a set of domain names
domain names, among which would be found the one that would allow the C&C
server. The algorithms with which the _bots_ manage to generate these domains are
called _Domain Generation Algorithm_ (DGA). With these algorithms, a list of domain names is
generated a list of pseudo-random domain names, which will be valid
for a certain period of time. At the same time, the _botmaster_ , who knows in
in advance how the names are generated, will use one in that time frame for its
time for its C&C server; once this time interval has expired, the
the regeneration of the domains, and the process starts again.
However, DGA have also have disadvantages, including the fact that the generation of domains is not
completely random. Like any process in computer science that generates
a random output, the generation of domains also depends on a
"seed", or at least on a pseudo-random mechanism. Therefore, by doing
_reverse engineering_ of the domain generation algorithm within
a _bot_ , it is possible to try to predict in advance the names that will be
generated, by taking preventive actions to obscure in advance the
domains in advance.

### Goal

The objective of the work presented here is to study how domain names are structured
domain names produced by various DGA families. To do this, it was decided to
follow the unsupervised learning approach, in particular
clustering. The idea is to try to understand whether there are similarities between the
domain names of certain malware families, so that we can
families into a single set and simplify the detection
of intrusions.

The work carried out followed the following steps:

- Identification of the sources to be used and consequent construction of the
    dataset to be studied
- Study of the samples in order to understand how the
    dataset
- Cleaning of noise or data not interesting for analysis
- Choice of the most appropriate _embedding_ technique among those available
- Choosing a clustering algorithm that meets the criteria defined above.
    previously defined.
- Design and execution of the actual analysis.

Thanks to this analysis, new information will be available on the various
families, so that the classes of interest can be studied individually in order to
so as to understand their behaviour as the parameters change.

## Materials and methods

### Description of the dataset

The dataset used for the clustering task was constructed from
from lists of malicious domain names obtained from _Bambenek
Consulting._ This organisation provides a daily list of
domain names, produced by DGAs, detected on the network and which are
classified according to the family of DGAs that generated them. For each file
is also provided a time estimate of the validity of the domain names in
it. In fact, as described in the introduction, domain names
domain names produced by DGAs have a limited life, after which they are
replaced. For this reason, _Bambenek Consulting_ inserts a domain name
domain name into the feed for five consecutive days, from two days
before it is generated up to three days after generation. In this
way, the feed allows those who obtain it both to record attacks in progress
and to analyse an attack that occurred in the previous days. This
peculiarity of the _feed_ provided by _Bambenek_ is handled in the ETL phase. I
feeds are structured as in Figure 1 and were collected for the period
from 20/07/2022 to 5/10/2022. All these files were then read to
obtain the accumulated data needed for the dataset.

### ETL

From the _feeds_ of _Bambenek_ , to build the required dataset it was
it was necessary to process all the feeds collected over time in such a way as to
obtain a dataset in which each row contained a domain name and
the associated class label. For reading and managing the data, the
the functionality of _PySpark_ , the implementation of _Spark_ on the
programming language _Python_.

First, having to work with many feeds collected over a
long time span, it was decided to implement a process of
incremental construction of the dataset. In fact, as the
data were collected from the _Bambenek Consulting_, at predetermined time intervals
predetermined intervals, these were processed and cleaned, so that
could become part of the final dataset for clustering. Thus,
every five days, the process of cleaning the
data within the five feeds and then merged them with the previously processed data into a single file.
previously processed into a single file.

As the domains were queued to form the overall dataset
the overall dataset, it was necessary to eliminate duplicates that might
duplicates that may have existed between the collected feeds, in order to have a dataset free of repetitions. For
this, duplicates existing in the five feeds considered at the time of
time of processing, given the possible presence of the same name
in several successive feeds of _Bambenek_. Subsequently,
obtained the list of domain names without repetitions, associated to the five
processed feeds, we proceeded to eliminate from this the domain names
already present in the single file. In fact, in the last feed of the series processed
the previous step, it cannot be excluded that there are names just
inserted, which will therefore also be inserted in the feeds processed in the current step.
current step. In this way, the final dataset has no duplicate elements
duplicates.
As far as individual feed processing is concerned, first a
assigned to each domain name a class label that
corresponds to the family extracted from the name description. In this
procedure takes into account the fact that DGA families may
have several names, which are aliases of each other; for instance, _wiki25_ is an alias
of _cryptolocker_, while _ptgoz_ is an alias of _zeus-newgoz_. In these cases, the
class labels have been modified to have in the dataset a
one-to-one correspondence between class label and family.
Subsequently, the columns relating to the description of the
name and manual reference, retaining only domain name and
class label. At this point, each domain name was replaced
by the bigrams obtained from the string formed by the second level domain and the
top level domain, concatenated without a space. For example, _mionome.com_
is transformed into _mionomecom_ and subsequently replaced with
the list of bigrams [ _mi_ , _io_ , _on_ , _no_ , _om_ , _me_ , _ec_ , _co_ , _om_ ].

The dataset thus constructed, where each row contains two fields, namely
the class label and the list of bigrams, is saved in _csv_ format and
structured as in Figure 2. It consists of approximately three million records
approximately.

### Embedding

In order to perform clustering, we need an _embedding_ layer that
converts the input into vectors that the algorithm can process.
In fact, the _embedding_ consists of a mapping of the input, through which
representing discrete variables (in this case domain names)
as continuous vectors that can be placed in a vector space.
vector space.

There are several types of _embedding_ , including random, in which the
randomly associating the characters in the entire dataset with a
random number and then transform all the domain names in the
dataset into numeric vectors, with the mapping defined. However, although this
mode is very simple and fast, performance is often not
optimal. This is why another type of _embedding_ was chosen, namely
FastText, which is an extension of an architecture, _Word2Vec_ , released by
Facebook AI Research in 2016. It is an open-source library whose
goal is to learn by itself, given a large corpus of sentences, the
continuous vector representations for the words contained in the text.

Compared to other techniques, FastText is known in that, to create the vector
to be associated with a word, it considers the internal representation of the
word itself, dividing it into n-grams. In fact, having taken a word, after
dividing it into n-grams, it obtains the vector representations of the
n-grams that form it, and then combines them in order to obtain a continuous vector
continuous vector that represents a word in space. In this way we
better representations of words are obtained and it is possible to construct
representations of words never seen in the dataset, as a sum of
n-grams. In addition to this, FastText has the advantage of being able to
customise the training phase by choosing various parameters and the training corpus
training corpus, which has proved very useful in the domain
names. For these, in fact, it is not possible to detect a meaningful context to
exploit for vectorisation, rendering other techniques of
pre-trained and word-based _embedding_ techniques useless.

The result of FastText training is a model containing all
word representations, found in the training corpus, as
𝑛-component vectors. Such models can be used in
_embedding_ to translate words into continuous vector representations.
In addition, FastText provides two different models to calculate the
word representations: _skipgram_ and _cbow_. The _skipgram_ model
learns to predict a target word from a neighbouring word. Instead,
the _cbow_ model predicts the target word based on its context, which is
represented as a word container, inside a window with
fixed dimensions, around the target word.
In the experiment, therefore, FastText was used as the
_embedding,_ and whole words were not used, but rather bigrams,
creating a training corpus with all the vectors of bigrams
associated with the names in the dataset. The corpus has as many rows as there are names
in the dataset, and each row contains the bigram representation of the
domain name. The
_skipgram_ , because there are no substantial differences in
execution time with the _cbow_ model. Furthermore, it is not very consistent to use the
c _bow_ , because it is not possible to identify a sensible context for the
bigrams, within a domain name. Following training,
each word is represented by the vector resulting from the
concatenation of the individual vectors associated with the bigrams that
composing it. This vector, due to the isomorphism between ℝ𝑚∙𝑛 and
𝑀𝑚×𝑛(ℝ), is equivalent to the matrix that has as columns the vectors
associated with the various bigrams. Given the greater efficiency of the calculations to be
performed with the vector representation, it was decided to concatenate the
vectors rather than working with a matrix representation.

### Methods and metrics

Among the various clustering techniques, the _Density-Based
Spatial Clustering of Applications with Noise_ (DBSCAN) was selected. The DBSCAN is a
density-based methodology: the cluster is defined by the points that are
part of it, and not by a particular point acting as a prototype, as in other
algorithms (e.g. K-Means). A density-based clustering method was chosen
based on density for two reasons:

- Having collected data from the network, it is not known a priori
    the number of classes within the observation period, which
    makes it unfair to use an algorithm in which one specifies a priori the
    number of clusters.
- The distribution of points constructed by FastText's _embedding_ is
    unknown, and it is more correct to rely on an algorithm that does not make assumptions
    a priori on the shape of the clusters in space, such as DBSCAN.

The DBSCAN algorithm bases the construction of clusters on two fundamental parameters
which are usually referred to as **_minPoints_** and **_eps_** **(ε)**.
Generally, points are classified in DBSCAN as **_core points,
border points_** and **_noise points._** _Core points_ are those points that have in their
around their radius _eps_ at least other _minPoints_ points. Border points_ are
those points which are in the neighbourhood of at least one _core point_, but are not
themselves _core points_. Finally, the _noise points_ are the remaining points (and therefore
outliers).

The operation of the algorithm is thus as follows:

- One starts from any point in the dataset. The choice is totally
  arbitrary and affects neither the result nor the efficiency of the algorithm.
- The radius _eps_ of the chosen point is checked.
    -- If there are not at least _minPoints_ points, then we move on to the
       next point.
    -- If, on the other hand, there are a number of points greater than or equal to
       _minPoints_ , then a new cluster has been found and the point is
       then classified as a _core point_. The same
       check on all points found around the _core point_. If
       at least _minPoints_ points are found in their surroundings, then those
       are themselves _core points_ and this process continues with the
       points in their surroundings. Otherwise, the point is classified as
       _border point_. In any case, all these points contribute to the
       formation of the identified cluster. The process is completed
       when there are no more new _core points_, resulting in a
       cluster.
- We move on to a new point not previously checked, and we
  repeat the process until all points have been checked. All those points
  that are classified neither as _core_ points nor as border points are classified 
  points nor as border points are classified as noise points, and
  will not properly be part of any cluster.
In the clustering algorithm, as a metric for measuring the distance between
vectors, associated with words, the Euclidean distance is used, since it is computationally
is the one that is computationally more immediate and easier to understand.

For the evaluation of clustering, special metrics were used,
both supervised and unsupervised, exploiting functionalities provided by the _sklearn library.
made available by the _sklearn_ library. The three supervised metrics are:

- The **homogeneity,** or **_homogeneity_** : a segmentation is
    perfectly homogeneous if each cluster has samples belonging
    to only one class. This measure is exploited because it takes into account
    the distribution of elements in the classes and the entropy.
- The **completeness,** or **_completeness_** : a segmentation is
    perfectly complete if all samples belonging to a certain
    class are in the same cluster. Like the previous one
    this takes the class distribution and entropy into account.
- The **_v-measure_** : represents the harmonic mean between homogeneity and
    completeness

The unsupervised metric chosen is the _silhouette_ , which expresses, given a
point 𝑥 of the dataset, whether it has been assigned to the correct cluster,
or should be assigned to another cluster. This metric gives for
each point a measure of the goodness of segmentation. It for each
point calculates how close it is to elements belonging to its own
cluster, and how far it is from elements in other clusters. The _silhouette_ ,
defined by 𝑎(𝑥) the average intra-cluster distance referred to point 𝑥 and by
𝑏(𝑥) the average inter-cluster distance referred to point 𝑥, it is calculated as:

```
𝑏(𝑥)-𝑎(𝑥)
max(𝑎(𝑥),𝑏(𝑥))
```

In general, the _silhouette_ , for a given point 𝑥, must be close to the
value 1, which indicates that point 𝑥 is very close to the elements of its
cluster and very far from the elements of others. Otherwise, if it is
negative, the clustering is incorrect, and if it is null, it cannot be said whether the
clustering is correct or not. For the evaluation of experiments
especially the initial ones, the relevant _sklearn_ function will be used which
calculates the average value of the _silhouette_ for all points, including those in the
noise cluster.
For the evaluation of the individual segmentations, two other supervised metrics were also considered
two other supervised metrics were also considered:

- The _precision_ , which assesses how much a cluster is individually
    homogeneous. It is given by the ratio of the elements of family _j_
    in cluster _i_ to all elements in cluster _i_.
- The _recall_ , which measures the ability of the cluster to represent a
    family. It is given by the ratio between the elements of family _j_ in the
    cluster _i_ and those belonging to family _j_.
    
## Experimental results

### Experimental setup

Defined the _embedding_ technique to derive a feature vector for each
biggram in the dataset considered, and chosen the clustering algorithm
DBSCAN clustering algorithm, a grid of experiments was defined on the basis of the parameters
to be set in both the _embedding_ and the clustering algorithm. I
free parameters, which needed to be tuned in order to
identify the best clustering, are the following:

- _epochs_ : the number of epochs for the training of the
    _embedding_ based on FastText.
- _dim_ : the size of the feature vector associated with a bigogram
    of the training corpus.
- _minPoints_ : the minimum number of points that must belong
    around a point under consideration in the DBSCAN algorithm, in order for
    such a point is considered as a core point.
- _eps_ : the radius of the hypersphere representing the neighbourhood of a point
    in the DBSCAN algorithm.

Based on the FastText documentation and the best practices it
suggested by it, it was assumed to conduct the experiments by making the
parameter _epochs_ the values ten, fifteen and twenty, and the parameter _dim_ the values
[100, 150, 200, 250, 300]. As for the parameter _minPoints_ ,
called 𝑛 the number of features, referring to the rule of thumb that
dictates:

- 𝑚𝑖𝑛𝑃𝑜𝑖𝑛𝑡𝑠≥ 𝑛+ 1 for datasets with moderate noise
- 𝑚𝑖𝑛𝑃𝑜𝑖𝑛𝑡𝑠≥ 2 ∙𝑛 for very noisy datasets

it was decided to evaluate values of _minPoints_ in the range

[𝑛, 6/5 ∙𝑛, 7/5 ∙𝑛, 8/5 ∙𝑛, 9/5 ∙𝑛 , 2∙𝑛]

The number of features 𝑛 depends on the
size of the vectors extracted by FastText, _dim_ , and the maximum length
maximum length of the strings in the dataset, _maxLen,_ according to the
relationship 𝑛=𝑑𝑖𝑚∙𝑚𝑎𝑥𝐿𝑒𝑛. The idea behind the choice of interval,
not knowing a priori what the noise or sparsity of the points in the
dataset, is to try to explore a search space in which there are
solutions as robust to the noise as possible. Finally, it was decided to
determine the value of _eps_ for each combination of the three parameters
previous three parameters, using an empirical technique, in order to reduce the
size of the search space. This technique consists of calculating the
distances between all the points in the dataset and select for each of them the
k-th nearest element, assumed 𝑘=𝑚𝑖𝑛𝑃𝑜𝑖𝑛𝑡𝑠. The distances to the
k-th element of each point are then graphed in order
ascending (or descending) order. The most suitable _eps_ for that _minPoints_ is to be
searched for in the region in which the graph presents an abrupt change
of curvature.

Having defined this experimental setup, the cardinality of the
dataset and the size of the feature vectors obtained for each word
within the dataset. Given the computational overhead of the algorithm
DBSCAN algorithm, amounting to 𝑂(𝑛^2 ), it was found that there was not a powerful enough
a machine powerful enough to carry out the experiments.
Therefore, the search space was revised by determining new ranges
of free parameters, suitable for the memory and computing power limits of
a _general purpose_ machine with 12 GB of RAM. Hence, we redefined
the variation range of the _dim_ parameter as [2, 4, 6, 8, 10] and determined a
determined a series of variation intervals for the parameter _eps_ , all

with the form [1/5 𝑒𝑝𝑠𝑚𝑎𝑥, 2/5 𝑒𝑝𝑠𝑚𝑎𝑥, 3/5 𝑒𝑝𝑠𝑚𝑎𝑥, 4/5 𝑒𝑝𝑠𝑚𝑎𝑥,𝑒𝑝𝑠𝑚𝑎𝑥], where:

- For 𝑑𝑖𝑚= 2 , 𝑒𝑝𝑠𝑚𝑎𝑥 = 2,25
- For 𝑑𝑖𝑚= 4 , 𝑒𝑝𝑠𝑚𝑎𝑥 = 3
- For 𝑑𝑖𝑚= 6 , 𝑒𝑝𝑠𝑚𝑎𝑥 = 4
- For 𝑑𝑖𝑚= 8 , 𝑒𝑝𝑠𝑚𝑎𝑥 = 4,5
- For 𝑑𝑖𝑚= 10 , 𝑒𝑝𝑠𝑚𝑎𝑥 = 5

In the previous range, the value 𝑒𝑝𝑠𝑚𝑎𝑥 represents the maximum value
of _eps_ linked to a certain dimension, and is chosen in such a way as to
respect the computational limits of the available machines. The reason
reason for choosing different ranges of _eps_ , varying the
value of the parameter _dim_ , depends on the fact that varying _dim_ varies the
number of features of each element of the dataset varies. If _dim_ increases, the
the number of features of the dataset elements also increases, which leads to an
increase in the distances between points in the space ℝ𝑛. Thus,
choosing the same interval for each dimension could have caused
a degradation of the results, as the use of too small or
too large of _eps_ leads to degenerate situations. In such cases it either occurs
that the entire dataset is included in the cluster noise or a
single matching cluster with the entire dataset.

In addition to the definition of the new variation intervals of the free parameters
the cardinality of the dataset was also resized,
reducing it from three million elements to approximately one hundred thousand elements. This
operation was not only done in order to reduce the load of the data structures in
memory, but also to adjust the size of the dataset to the
_minPoints_. In fact, by reducing the number of features of the vectors extracted by
FastText for each biggram, the parameter _minPoints_ is also reduced, which
but at the same time must always have a value that is
at least two orders of magnitude smaller than the cardinality of the dataset.
Therefore, when reducing the cardinality of the dataset, this aspect was also considered
this aspect, thus operating a _undersampling_, in such a way as to
respect the initial distribution of the population, and thus satisfying
both the computational limits and the constraint on _minPoints_.

Completed the definition of the free parameter variation intervals,
a _Python_ algorithm was developed to realise the process of
clustering of domain names. This algorithm comprises three steps:

- **Data pre-processing** : the dataset is collected, all class labels are extracted
    all the class labels and assign each of them an
    integer number. Then the maximum length
    of the strings in the dataset.
- **Embedding** : the _embedding_ process, defined in the
    section of the same name, and, having obtained the feature vectors for each word, we
    apply a _padding_ of zeros to each vector, in order to provide
    the clustering algorithm with vectors of equal length.
- **Clustering and metrics calculation** : the DBSCAN clustering algorithm is applied
    clustering algorithm DBSCAN and then calculate metrics for the evaluation
    of the segmentation, i.e. homogeneity, completeness
    v-measure and average silhouette.

### First experiment

Having defined the experimental setup and the series of experiments to be
conducted, the designed algorithm was run for all the parameter values
chosen, obtaining the results of the metrics presented in Figure 3, Figure 4,
Figure 5. The figures are heat maps in which blue represents the
highest values, green the lowest, and grey boxes the missing values
due to degenerate cases. Comparing the results obtained in the three figures for
the three values of the parameter _epochs_ , it is immediately apparent that as the parameter increases
parameter there is no significant increase in the results, but there is
certainly an increase in execution time. This increase
does not affect the size of the dataset in question, but for large datasets it becomes significant.
large datasets it becomes significant. As the parameter _dim_ changes, however, there are
significant variations are observed for all three metrics
considered, i.e. homogeneity, completeness and average silhouette of the
clusters.

Analysing only the average _silhouette_, it can be seen that, as the
the number of features extracted by FastText, for each biggram, the quality
of the clusters worsens with the same values of the parameters _minPoints_ and
_eps_. Moreover, considering a single value of the parameter _dim_ at a time,
it can be observed that in general, the quality of the clustering improves
as the value of the parameter _eps_ increases. On the other hand, when varying the
parameter _minPoints_ , it is not possible from these maps to record
a clear trend towards an improvement or worsening of the
average _silhouette_.

Due to the above, the relatively low value of the _silhouette_ in the
different cases, indicating a poor quality of segmentation, is
due to the fact that the finite points within the noise cluster were poorly
clustered.

Looking at the results obtained, it was noted that between values 2 and 4 for the
parameter _dim_ had the best possible silhouette values for all the
experiments performed, acceptable in light of the considerations made on the
noise. Therefore, it was decided to study further in the range
consisting of the integer values [2,3,4]. Firstly, taking into account the
trends recorded with the first experiments, the algorithm was re-run for
the value of the parameter _dim_ equal to three, considering only the _eps_
as high as possible, and values of _minPoints_ in the same range
used previously. The process yielded the results of the silhouette
average shown in Figure 6.

In the figure it is clearly observable that as the value of the
parameter _minPoints_ , the average silhouette has a slight decrease.
On the other hand, overall, the tendency of the silhouette to decrease is confirmed,
as the number of features extracted by FastText increases for each
biggram. In this case, it was not necessary to investigate all possible
values of _eps_ , for dimension three, as it represented the only
integer value present between two and four, and the trend was
sufficiently clear from previous results.

Once the best clustering was identified, the metrics were evaluated for all the
values of the free parameters considered the supervised metrics, i.e.
homogeneity and completeness, shown within the maps in Figure 7 and
Figure 8. In these, the metrics are shown only for _epoch_ =20, as
for the other values they do not vary much.

Looking at the homogeneity and completeness values obtained, it can be seen that the
best trade-off between the two metrics is obtained at values
values of _dim_ ranging from two to six, while for the values thereafter, we obtain
worse results overall.

Taking into account that the best segmentation is obtained for values
of _dim, minPoints_ and _eps_ defined above, and that for the same values
a good compromise is obtained for the supervised metrics, it was decided
to conduct more precise analyses for these parameter values.

### Best segmentation analysis

Having identified the intervals, or free parameter values, in which the
best values of the average silhouette, it was decided to conduct the
experiments for those values, calculating some more in-depth metrics.
These include the precision and recall for each cluster-class pair,
the confusion matrix for clustering and the silhouette of each sample.
In particular, the confusion matrices proved very useful for
identify some relevant phenomena observed in the experiments.

Considering the first experiment, conducted with the parameter values
free 𝑑𝑖𝑚= 4 ,𝑒𝑝𝑜𝑐ℎ𝑠= 20 ,𝑒𝑝𝑠= 3. 0 ,𝑚𝑖𝑛𝑃𝑜𝑖𝑛𝑡𝑠= 189 , we determined
the clustering confusion matrix visible in Figure 9, in which
six clusters are detected, in addition to the _cluster_noise_. The first important thing
to emphasise is that all elements of the class " _bazarbackdoor_ " are
isolated within the _cluster_ 5_.
This means that the _cluster_5_ is representative of the class in question,
as also evidenced by the high precision and recall values compared to the
malware family _bazarbackdoor_ , visible in Figure 10, and
in Figure 11. On the other hand, considering the _monerodownloader_ class, it can be seen
that all elements are evenly distributed within the _cluster_1_ ,
the _cluster_2_ , the _cluster_3_ and the _cluster_4_. These clusters contain mostly
mostly only elements of the malware family under investigation, and therefore
have very high precision values with respect to it, making them
as very homogeneous clusters. On the other hand, since the samples
distributed in several clusters, it results that these clusters are not
representative of the _monerodownloader_ class, generating values of
recall values that are not high. For the other families, it is observed that all elements to
belonging to them, converge in one cluster, namely the _cluster_0_. It
is very complete with respect to the individual classes, because it
contains most of the samples, but not very homogeneous, because it is not
is representative of a single family, as can be observed in Figure 10
and Figure 11. Exceptions to the last commented case are the
families _dyre_ and _zeus-newgoz_ , whose samples are mainly placed
within the _cluster_noise_ , together with a few samples from the other classes.
In addition to these, the _cluster_noise_ contains all the elements
families with a cardinality in the dataset below the
parameter _minPoints_.
Moving forward, the second experiment was conducted with the values of the
free parameters 𝑑𝑖𝑚= 3 ,𝑒𝑝𝑜𝑐ℎ𝑠= 20 ,𝑒𝑝𝑠= 2. 75 ,𝑚𝑖𝑛𝑃𝑜𝑖𝑛𝑡𝑠= 142 ,
always calculating the values of _precision_ and _recall_ for each pair of
cluster-class and the clustering confusion matrix visible in
Figure 12. These results are similar to the previous ones, in that the
obtained the same number of clusters, with some differences with respect to the
segmented families.

As before, the _bazarbackdoor_ family is isolated entirely
within a single cluster, while the elements of the family
_monerodownloader_ family are divided into several clusters. Unlike before, the
domain names _monerodownloader_ are only divided into three clusters, namely
_cluster_4_ , _cluster_3_ and _cluster_2_ , the latter of which has a cardinality
greater cardinality than the other two. In fact, while the precision values are high
for all three, as can be seen in Figure 13, we can certainly see that the
value of the _recall_ with respect to this family is much higher for the _cluster_2_ ,
than the others, as shown in Figure 14. Unlike
to the previous experiment, it can be seen that the _cluster_1_ is very homogeneous
with respect to the _dyre_ family, containing only samples of the same. At
At the same time, however, the cluster is not representative of the family, as it
contains only half of the elements, generating a _recall_ value that is not
high. Finally, as in the previous case, it is noted that the rest of the
families are predominantly within the _cluster_0_ , with the exception of the _zeusus_ family.
except again for the _zeus-newgoz_ family, whose elements are
classified as noise, and for some elements of the _dyre_ family which
merge into the _cluster_noise_.

Then, for the last experiment, free parameter values were set
𝑑𝑖𝑚= 2 ,𝑒𝑝𝑜𝑐ℎ𝑠= 20 ,𝑒𝑝𝑠= 2. 25 ,𝑚𝑖𝑛𝑃𝑜𝑖𝑛𝑡𝑠= 95 , and calculated the
same metrics as in the previous experiments. For this experiment we
slightly different results from the other two, as shown
also in the confusion matrix in Figure 15. First of all, the first thing
that can be noticed, at the macroscopic level, is that the number of clusters
identified goes from six to three.
In detail, the _cluster_1_ contains almost all the elements of the _dyre_ family,
resulting very homogeneous and complete with respect to the family, as
demonstrated by the high values of _precision_ and _recall_ visible in Figure 16 and Figure 17.
Similarly, the _cluster_2_ is very representative of the
family _monerodownloader_ , which was previously
broken down into three different clusters. This cluster also absorbs all
elements of other families such as _enviserv_ , _infy_ , _kingminer_ and
_pandabanker_. In contrast, in this case, the samples of the family
_bazarbackdoor_ family fall into the _cluster_0_ , within which nearly
all malware families in the dataset, even some very numerous ones
such as _flubot_ , _murofet_ , _necurs_ and _qakbot_. Finally, as before, also
in the current experiment the case of the _zeus-newgoz_ family stands out, which
is completely included in the _cluster_noise_. Although this
might seem a negative factor at first sight, another key to
reading could be to consider the _cluster_noise_ as an
alternative cluster, whereby this family would be segmented in a
discrete.

Finally, for all experiments, the silhouettes of each
sample in the dataset, in order to establish more precisely the quality
of the segmentation, either by considering elements of a single family
or by considering entire clusters. In fact, by investigating the silhouettes in more depth
of each sample, one can clearly see the effect that the
silhouettes of the elements in the dataset that feed into the _cluster_noise_.
As shown in Figure 18, it can be seen that for all experiments the
majority of the elements belonging to the _cluster_noise_ , have a
silhouette between -0.2 and 0.1, with a small number of elements tending to
have a silhouette close to -0.5.
Thus, most elements belonging to the _cluster_noise_
has a silhouette value that is either very low or such that it is not
can be indicated whether they have been placed in the correct cluster or not.
In addition, since approximately one third of the elements in the dataset end up in the _cluster_noise_, these
elements in the dataset, these silhouette values tend to have a
strong influence on the average value, which is significantly reduced. At
contrary considering the _cluster_0_ , which in all experiments proved to be
proved to be the one that absorbs the most elements within the
dataset, we verify that the silhouette values associated with each element
are satisfactory. In fact, as shown in Figure 19 it can be seen that
most of the elements within the _cluster_0_ are associated with a
silhouette value of 0.5 on average, with very few outliers having
a negative or null value. Thus, despite the fact that it contains
many elements from different families, the quality of the cluster under consideration is
discrete for the parameter values considered.
In addition to the individual clusters, the silhouettes of the
elements belonging to individual malware families, in particular
_bazarbackdoor_ and _monerodownloader_. Starting with the former, as
visible on the left in Figure 20, it can be seen that most of the elements
have associated silhouette values that are more than satisfactory, testifying
that the quality of the _cluster_5_ , in which these elements are collected, is very
high. Not all _bazarbackdoor_ samples in the dataset converge
into the _cluster_5_ , as 66 elements are included in the _cluster_noise_ , as
seen in Figure 20 on the right, which are associated with negative silhouette values
negative, testifying to incorrect segmentation.

In the case of the _monerodownloader_ family, in the experiment performed with
𝑑𝑖𝑚= 4 ,𝑒𝑝𝑜𝑐ℎ𝑠= 20 ,𝑒𝑝𝑠= 3. 0 ,𝑚𝑖𝑛𝑃𝑜𝑖𝑛𝑡𝑠= 189 , the elements of the
dataset are distributed in four clusters, so the distribution of the
majority of the samples in it is associated with values
positive values of the silhouette, on average the silhouette is not very high. The
majority of the elements contained in these clusters have a value
of the silhouette around 0.4, with some even settling around
0.3. This testifies to the fact that probably, although the elements
within the clusters are very close, however the four clusters
identified are not very far apart in space, referring to
referring to elements of the same family.

### Analysis of feature averages and variances

Having identified well-segmented families for each dimension, one can
can proceed with their study in order to detect any patterns. In
Specifically, the averages and variances of the families considered can be analysed
taken into consideration, as the dimension varies. In order to visualise
effectively, the mean and variance will be considered separately.
features of the mean and variance vectors. Depending also on the dimension
considered, each feature relating to a bigogram will have its own colour in the
graphs presented later. The first feature will be represented
in red, the second in green, the third in blue and the fourth in cyan.

First, it is necessary to emphasise that by applying a
padding of zeros to equalise the length of the vectors independently
the length of the source words, the feature values are
always, from a certain point onwards, equal to 0. In addition, a relevant factor
for the trend of mean and variance is the length of the domain names.
If this is highly variable for a family, in fact, one will have names
with more non-zero features than others, and this will inevitably influence
the calculation of mean and variance for those specific features. These effects
are visible in the statistical trends of various families examined below.
below.

Considering the _dyre_ family, a regular trend of
mean and variance, except for the first and last values. In detail, for
dimension 2 (Figure 22), the variance of the first features (ca. 0.10) is higher than that of the second features (ca. 0.10).
higher than that of the second ones (about 0.02), and both remain stable.
In the case of dimension 3 (Figure 23), some changes are noticeable; in
in particular, the variances of the first and second features settle
on very similar values (around 0.05), while the third features have higher values
(above 0.10), while all still maintaining a regularity.

It is also interesting to note that the averages of the second and third features have very similar values.
features have very similar values. With regard to the length
of the words, in most cases these contain 35 bigrams.
Thus, this family of DGAs, constructs domain names of little variable length.
little variable. With 𝑑𝑖𝑚= 2 , the elements fall entirely within a
single cluster, as these are probably located close to each other.
However, as the size increases, the sparsity of the points increases and, as a
consequently, segmentation worsens and a representative cluster cannot be identified.
a representative cluster.The _monerodownloader_ family has a different behaviour, as
in that a representative cluster is obtained at the
dimension 2 (Figure 24), whereas in the other cases the samples are
distributed over several clusters. Considering the best segmentation, the
mean and variance values of the first features are significantly higher
than the second ones, but from the second half (after the 23rd feature) the values lose
regularity. This phenomenon also occurs with dimension 3 (Figure
25 ) and with dimension 4 (Figure 26), probably due to the
great variability in word length. In fact, it has been seen that the
domain of this family have four possible lengths, which is why
which is why the relative points are distributed in four different clusters.

As the size increases, the number of features per domain name increases
domain name, and thus the greater information content and the different
length of the names cause them to be distributed in more clusters.

Turning to the _bazarbackdoor_ family, which is correctly
segmented with both dimension 3 and dimension 4, a totally different trend from the previous cases appears.
totally different trend from the previous cases. The mean and
variance do not stabilise on one value, but change
their value as the features change. Focusing our attention
on dimension 3 (Figure 27), the averages of the first and third features
decrease with the same trend, while the average of the second
is initially stable, then peaks towards 0, then
decreases towards the values of the thirds (-1.0). Instead, all variances tend
to 0, although outliers are present. With regard to
dimension 4 (Figure 28), the averages of the first, third and fourth features
are regular, although the last values vary the most. Instead, the
average of the second features, although initially following the first, does not follow a regular
regular trend, as there are always noticeable fluctuations.
However, the values of the variances tend to decrease, as before,
and at the same time are an order of magnitude smaller than
than the averages.

In both dimensions, great variability is observed in the average
of the features of the _bazarbackdoor_ domain names, but at the same time the
variance values are small compared to the mean values. This,
combined with the fact that most of the domain names in the family
has a length of 12 bigrams, allows the elements to be segmented into
a single cluster. Only a few elements do not fall within
that set, having a shorter length of 10 bigrams and due to
due to the great variability of the average.
Another family that is analysed in detail is _zeus-newgoz_ , which
represents approximately one third of the dataset. Although the samples of this
family do not really fall into a cluster, it can be seen that they
are almost always categorised as noise. Therefore, this analysis aims to
verify the presence of a different trend from those observed that
justify this phenomenon. Indeed, the averages and variances do not
settle on precise values, but have a certain periodicity. In the case of
dimension 2 (Figure 29) and dimension 3 (Figure 30), the mean and
variance of the features have a wave-like trend until
they begin to tend towards 0. As for dimension 4 (Figure
31), we notice that the fourth feature (not present in the previous cases) has a
more regular trend: the mean and variance values are stable
(around 1.4 and 0.04 respectively), although the last values differ.
Since the lengths of the domain names belonging to this family
are highly variable, the samples are scattered and, consequently, a
density-based_ algorithm struggles to isolate them correctly.
### Analysis without correctly segmented families

Having identified some clusters with high values of _precision_ and _recall_ , one can
consider the part of the dataset that is not segmented correctly.
correctly segmented. Thus, one can exclude the samples of the families
previously identified and the same study can be carried out on the
remaining dataset. In fact, it may be the case that the training of
FastText is strongly influenced by those samples, so that the
of cluster construction is unbalanced towards a few families.

After removing _bazarbackdoor_ , _dyre_ and _monerodownloader_ for the
dimensions in which they were segmented correctly, we run
the FastText training and clustering algorithm again. In the
case where 𝑑𝑖𝑚= 2 , 𝑒𝑝𝑜𝑐ℎ𝑠= 20 , 𝑚𝑖𝑛𝑃𝑜𝑖𝑛𝑡𝑠= 95 ,𝑒𝑝𝑠= 2. 25 , only one cluster is
only one cluster is identified (Figure 32). The _cluster_0_ includes samples
from different families, some of which are quite numerous, such as _murofet_
or _necurs_ , and others of small size, such as _vidro_ or _mirai_. Within the
_cluster_noise_ are elements of several families, including _zeus-
newgoz_. It can be seen that _bazarbackdoor_ is included within the
identified cluster. In general, the values of _precision_ are low, while
those of _recall_ vary according to the different families.

Instead, with the configurations 𝑑𝑖𝑚= 3 , 𝑒𝑝𝑜𝑐ℎ𝑠= 20 ,
𝑚𝑖𝑛𝑃𝑜𝑖𝑛𝑡𝑠= 142 ,𝑒𝑝𝑠= 2. 75 (Figure 33), the DBSCAN produces two clusters
( _cluster_0_ and _cluster_1_ ). Within the first are samples
from different families, similar to the previous case, and therefore
the _precision_ of the individual families in this cluster is lowered
appreciably. In the second cluster there are only samples
belonging to the _dyre_ class, however, this family is also present in the
_noise_ cluster. In addition, many points from various families, including _zeus-
newg
In the case of 𝑑𝑖𝑚= 4 , 𝑒𝑝𝑜𝑐ℎ𝑠= 20 , 𝑚𝑖𝑛𝑃𝑜𝑖𝑛𝑡𝑠= 189 ,𝑒𝑝𝑠= 3. 0
(Figure 34), there is only one cluster, as was the case for dimension
2, in which the same families as previously seen are present. One can
underline the fact that the elements of the _dyre_ family are categorised
as noise, in which other families are also included, similarly to the
previous cases. Thus, this in-depth analysis shows that the
eliminated families do not adversely affect segmentation, which
probably depends only on variations in the free parameters, in particular
in particular _dim_ , _minPoints_ and _eps_.

## Conclusions

Through the analysis carried out on the available dataset it was possible to
the distribution of the samples and the presence of patterns within the
families, using the DBSCAN segmentation technique. The
different configurations adopted made it possible to detect the
characteristics of the families and their behaviour as parameters change.
parameters. In addition, the use of segmentation represents an
alternative technique to that typically used in this field,
i.e. classification, often with neural networks. One of the objectives of
of this work was to study how an unsupervised approach
how an unsupervised approach behaves in this domain in comparison to more classical
supervised approaches.

The results obtained from the experiments performed were negatively
influenced by the computational limitations encountered. In order to
more in-depth analysis of the distribution of families would
required much more memory than was available during the tests.
available during the tests. The first limitation is the value of the
size of the output vectors from the processing of _FastText_. In fact, the
default size proposed by _FastText_ is 128, but in many cases it is
required for this number to increase in order to improve performance.
However, considering that each single word in the dataset is composed of
several dozen bigrams, the memory occupied by the dataset transformed into
vectors, grows so quickly as the size increases that
it quickly becomes unmanageable. Similar limitations were
also found in the choice of the _eps_ parameter, chosen through the
heuristic technique illustrated above_._ Unfortunately, values of _eps_ were given
of _eps_ too large, which caused an overfilling of the
memory to construct the cluster representations.

In addition to increasing the available computing power, another possible
solution is the training and use of an
encoder: by finding a space equivalent to the starting space, but of a smaller
smaller size, one can decrease the number of features input to the
DBSCAN, so that the calculations can be simplified. In this way, in the
In this way, in the target space, clustering is less computationally
computationally, and consequently larger parameters can be used.
larger parameters. Of course, to return to the starting domain and verify the
result, the Decoder is required.


In conclusion, although better performance cannot be excluded
with more suitable parameters, the results obtained from this approach are not
encouraging. In fact, for different parameter combinations, not only did the
not only did the goodness of the segmentation vary, but the very
families that the algorithm was able to identify. If this phenomenon was
characteristic of the approach and was not caused by the limitations
found, it would mean that clustering is not a viable way forward.
practicable. Indeed, a combination of parameters would be required for each family to be isolated and, consequently, the process would be
too complex and not very adaptable to the appearance of new malware families in the future.

#### _Disclaimer: images in the text refer to 'Relazione BDA.pdf' file_
