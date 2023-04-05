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
domain names produced by various ADI families. To do this, it was decided to
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
procedure takes into account the fact that ADI families may
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
