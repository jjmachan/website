---
layout: post
title: "Hangar Tutorial (1/2): Adding your data to Hangar"
category: MLOps
---

#### *A step by step guide to setting up a hangar repo and adding your data to it.*

Data is arguably the single most important piece in machine learning. The vast amount of data collected each moment is what fuels the deep learning algorithms, running on huge computational resources, to mimic any form of intelligence. Hence collecting, storing and maintaining this ever-growing data is important for training better deep learning algorithms.

This is where Hangar comes in.

**Hangar is version control for tensor data.** Its like git but for your tensors ie numeric data. Using hangar you can time travel through the evolution of the dataset, create different branches for experimenting without any overhead, collaborate with multiple people in your organisation in creating the datasets, create a local copy with only a subset of the data and many more. Hangar makes handling your datasets a breeze.

![hangar](https://miro.medium.com/max/1000/1*h3ZqORZA4VaHyd3DjaaPBQ.jpeg "Chill man...Hangar has got you back")

## Core Concepts

**1. Datasets**: This is the collection of data samples that are used for training. Each sample in a dataset is the smallest individual entity that still has meaning when considered individually. This sample can further be broken down in smaller attributes and each of these attributes can be stored separately. For example in the case of the MNIST dataset, the single 28x28 image is the sample but in the case of tabular datasets like the Titanic Dataset in Kaggle, each row is the sample and each of the columns are the attributes.

**2. Columns**: A dataset can be broken down into Columns and that is how they are stored in Hangar. It is similar to how columns in tables are. Each sample in the dataset is broken down into columns that represent the different attributes of that sample. Only when all the attributes (columns) are combined that the sample is described fully.

![An Illustration showing Hangar Columns](https://cdn-images-1.medium.com/max/800/1*Svkl5Q8xLHWfwC-IQf5ytg.png)

Hangar offers 2 types of columns

- Numerical Columns (ndarray): These are columns used to store numerical data in numpy like arrays.
- String Columns (str): These columns store data of type String.

Note: In versions before *Hangar 0.5*, columns used to be called arraysets and was used only to store numerical data while metadata, now called String Columns, were used to store strings.

**3. Repository:** This is where data is stored as a list of commits in time, in various branches by the different contributors that work on it. It is similar to a git repository. We typically use one repository per dataset. Each sample of data is stored in Arraysets.

If you want to know more about the concepts I would highly recommend you to check out the [docs](https://hangar-py.readthedocs.io/en/stable/concepts.html).

## Setup Repository and Arraysets

For this tutorial, we are going to setup hangar on the [MNIST dataset](https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz) (download the dataset and save it in your working dir). We will add the dataset to Hangar and in the process learn the basic functionalities and usage of Hangar. One important thing to note is that unlike git, where the main interaction with the versioning system is the CLI, here the main access point is through the python APIs. This makes dataset versioning work seamlessly with the large dataset pipelines.

Installation is simple, just run `pip install hangar` or `conda install -c conda-forge hangar` if you're using anaconda. For installation from the source just run

~~~
$ git clone https://github.com/tensorwerk/hangar-py.git
$ cd hangar-py
$ python setup.py install
~~~

### Initialize the Repository

Each time you work with a new dataset we have to create a new repository. We use the repository init() method to initialize a repository. This is where you provide Hangar with your name and email address that will be used in the commit logs.

<script src="https://gist.github.com/jjmachan/774bffddec1b6b2af4bbe2c9128e8a5f.js"></script>

This generates the repository and creates the underling files in a .hangar folder.

~~~
Hangar Repo initialized at: /home/jjmachan/jjmachan/hangar_examples/mnist/.hangar
Out []:
Hangar Repository               
    Repository Path  : /home/jjmachan/jjmachan/hangar_examples/mnist               
        Writer-Lock Free : True
~~~

Now we have to create a checkout of the repo to access the data in it. There are two checkout modes.

1. Write-enabled checkout: In this mode, all operations are performed on the current state of the staging area. Only one write-enabled checkout can be active at a given time and has to be properly closed before exiting.
2. Read-only checkout: This mode is used to only read from the repo. More than one checkout can be active at a given time.

<script src="https://gist.github.com/jjmachan/822ae9ae3b8817d93d1fc859ecdcb687.js"></script>

This creates a Write-enabled checkout.

~~~
Out[]:
Hangar WriterCheckout                
    Writer       : True                
    Base Branch  : master                
    Num Columns  : 0
~~~

As mentioned earlier a dataset is stored as Columns which contain the different attributes that describe a sample. To initialize a Column we need the *name, dtype(dataType)* and *shape*. We can also show it a *prototype*, which is a sample array with the correct *dtype* and *shape* as that of the sample data and hangar will infer the *shape* and *dtype* of the column from that.

A Numerical column is initialized by the `add_ndarray_column()` method while a new String column is initialized by `add_str_column( )`. It takes in the name and prototype of the sample.

In this example, we are versioning the [MNIST](https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz) dataset. To continue with the tutorial, download the dataset and place it in the working dir. Then run the following code to load up the dataset and create a prototype for the initializing the Column.

<script src="https://gist.github.com/jjmachan/1ce2b451fa6b023649ed3aeee95beb5a.js"></script>

Now we have the sample image (sample_trimg) and label (sample_trlabel) to initialize out columns.

<script src="https://gist.github.com/jjmachan/2c2c19ed6e03e89b5e419fb1a45aa9cf.js"></script>

~~~
Out []:
Hangar FlatSampleWriter                 
    Column Name              : mnist_training_images                
    Writeable                : True                
    Column Type              : ndarray                
    Column Layout            : flat                
    Schema Type              : fixed_shape                
    DType                    : float32                
    Shape                    : (784,)                
    Number of Samples        : 50000                
    Partial Remote Data Refs : False
~~~

### Adding the Data
When a column is initialized a Column accessor object is returned and we can access the samples in the columns using that object. We can also access the Column accessor object using the checkout instance for our repo.

`checkout[column_name]` -> Column

The sample data is stored in a dictionary-like fashion with a key and value. new samples can be added similar to how you add data in python dictionaries.

`column_1[key]` = value

The keys can either be *int* or *str*. The keys act as indexes to retrieve the corresponding samples. You can either use that or the function get( ) for columns which return None if no data was found instead of throwing a KeyExeption.

`column_1[key]`

`column_1.get(key)`

`checkout[column_name, key]`

A word about Context Managers in Hangar. Security is a big focus point for Hangar, right from day one. Hangar has lock and checks on the methods used to access the data stored so that no matter at what scale Hangar is running, the data is safe from getting corruptions. Each time we try to retrieve or set data, Hangar performs some checks and only after that the action is performed. This can turn out to be a big overhead when performing lots of read and write tasks. Hence to overcome this it is recommended to use python context managers using the `with` keyword. Using it makes sure that the checks are performed for each read/write session and not every single operation individually.

Now we have the final code to add our MNIST data to columns.

<script src="https://gist.github.com/jjmachan/59036f7f9a2c57337ee0e2d1eb47c04a.js"></script>

*As an exercise try to execute the code without the context manager ( with img_col, label_col: line) and see the performance difference.*

### Commit the Changes

After the data has been added you can simply commit the addition to the repo using the `commit()` method. It takes the commit message as an argument. After all the changes have been committed, make sure you close the connection especially when you are performing a write-enabled checkout.

<script src="https://gist.github.com/jjmachan/095d4a75043f90f2fdc892ca4e197262.js"></script>

~~~
Out []:
'a=3f4c4497c93982d0d182110277841ed37d1bbf8e'
~~~

And with that, our data is hashed and safely stored in Hangar.

To view the logs and details of the repository we can use the Repository object. Use `repo.log()` to get the logs of the commit messages for the repo. `repo.summary()` returns the most important information regarding the repository.

~~~
Out []:
Summary of Contents Contained in Data Repository 
 
================== 
| Repository Info 
|----------------- 
|  Base Directory: /home/jjmachan/jjmachan/hangar_examples/mnist 
|  Disk Usage: 105.90 MB 
 
=================== 
| Commit Details 
------------------- 
|  Commit: a=41f88d0fe944f431339d3c0b3935e787328d01cd 
|  Created: Mon Apr 13 13:12:04 2020 
|  By: jjmachan 
|  Email: jjmachan@g.com 
|  Message: added all the mnist datasets 
 
================== 
| DataSets 
|----------------- 
|  Number of Named Columns: 6 
|
|  * Column Name: ColumnSchemaKey(column="mnist_test_images", layout="flat") 
|    Num Data Pieces: 10000 
|    Details: 
|    - column_layout: flat 
|    - column_type: ndarray 
|    - schema_hasher_tcode: 1 
|    - data_hasher_tcode: 0 
|    - schema_type: fixed_shape 
|    - shape: (784,) 
|    - dtype: float32 
|    - backend: 00 
|    - backend_options: {'complib': 'blosc:lz4hc', 'complevel': 5, 'shuffle': 'byte'} 
|
|  * Column Name: ColumnSchemaKey(column="mnist_test_labels", layout="flat") 
|    Num Data Pieces: 10000 
|    Details: 
|    - column_layout: flat 
|    - column_type: ndarray 
|    - schema_hasher_tcode: 1 
|    - data_hasher_tcode: 0 
|    - schema_type: fixed_shape 
|    - shape: (1,) 
|    - dtype: int64 
|    - backend: 10 
|    - backend_options: {} 
|
|  * Column Name: ColumnSchemaKey(column="mnist_training_images", layout="flat") 
|    Num Data Pieces: 50000 
|    Details: 
|    - column_layout: flat 
|    - column_type: ndarray 
|    - schema_hasher_tcode: 1 
|    - data_hasher_tcode: 0 
|    - schema_type: fixed_shape 
|    - shape: (784,) 
|    - dtype: float32 
|    - backend: 00 
|    - backend_options: {'complib': 'blosc:lz4hc', 'complevel': 5, 'shuffle': 'byte'} 
|
|  * Column Name: ColumnSchemaKey(column="mnist_training_labels", layout="flat") 
|    Num Data Pieces: 50000 
|    Details: 
|    - column_layout: flat 
|    - column_type: ndarray 
|    - schema_hasher_tcode: 1 
|    - data_hasher_tcode: 0 
|    - schema_type: fixed_shape 
|    - shape: (1,) 
|    - dtype: int64 
|    - backend: 10 
|    - backend_options: {} 
|
|  * Column Name: ColumnSchemaKey(column="mnist_validation_images", layout="flat") 
|    Num Data Pieces: 10000 
|    Details: 
|    - column_layout: flat 
|    - column_type: ndarray 
|    - schema_hasher_tcode: 1 
|    - data_hasher_tcode: 0 
|    - schema_type: fixed_shape 
|    - shape: (784,) 
|    - dtype: float32 
|    - backend: 00 
|    - backend_options: {'complib': 'blosc:lz4hc', 'complevel': 5, 'shuffle': 'byte'} 
|
|  * Column Name: ColumnSchemaKey(column="mnist_validation_labels", layout="flat") 
|    Num Data Pieces: 10000 
|    Details: 
|    - column_layout: flat 
|    - column_type: ndarray 
|    - schema_hasher_tcode: 1 
|    - data_hasher_tcode: 0 
|    - schema_type: fixed_shape 
|    - shape: (1,) 
|    - dtype: int64 
|    - backend: 10 
|    - backend_options: {}
~~~

## Conclusion
Like I mentioned earlier the python APIs are the most used interface to Hangar but Hangar also offers an intuitive CLI also. You can create, delete Columns, manage the various branches and remote data sources, push and fetch data from remote, manage the hangar server etc. For more details check the [official docs](https://hangar-py.readthedocs.io/en/stable/cli.html<Paste>).

Now that we have successfully added the MNIST dataset to Hangar in the next part we will create a dataloader to get the data from Hangar and train a model using it.

Seeya Then!
