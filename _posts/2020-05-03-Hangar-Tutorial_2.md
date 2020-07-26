---
title: "Hangar Tutorial (2/2): Training a Model with Versioned Data"
layout: post
data: 2020-05-03 11:00
image:
headerImage: false
tag:
 - MLOps
 - Hangar
 - Version Control
star: false
category: blog
author: jjmachan
description: How to get the data from Hangar and train a PyTorch model.
---

In the last tutorial, we introduced Hangar, a python library that helps you
version control data. Well, data is the new oil and properly managing and
streaming your dataflow is vital for your company. Hangar is a nice tool to have
in your toolbox which brings in best practices like version control from
software development to ML engineering which in turn helps you to serve robust
ML services to your customer. If you haven’t checked out the [previous
blog](https://towardsdatascience.com/hangar-tutorial-1-2-adding-your-data-to-hangar-d7f039265455)
do check it out. I discuss the core concepts of Hangar and how to create your
own Hangar repo there.

<div style="width:100%;height:0;padding-bottom:140%;position:relative;">
  <iframe src="https://giphy.com/embed/HUplkVCPY7jTW" width="100%" height="100%" style="position:absolute" frameBorder="0" class="giphy-embed" allowFullScreen>
  </iframe>
</div>

<figcaption class='caption'>
ML workflow is getting increasingly complex and I think Hangar is a really
important tool to have in your toolbox. 
<a href="https://giphy.com/gifs/watson-geekout-HUplkVCPY7jTW">(via GIPHY)</a>
</figcaption>

_**Note:** There is an accompanying GitHub repo with some detailed notebooks
explaining all of the concepts discussed. You can try that out along with this
tutorial. [[link](https://github.com/jjmachan/hangar_tutorials)]_

### Where were we..?

So we have learned the core concepts of Hangar and we added our toy dataset
(MNIST) into Hangar. Today we will continue from where we left off and train a
model from the data we have added to Hangar. First of all, let’s connect to the
Hangar repo.

<script src="https://gist.github.com/jjmachan/c4a770a059ee33ef895202425b8ee761.js"></script>

Let's take a quick look at the summary to get an idea of the data in our repo.

<script src="https://gist.github.com/jjmachan/c41c91d6ea95d3f46aae8fd93960735b.js"></script>

    Summary of Contents Contained in Data Repository 
     
    ================== 
    | Repository Info 
    |----------------- 
    |  Base Directory: /home/jjmachan/jjmachan/hangar_tutorial 
    |  Disk Usage: 105.88 MB 
     
    =================== 
    | Commit Details 
    ------------------- 
    |  Commit: a=39a36c4fa931e82172f03edd8ccae56bf086129b 
    |  Created: Fri May  1 18:23:19 2020 
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

As you can see the data is stored in 6 ndarray columns. 2 columns each store the
image and target pair for the 3 data splits, train, test and validation. Now
let’s create a read-only checkout from the master branch to access the columns.

<script src="https://gist.github.com/jjmachan/8e5568fd7bcd0f2eb2a0862b4a940c8d.js"></script>

    * Checking out BRANCH: master with current HEAD: a=39a36c4fa931e82172f03edd8ccae56bf086129b

### Dataloaders

Now, let’s load the data from Hangar to feed it into our neural network for
training. Even though you can directly access the data using the checkout, the
recommended way is to use the `make_torch_dataset`. It offers more configurable
options and makes it easier to load into PyTorch.

**make_torch_dataset(columns**, *keys: Sequence[str] = None*, *index_range:
slice = None*, *field_names: Sequence[str] = None***)**

The `make_torch_dataset` creates a PyTorch Dataset with the columns passed to it.
In our example, if we would pass the ‘mnist_training_images’ column and
‘mnist_training_labels’ column. Now `make_torch_dataset` will return, for each
index, a tuple with the image and label corresponding with the index.

Since it returns a PyTorch Dataset we can pass it into a PyTorch DataLoader to
perform shuffling, batching etc. This means we get all the goodness from the
Pytorch DataLoader

<script src="https://gist.github.com/jjmachan/a07711ea98d42c1358e6444e63844c37.js"></script>

Now, iterating through the trainDataloader will give us the data in batches and
as tensors ready to be used for training.

### Training

We will not go into the details of how models are defined and trained. If you
find the code below hard to catch up I highly recommend that you check out the
[Pytorch
tutorials](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

We will define a simple 3 layer, fully connected neural network for MNIST.

<script src="https://gist.github.com/jjmachan/1b8e3c853c8817a22cff2038e8c972b7.js"></script>

Now using the dataloader we iterate through batches, do the forward propagation,
calculate the loss, find the gradients via backpropagation and adjust the model
parameters, just the standard the neural network training you are familiar with.

<script src="https://gist.github.com/jjmachan/05eac2a88e0dc819e75e5426041aa313.js"></script>


    [EPOCH 0/10] Train Loss: 1.2083537247954312
    Test Loss: 0.47797256113050846 Accuracy: 0.865814696485623
    [EPOCH 1/10] Train Loss: 0.3944594695549208
    Test Loss: 0.3467818654228609 Accuracy: 0.897064696485623
    [EPOCH 2/10] Train Loss: 0.31767420198765994
    Test Loss: 0.2960374093682954 Accuracy: 0.9107428115015974
    [EPOCH 3/10] Train Loss: 0.27709063706813486
    Test Loss: 0.2613061714274719 Accuracy: 0.9223242811501597
    [EPOCH 4/10] Train Loss: 0.24662601495887404
    Test Loss: 0.234231408689909 Accuracy: 0.9306110223642172
    [EPOCH 5/10] Train Loss: 0.22161395768786918
    Test Loss: 0.21181030162928488 Accuracy: 0.9365015974440895
    [EPOCH 6/10] Train Loss: 0.20021527176466666
    Test Loss: 0.19286749035137862 Accuracy: 0.9421924920127795
    [EPOCH 7/10] Train Loss: 0.18172580767267993
    Test Loss: 0.1767021114335428 Accuracy: 0.946685303514377
    [EPOCH 8/10] Train Loss: 0.16573666792806246
    Test Loss: 0.16299303889887545 Accuracy: 0.9507787539936102
    [EPOCH 9/10] Train Loss: 0.1518441192202165
    Test Loss: 0.15127997121999795 Accuracy: 0.954073482428115

and voilà!

We have successfully trained our model using the data from Hangar.

### Conclusion

By now you should have learned the basics of Hangar, how to define columns
according to your dataset, add data to these columns and load the data from it
to train models in PyTorch. Hangar also has a bunch of other cool features like
[Remote
Repositories](https://hangar-py.readthedocs.io/en/stable/Tutorial-003.html),
[Partial
Downloads](https://hangar-py.readthedocs.io/en/stable/Tutorial-003.html#Partial-Fetching-and-Clones)
of data from these remote repos, [Hangar
External](https://hangar-py.readthedocs.io/en/stable/externals.html) which is a
high-level interface to Hangar which you can use when you are integrating Hangar
with your existing dataflow etc. I’ve linked the relevant docs and I recommend
that you check them out to unlock all the features Hangar has to offer.

As more and more companies are figuring out new ways to get the power of machine
learning into their products it is important that we incorporate some best
practices into it in order to avoid building up technical debt. This is still a
growing field and everyone is still figuring out what these “best practices” are
and which workflows might work best for them. Versioning your data is one such
practice that can be added to ensure reproducibility, collaboration and better
management. Tools like Hangar and DVC are the best options when it comes to this
even though there is still a lot of work to be done. Hangar is super easy to add
to your existing ML workflow and hopefully, this tutorial has given you a better
understanding it.

Cheers :heart:

### *Auxilary*

This is an auxiliary section which compares two aspects hangar

1.  The size of the dataset before and after storing in Hangar
1.  speed of the dataloaders using Hangar and a simple dataloader that takes a
mnist_dataset that we have defined. 

Now, these are my personal experiments and benchmarks so try them out for
yourself and check if the results I discuss below are the same. If now I would
love it if you could share it for everyone! The code is available on the [GitHub
repo.](https://github.com/jjmachan/hangar_tutorials)

First, let's talk about dataset sizes. Now internal Hangar does perform a really
good job to crunch the data when saving it into disk. In our example the
original MNIST file is 210Mb but in Hangar it only takes up 105Mb. But I’m
talking about the uncompressed version of MNIST. The compressed version is just
17Mb which is a lot. This is one of the biggest problems I have seen with
Hangar. It does a great job of compressing numerical data but most probably
there are other formats for your data that will likely offer significant size
reduction. I have first experience when I tried to save jpeg files in Hangar and
they took up a significant amount of space.

When I discussed this with the awesome people at TensorWerk (the company
supporting hangar), they said that this configuration is to ensure best
read speed/size on disk trade off. You can go into hangar and configure it to
optimize in which ever way we seem fit but naturally given the fact the compute
is a lot more expensive than storage, it's wise to optimize for read speeds
instead. 

So I would advise you to store the data after all the preprocessing is done and
in the form that is ready to be fed into the models. The original files can be
compressed and stored in your data lakes.

Second is loading from disk for training. My experiments show hangar turns out
to be a slower that native implementation or when using the PyTorch
implementations. 

This is the output from my system.

![](https://cdn-images-1.medium.com/max/800/1*IXl0EFD-XGI1IrTDl1T4gQ.png)

In this case the major source of performance dip is because hangar is reading
directly from file, where as in the other case I have loaded it into memory. But
I think that there is still a lot of work needed in that direction to improve
the read and write speeds. Now hangar is still a very small project and still
under active development. If the project interests you I would recommend you
checkout the [repo](https://github.com/tensorwerk/hangar-py).

