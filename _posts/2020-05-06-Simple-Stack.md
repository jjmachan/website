---
title: "PyTorch + BentoML + Heroku: The Simple Stack"
layout: post
data: 2020-05-03 11:14
image: "https://cdn-images-1.medium.com/max/800/1*huEK2vy4PBKWXrgVf5BjxA.png" 
headerImage: true
tag:
 - MLOps
 - bentoml
 - pytorch
 - heroku
category: blog
author: jjmachan
description: Introduction what I call the simple stack with pytorch, bentoml and heroku.
---

# Introduction

> “ It takes a lot of hard work to make something simple”

To be honest I’m a sucker for minimalistic designs. They are functional yet in
their own way beautiful. This is why I love the [UNIX
Philosophy](https://homepage.cs.uri.edu/~thenry/resources/unix_art/ch01s06.html)
so much. It talks about creating beautiful tools that do one thing, does them
well and works seamlessly with other programs. A well-designed software tool
helps you go further when creating your products and brings out the art in
programming and these tools I feel come close to having that clean design that
makes developing using these really fun and fulfilling.

Simple tools are also easy to learn and teach since you get a top-down view of
what is happening. That is why I’m using these tools for showing you how you can
make a deep learning model and put it on the cloud for everyone to use. For this
tutorial, A we are going to apply transfer learning to train a PyTorch model
that can tell if given an Image whether it is an Ant or a Bee. Its a simple
model and is taken from the [official Pytorch
tutorials](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
Then we will use BentoML to take the model we trained, pack it into a container
and create an API endpoint for it. Finally, we will use Heroku which is a cloud
platform for hosting apps to deploy our model so that others can use it.


I won’t be able to go deep into each step but I will make sure you understand
the basics so that if issues arise you will be able to find solutions on your
own.

_**Note:** There is an accompanying [GitHub repo
](https://github.com/jjmachan/resnet-bentoml) which has all the code for this
tutorial do download it and follow along._

With that lets start!

# Training the model in PyTorch.

I assume you are familiar with PyTorch and have trained models using it. If not
could check out these awesome tutorials to get a better idea of whats going on
in the Pytorch end.

1.  [Pytorch 60min
Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)-
perfect for those who know ML and have used another framework before.
1.  [Pytorch Zero to All](https://github.com/hunkim/PyTorchZeroToAll)- A
comprehensive PyTorch tutorial.
1.  [Ritchie's The Incredible
PyTorch](https://www.ritchieng.com/the-incredible-pytorch/)- A list of other
awesome PyTorch resources.

The training is scripted and you can get away even if you don’t code PyTorch but
I highly recommend that you do check out the resources mentioned.

As I mentioned earlier this section is built on top of PyTorch’s official
tutorial on transfer learning. Basically, we take a ResNet model that is
pre-trained on the ImageNet dataset and we modify the last layers so that it can
differentiate between the picture of an ant or bee.

To train a model we need data. The Pytorch devs use this dataset for training.
You can download, extract and rename the dataset to data or you can run the
get_data.sh script to do the same

    $ sh get_data.sh

Now we can train the model. I’ve written a script which trains loads the data,
defines the model and trains it. It is based on the PyTorch tutorial so check
that out for a detailed explanation of the steps involved. Run

    $ python train.py

which will run the script fro 25 epochs on the GPU (or CPU if it is not
available). The script will train the model and save it as a checkpoint in the
saved_models folder. It takes around 10 mins to train on a GPU and 30 mins on
CPU. If you don’t have a GPU reduce the number of epochs.

After training the script will ask if you want to save the model into bento.
Skip this step for now.

# Packing and Serving Using Bento

BentoML is a tool that helps you build production-ready API endpoints for your
ML models. All the ML frameworks are supported like TF, Keras, Pytorch,
Scikit-learn and more. It is still relatively new and under *active development*,
though the current APIs are stable enough for production use. Since its a new
tool I’ll tell you more.

## Core Concepts

So the main idea of BentoML is to help Data Science teams ship prediction
services faster while making it easy to test and deploy models following the
best practices from DevOps. It's almost like a wrapper for your model which
creates an API endpoint and making it easier to deploy. The following are the
important concepts you use to create any prediction service using BentoML.

**Bento Service:** This is the base class for building our prediction service
and all the services we write inherited all its properties from this class. we
define the properties specific to the API like what type of data the endpoint
expects, what the dependencies are, how the model handles the data we get from
the endpoint and more. In short, all the information on how to create the
endpoint and pack the model is inferred from the attributes of this class.

**Model Artifacts:** artifacts are the trained models which are packed using
bento. BentoML supports different ML frameworks and these are to be handled in
their own ways. The model artifact handles serialization and deserialization
automatically according to the artifact chosen corresponding to the ML framework
used. For a complete list of artifacts supported check out the
[doc](https://docs.bentoml.org/en/latest/api/artifacts.html)

**Handlers:** these specify the type of data the model is expecting. I can be
JSON, Pandas dataframe, Tensorflow tensor, images etc. The complete list of
handlers is given in the
[doc](https://docs.bentoml.org/en/latest/api/handlers.html)

<script src="https://gist.github.com/jjmachan/129d3264e2c2a291d5cb7b53a58760e4.js"></script>

**Saving BentoService:** This is where the magic happens. The word bento is
taken from the Japanese and it is a well-packed meal box that has all the
different items like rice, chicken, fish, pickles etc all neatly packed into a
box and this perfectly denotes bento. In fact, that is basically all that bento
does. It takes you models and packs them in a containerized package according to
your BentoService, ready to be served for the world to consume.

<script src="https://gist.github.com/jjmachan/fedc642e56fba139c5c49b267f6536ab.js"></script>
<figcaption class='caption'>Saving the model in Bento</figcaption>

Now the model is containerised and saved into the saved_path. Take a note of the
path since we will be using this for serving the models. Now we can check out
the service we created by serving the model in the development environment bento
provides. Now you can save the model by calling the `saveToBento.py` script.

    $ python saveToBento.py 

This will pack and save the checkpoint created by the train script. The train
script also has a prompt asking you whether you want to save after each
training.

To serve all you have to do is run

    $ bentoml serve IrisClassifier:latest

Or if you know the saved_path

    $ bentoml serve 

# Hosting on Heroku

Heroku is a cloud platform that lets you deliver your apps without worrying too
much about the infrastructure details. It’s a great tool since it enables you to
work on what you love, which is building your app and Heroku takes care of
serving it to millions of users.

BentoML does not have built-in support for Heroku right now (will be out in the
next release!) but with a few tweaks we can pull it off. BentoML packs our
models into docker containers and we can deploy docker container using Heroku’s
Container Registry.

First, make sure you have Heroku-cli and Docker installed and that you have
logged into heroku-cli. Now navigate to the $saved_path where bento packed our
model. Like I said it is already containerised but Heroku requires us to make
some changes to the Dockerfile to work with its Container Registry.

Before that a word about docker. Docker makes it easy to package your app and
all its dependencies so that it is easy to run in different systems without
about setting up and the environment in it. This makes it super easy for
developers since they can be sure that the app they wrote in their dev system
will run in the production systems without additional setup since docker packs
all of these dependencies into it. How this works in basically defining all the
steps you need to set up the environment in a Dockerfile. This file contains all
the instructions for setup up and running your app. Also, it is to be noted that
the app doesn’t exactly run directly on your device. It runs on Linux containers
which sit on top of the Linux kernel of the host system and run the apps in its
own isolated environment. To know more about docker check out this [awesome
blog.](https://djangostars.com/blog/what-is-docker-and-how-to-use-it-with-python/)

Coming to our example, if you check the Dockerfile in $saved_path we can see
that it exposes port 5000. This is so that our server, Gunicorn, listens for
requests coming through this port. Heroku, on the other hand, requires that
containers deployed using its Container Registry to listen to a port set by
Heroku. The port number is passed to it using the environment variable $PORT. So
our script has to run our server on $PORT instead of 5000.

For that we change the last CMD command from

    CMD ["bentoml serve-gunicorn /bento $FLAGS"]

To

    CMD bentoml serve-gunicorn /bento --port $PORT

Also, comment out EXPOSE 5000 and ENTRYPOINT to work on Heroku. The final
Dockerfile will be like this

<script src="https://gist.github.com/jjmachan/6caa2dd0d1bc4ad6e091abd5c4a97782.js"></script>

Now let's run it on our system to check if everything is working. First, build
the docker image by running

    $ docker build . -t resnet_heroku

This will build the image with the tagname resnet_heroku. Now to run the docker
container

    docker run -p 5000:5000 -e PORT=5000 resnet_heroku

* -p binds our 5000 port to that of the container
* -e passes the environment variable $PORT to docker

Go to localhost:5000 and check if everything is working. If it is great, let
push it to Heroku!

First login to Heroku’s Container Registry using

    $ heroku container:login

Heroku stores everything as apps, to create the Heroku app run

    heroku create -a

Now you can build and push the BentoService to your Heroku app.

    $ heroku container:push web --app 

And release the app

    $ heroku container:release web --app 

Congrats you just deployed your first ML API to the world! Let's try it out.
Download any picture from the internet showing ants and bees and pass it to the
API using curl.

    $ curl POST https://ant-bee-classifier.herokuapp.com/predict -H "Content-Type: image/*" --data-binary @image.jpg -v

This should output the results form the API. If it doesn’t and something is
wrong, try checking the logs by running

    $ heroku logs --tail -a 

# Conclusion

With that, you have now learned to train and push a model into production using
the “Simple Stack”. This is perfect for simple setup so that you can get your ML
model our into the world with minimum work.

