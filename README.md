# k-means-app
Streamlit application to explore k-means clustering

# Deployed with streamlit:
https://alicjakarczewska-k-means-app-kmeans-iris-qpq98y.streamlitapp.com/

# The aim and use of application
This application shows the method k-means clustering on Iris dataset.
It include 4 metrics (Euclidean, L1, Chebyshev and Mahalanobise distances) and various techniques for measuring similarity (accuracy score, jaccard score, silhouette sc, cosine similarity).

Used python libraries:
* streamlit,
* pandas,
* seaborn,
* sklearn,
* numpy,
* matplotlib...

# Running aplication locally

It is recommended to firstly manage an environment, for example conda:

`conda create --name <env_name> --file requirements.txt`

And then activate it:

`conda activate <env_name>`

To run application use:

`streamlit run kmeans-iris.py`
