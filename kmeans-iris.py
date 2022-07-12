# Base libraries
import numpy as np
import scipy as sp

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import streamlit as st

# Algorithm testing libraries
from sklearn.cluster import KMeans

from sklearn.metrics.pairwise import manhattan_distances

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import jaccard_score
from scipy.spatial import distance
# distance.chebyshev([1, 0, 0], [0, 1, 0])
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score

st.set_page_config(
     page_title="K-means App",
    #  layout="wide",
    #  initial_sidebar_state="collapsed",
    )

url = 'IRISDAT.TXT'
df = pd.read_csv(url, sep=',', comment='#') 


with st.sidebar:    
    # st.subheader('Info')
    exp_1 = st.expander("Info", expanded=False)
    with exp_1:
        repo = '[Github](http://github.com)'
        ak = '[Alicja Karczewska](http://github.com)'
        mc = '[Marek Czarkowski](http://github.com)'
        st.subheader('Repozytorium kodu')
        st.markdown(repo, unsafe_allow_html=True)
        st.subheader('Autorzy')
        st.markdown(ak, unsafe_allow_html=True)
        st.markdown(mc, unsafe_allow_html=True)
    exp_2 = st.expander("Grupowanie", expanded=False)
    with exp_2:
        st.markdown('W celu rozwiązania problemu grupowania (analizy skupień) zastosowano metodę k-średnich. \
            Jako dane wejściowe pobiera ona zbiór danych z obiektami bez określonych klas decyzyjnych oraz \
            liczbę naturalną k określającą liczbę klas, na które wejściowy zbiór danych ma zostać podzielony. \
            W wyniku dla każdego z obiektów powinna zostać przypisana \
            klasa decyzyjna (liczba z zakresu 1 ... k)\n \
            Metoda wyboru początkowego zestawu średnich może być dowolna. \n\n Zostały użyte \
            następujące metryki oceny odległości obiektów: odległość euklidesowa, L1, \
            L-nieskończoność, Mahalanobisa.')
    exp_3 = st.expander("Metoda k-średnich", expanded=False)
    with exp_3:
        st.markdown('Metoda k-średnich jest metodą należącą do grupy algorytmów analizy skupień tj. analizy \
            polegającej na szukaniu i wyodrębnianiu grup obiektów podobnych (skupień) . Reprezentuje \
            ona grupę algorytmów niehierarchicznych. Główną różnicą pomiędzy niehierarchicznymi \
            i hierarchicznymi algorytmami jest konieczność wcześniejszego podania ilości skupień. \
            Przy pomocy metody k-średnich zostanie utworzonych k różnych możliwie odmiennych \
            skupień. Algorytm ten polega na przenoszeniu obiektów ze skupienia do skupienia tak długo \
            aż zostaną zoptymalizowane zmienności wewnątrz skupień oraz pomiędzy skupieniami. \
            Oczywistym jest, iż podobieństwo w skupieniu powinno być jak największe, zaś osobne \
            skupienia powinny się maksymalnie od siebie różnić.')
    exp_4 = st.expander("Opis algorytmu", expanded=False)
    with exp_4:
        st.write('Zasada działania zaimplementowanego algorytmu jest następująca: \
            \n\n1. Ustalamy liczbę skupień. \
            \n2. Ustalamy wstępne środki skupień. \
            Środki skupień tak zwane centroidy możemy dobrać na kilka sposobów: losowy \
            wybór k obserwacji, wybór k pierwszych obserwacji, dobór w taki sposób, aby \
            zmaksymalizować odległości skupień. W przypadku implementacji naszego projektu \
            zdecydowaliśmy się na wybór k pierwszych obserwacji. \
            \n3. Obliczamy odległości obiektów od środków skupień. \
            Wybór metryki jest bardzo istotnym etapem w algorytmie. Wpływa ona na to, które z \
            obserwacji będą uważane za podobne, a które za zbyt różniące się od siebie. \
            Najczęściej stosowaną odległością jest odległość euklidesowa. Nasza aplikacja \
            umożliwia wybór następujących metryk: euklidesowej, L1, L nieskończoność, \
            Mahalanobisa.\
            \n4. Przypisujemy obiekty do skupień\n\
            Dla danej obserwacji porównujemy odległości od wszystkich skupień i przypisujemy \
            ją do skupienia, do którego środka ma najbliżej.\
            \n5. Ustalamy nowe środki skupień\n\
            Nowym środkiem skupienia jest punkt, którego współrzędne są średnią arytmetyczną \
            współrzędnych punktów należących do danego skupienia.\
            \n6. Wykonujemy kroki 3,4,5 do czasu, aż warunek zatrzymania zostanie spełniony.\
            \n7. Najczęściej stosowanym warunkiem stopu jest ilość iteracji zadana na początku lub \
            brak przesunięć obiektów pomiędzy skupieniami.\
            \n\nFunkcja odpowiadająca za przeprowadzenie algorytmu jako argumenty pobiera badany\
            zbiór danych, liczbę klastrów, metrykę, liczbę iteracji. Pozwala na ukazanie porównania\
            liczebności klas względem liczebności utworzonych klastrów, utworzenie macierzy pomyłek\
            (przypisanie odpowiednich nazw klas do poszczególnych klastrów <klastry są oznaczone\
            kolejnymi liczbami naturalnymi> wymaga samodzielnego dopasowania przez użytkownika) i\
            dokładności (accuracy), a także wyliczenie miar podobieństwa (jaccard score, silhouette\
            score, cosine similarity) między znalezionym zbiorem klastrów i zbiorem podzielonym\
            względem klas z danych.\
            \n\nAplikacja pozwala także na wyświetlenie wykresu łokciowego, który pomaga w wyborze\
            optymalnej liczby klastrów dla badanego zbioru.')
    exp_5 = st.expander("Miary podobieństwa", expanded=False)
    with exp_5:
        st.write('Indeks Jaccarda, współczynnik podobieństwa Jaccarda – statystyka używana do \
            porównywania zbiorów. Współczynnik Jaccarda mierzy podobieństwo między dwoma \
            zbiorami i jest zdefiniowany jako iloraz mocy części wspólnej zbiorów i mocy sumy tych \
            zbiorów. Wartości przyjmowane przez współczynnik Jaccarda zawierają się w podzbiorze \
            zbioru liczb rzeczywistych <0,1>. Jeśli współczynnik Jaccarda przyjmuje wartości bliskie \
            zeru, zbiory są od siebie różne, natomiast gdy jest bliski 1, zbiory są do siebie podobne. \
            \n\nSilhouette Coefficient (silhouette score) jest miarą używaną do obliczenia jakości techniki \
            grupowania. Jego wartość waha się od -1 do 1: \n * 1: Zgrupowania średnich są dobrze od siebie oddzielone i wyraźnie rozróżnione. \
            \n* 0: Odległość między skupieniami nie jest znacząca. \
            \n* -1: Oznacza, że klastry są przypisane w niewłaściwy sposób \
            \n\n\nPodobieństwo cosinusowe to miara, która określa ilościowo podobieństwo między dwoma \
            lub więcej wektorami. To cosinus kąta między wektorami. Wektory są zwykle niezerowe i \
            znajdują się w wewnętrznej przestrzeni iloczynu. \
            Oblicza podobieństwo cosinusów między próbkami w zbiorach X i Y, jako znormalizowany \
            iloczyn skalarny X i Y: K(X, Y) = <X, Y> / (||X||*||Y||) ')
    exp_6 = st.expander("Zbiór danych", expanded=False)
    with exp_6:
        st.write('Badania przeprowadzono na zbiorze Iris')
        df_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label='📥 Pobierz zbiór danych',
                                        data=df_csv ,
                                        file_name= 'IRISDAT.csv')

all_res = []

column_list = df.columns

cols = ['LISDLG', 'LISSZE', 'PLADLG', 'PLASZE']
crim_lstat_array = np.array(df[cols])

def recalculate_clusters(X, centroids, k, dist):
    """ Recalculates the clusters """
    # Initiate empty clusters
    clusters = {}
    # Set the range for value of k (number of centroids)
    for i in range(k):
        clusters[i] = []
    # Setting the plot points using dataframe (X) and the vector norm (magnitude/length)
    for data in X:
        if(dist == "euc"):
            # Set up list of euclidian distance and iterate through
            euc_dist = []
            for j in range(k):
                euc_dist.append(np.linalg.norm(data - centroids[j]))
            # Append the cluster of data to the dictionary
            clusters[euc_dist.index(min(euc_dist))].append(data)
        if(dist == "l1"):
            # Set up list of l1 distance and iterate through
            l1_dist = []
            for j in range(k):
                l1_dist.append(manhattan_distances([data], [centroids[j]]))
            # Append the cluster of data to the dictionary
            clusters[l1_dist.index(min(l1_dist))].append(data)
        if(dist == "chebyshev"):
            # Set up list of chebyshev distance and iterate through
            chebyshev_dist = []
            for j in range(k):
                # chebyshev_dist.append(distance.chebyshev([data], [centroids[j]]))
                chebyshev_dist.append(np.max(np.abs(data - centroids[j])))
            # Append the cluster of data to the dictionary
            clusters[chebyshev_dist.index(min(chebyshev_dist))].append(data)
        if(dist == "mahal"):
            # Set up list of mahal distance and iterate through
            mahal_dist = []
            # iv = np.cov(X)
            iv = covMatrix
            for j in range(k):
                mahal_dist.append(distance.mahalanobis([data], [centroids[j]], iv))
            # Append the cluster of data to the dictionary
            clusters[mahal_dist.index(min(mahal_dist))].append(data)
    return clusters    
 
def recalculate_centroids(centroids, clusters, k):
    """ Recalculates the centroid position based on the plot """ 
    for i in range(k):
        # Finds the average of the cluster at given index
        centroids[i] = np.average(clusters[i], axis=0)
    return centroids

def plot_clusters(centroids, clusters, k):
    """ Plots the clusters with centroid and specified graph attributes """ 
    colors = ['red', 'blue' , 'green', 'orange', 'blue', 'gray', 'yellow', 'purple']
    fig = plt.figure(figsize = (6, 4))  
    area = (20) ** 2
    for i in range(k):
        st.write(f'Klaster nr {i % k} jest oznaczony kolorem: {colors[i % k]}')
        for cluster in clusters[i]:
            plt.scatter(cluster[0], cluster[1], c=colors[i % k], alpha=0.6)          
        plt.scatter(centroids[i][0], centroids[i][1], c='black', s=200)
    st.write(fig.figure)

def k_means_clustering(X, centroids={}, k=3, repeats=10, dist="euc"):
    """ Calculates full k_means_clustering algorithm """
    for i in range(k):
        # Sets up the centroids based on the data
        centroids[i] = X[i]
        # st.write(f"centroids: {centroids[i]}")

    # Outputs the recalculated clusters and centroids 
    st.subheader('Wykres danych przy pierwszej i ostatniej iteracji')
    for i in range(repeats):        
        clusters = recalculate_clusters(X, centroids, k, dist)  
        centroids = recalculate_centroids(centroids, clusters, k)

        # Plot the first and last iteration of k_means given the repeats specified
        # Default is 10, so this would output the 1st iteration and the 10th        
        if i == range(repeats)[-1] or i == range(repeats)[0]:
            plot_clusters(centroids, clusters, k)

    df1 = pd.DataFrame([ [key, len(value)] for key, value in clusters.items()], columns = ['Cluster', 'Number'])
    st.subheader(f"Liczność klastrów")
    st.write(df1)
    st.subheader("Liczność rzeczywistych klas danych")
    st.write(df['ODMIRYS'].value_counts()) 

    new_df = pd.DataFrame()

    for cluster in clusters:
        for index, value in enumerate(clusters[cluster]):
            df_with_cluster = pd.DataFrame.from_dict(clusters[cluster])
        df_with_cluster['cluster'] = cluster
        
        new_df = pd.concat([new_df, df_with_cluster], axis=0)
        new_df.reset_index(drop=True)
    
    df_to_join_1 = df.sort_values(by=cols).reset_index(drop=True)
    df_to_join_2 = new_df.sort_values(by=[0, 1, 2, 3]).drop_duplicates().reset_index(drop=True)
    df_info = pd.merge(df_to_join_1, df_to_join_2,  how='left', left_on=cols, right_on = [0, 1, 2, 3])
    
    df_info["cluster2"] = df_info["cluster"].map({0: 'SETOSA', 1:'VIRGINIC', 2:'VERSICOL'}) 
    df_info["ODMIRYS_to_num"] = df_info["ODMIRYS"].map({'SETOSA': 0, 'VIRGINIC': 1, 'VERSICOL':2}) 

    df_confusion = pd.crosstab(df_info["cluster2"], df_info["ODMIRYS"], rownames=['Actual'], colnames=['Predicted'], margins=True)

    st.subheader("Macierz pomyłek - confusion matrix")
    st.dataframe(df_confusion)

    st.subheader("Miary podobieństwa")
    accuracy_sc = accuracy_score(df_info["cluster2"], df_info["ODMIRYS"])
    st.write(f'accuracy_score: {accuracy_sc}')

    jaccard_sc = jaccard_score(df_info["cluster2"], df_info["ODMIRYS"], average=None)
    st.write(f'jaccard_score: {jaccard_sc}')

    silhouette_sc = silhouette_score(df_info[['LISDLG', 'LISSZE', 'PLADLG', 'PLASZE',"ODMIRYS_to_num"]], df_info["cluster"])
    st.write(f'silhouette_sc: {silhouette_sc}')
    
    cosine_sc = cosine_similarity(np.asmatrix(df_info["cluster"]), np.asmatrix(df_info["ODMIRYS_to_num"]))[0][0]
    st.write(f'cosine_similarity: {cosine_sc}')

    
    res = pd.DataFrame([{'metrics':dist, 'accuracy_score':accuracy_sc, 'jaccard_score':jaccard_sc, 'silhouette_score':silhouette_sc, 'cosine_similarity':cosine_sc}])
    all_res.append(res)

st.title("GRUPOWANIE (ANALIZA SKUPIEŃ)")
st.header("1. Metryka euklidesowa")
k_means_clustering(crim_lstat_array, k=3, repeats=20, dist="euc")
plt.clf()

st.header("2. Metryka Manhattan - L1 - taksówkowa")
k_means_clustering(crim_lstat_array, k=3, repeats=20, dist="l1")
plt.clf()

st.header("3. Metryka Chebysheva - L∞ - maksimowa")
k_means_clustering(crim_lstat_array, k=3, repeats=20, dist="chebyshev")
plt.clf()

st.header("4. Metryka Mahalanobisa")
data = np.array([df['LISDLG'],df['LISSZE'],df['PLADLG'],df['PLASZE']])
covMatrix = np.cov(data,bias=True)
st.write(covMatrix)
k_means_clustering(crim_lstat_array, k=3, repeats=10, dist="mahal")
plt.clf()


def sklearn_k_means(X, k=3, iterations=10):
    """ KMeans from the sklearn algorithm for comparison to algo from scratch """ 
    km = KMeans(
        n_clusters=k, init='random',
        n_init=iterations, max_iter=300, 
        random_state=0
    )

    y_km = km.fit_predict(X)

    plt.clf()

    # plot the 3 clusters
    fig1 = plt.scatter(
        X[y_km == 0, 0], X[y_km == 0, 1],
        s=50, 
        c='blue',
        label='cluster 1'
    )

    plt.scatter(
        X[y_km == 1, 0], X[y_km == 1, 1],
        s=50, 
        c='red',
        label='cluster 2'
    )

    plt.scatter(
        X[y_km == 2, 0], X[y_km == 2, 1],
        s=50, 
        c='green',
        label='cluster 3'
    )

    # plot the centroids
    for i in range(3):
        plt.scatter(
            km.cluster_centers_[-i, 0], km.cluster_centers_[-i, 1],
            s=100,
            c='black',
            label='centroids'
        )
    plt.legend(scatterpoints=1)
    plt.title('Clustered Data')
    plt.grid()
    plt.show()
    st.write("fig1")
    st.write(fig1.figure)

    plt.clf()

    # "Elbow Plot" to demonstrate what is essentially the usefulness of number for k
    # Calculate distortion (noise) for a range of number of cluster
    distortions = []
    for i in range(1, 11):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(X)
        distortions.append(km.inertia_)

    # Plot k vs distortion
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    fig2 = plt.ylabel('Distortion')
    plt.show()
    st.subheader("Wykres łokciowy - Elbow plot")
    st.write(fig2.figure)

def elbow_plot(X, k=3, iterations=10):
    plt.clf()

    # "Elbow Plot" to demonstrate what is essentially the usefulness of number for k
    # Calculate distortion (noise) for a range of number of cluster
    distortions = []
    for i in range(1, 11):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(X)
        distortions.append(km.inertia_)

    # Plot k vs distortion
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    fig2 = plt.ylabel('Distortion')
    plt.show()
    st.subheader("Wykres łokciowy - Elbow plot")
    st.write(fig2.figure)


# sklearn_k_means(crim_lstat_array)
elbow_plot(crim_lstat_array)


df_res = pd.concat(all_res).reset_index(drop=True)
st.title("Podsumowanie")
st.header("Zestawienie wyników miar podobieństwa dla poszczególnych metryk")
st.write(df_res)
