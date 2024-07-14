# Netflix Recommender Systems

Netflix, an American subscription streaming service and production company, had 222 million subscribers worldwide by September 2022, offering thousands of movies. Since users won't watch every movie, recommending appropriate content based on user data is crucial. The challenge lies in selecting movies and series that align with user preferences.

In this project, we employ three methods—Naive Approach, UV Matrix Decomposition, and Matrix Factorization—to recommend content. For instance, if users *A* and *B* share similar preferences, recommending a movie seen by *B* but not *A* can boost *A*’s click-through rate, thereby increasing revenue for the company.

### Data

We use the MovieLens 1M Dataset from [Grouplens](https://grouplens.org/datasets/movielens/), which includes ratings, users, and movies data files. The users and movies files provide basic information about the respective entities. The ratings file, formatted as (user :: movie :: rating :: timestamp), contains 1,000,209 ratings from 6,040 users for 3,952 movies, resulting in nearly 4% saturation. Each user has made at least 20 ratings.

### Methods

Naive Approaches

UV Matrix Decomposition

Matrix Factorization

*Please see the report for detailed information.*


### Enviroment Requirement

You can install the necessary pacakages by using command:

```{bash}
pip install requirement.txt
```

The path we used for the dataset is:

```{bash}
/ml-1m
```

which means you need to create a folder (named `ml-1m`) under the path of the code.



### How to reproduce the result of Naive Approches

Run command:

```{bash}
python naive.py
```

### How to reproduce the results of UV Matrix Decomposition

Run command:

```{bash}
nohup ./runUV.sh >> resultUV.log 2>&1 &
```

And the results of UV matrix decomposition will be written to the `resultUV.log` file.

And it will produce two figures which named as `UV_A.png` and `UV_B.png`.

### How to reproduce the results of Matrix Factorization

Run command:

```{bash}
nohup ./runMF.sh >> resultMF.log 2>&1 &
```

And the results of Matrix Factorization will be written to the `resultMF.log` file.

### How to generate the plots of Matrix Factorization

Run command:

```{bash}
nohup ./runMFplot.sh >> resultMFplot.log 2>&1 &
```

And it will produce eights figures which named as `baseline.png` and from`configA.png` to `configH.png`.

### How can we adjust the parameters of the algorithms?

You can tune the number of factor, number of iterations and any other parameters of UV matrix decomposition and matrix factorization. You can use the help command to see the meaning of each parameters like:

```{bash}
python UV.py -h
python MF.py -h
```

