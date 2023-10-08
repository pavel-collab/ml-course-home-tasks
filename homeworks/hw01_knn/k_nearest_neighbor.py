import numpy as np
"""
Credits: the original code belongs to Stanford CS231n course assignment1. Source link: http://cs231n.github.io/assignments2019/assignment1/
"""

class KNearestNeighbor:
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                
                # посчитаем вектор который является разницой 
                # векторов тестовой и тренировочной точек
                d_vec = X[i] - self.X_train[j]
                # используем скалярное произведение вектора самого на себя, 
                # чтобы вычислить его длину
                dists[i][j] = np.sqrt(np.dot(d_vec, d_vec))

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # построим матрицу; каждая столбец матрицы -- 
            # разница между тренировочной точкой и iй тестовой точкой 
            d_vecs = self.X_train - X[i]
            # возведем каждую компоненту матрицы в квадрат, таким образом,
            # каждый компонент столбца есть (xi - yi)^2
            d_vecs2 = d_vecs**2
            # суммируя компоненты каждого столбца и вычисляя кв. корень 
            # получаем строку с расстояниями от iй тестовой точки до каждой тренировочной
            dists[i] = np.sqrt(sum(d_vecs2.T))

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # чтобы получить матрицу расстояний между тестовыми и тренировочными точками
        # используем теорему косинусов
        # x^2 = a^2 + b^2 - 2 (a, b)
        # где 
        # a^2 -- квадрат модуля вектора тренировочной точки
        # b^2 -- квадрат модуля вектора тестовой точки
        # (a, b) -- скалярное произведение векторов

        #! NOTE sum складывает элементы по столбцам, а не по строкам
        '''
        NOTE: по какой-то дурацкой причине numpy не умеет складывать столбцы,
        только строки. Если хочешь сложить столбец с матрицей -- будь добр, транспонируй столбец 
        в строку. А поскольку ихменился порядок размерности, матрицу тоже надо транспонировать.
        А так как ответ долже быть определенной размерности, то в конце полученный ответ надо
        транспонировать еще раз.
        '''
        a2 = sum((self.X_train**2).T)
        b2 = sum((X**2).T)    

        ab = np.dot(self.X_train, X.T)
        dists = np.sqrt((a2 - (2*ab).T).T + b2.T)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists.T

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            ind = np.argsort(dists)
            closest_y = np.zeros((num_test, k))

            for i in range(num_test):
                idxes = ind[i][:k]
                closest_y[i] = np.array([self.y_train[j] for j in idxes])


            closest_y = closest_y.astype('int32', copy=False)

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            for i in range(num_test):
                closest_y_class = np.argmax(np.bincount(closest_y[i]))
                closest_y_class_idx = list(closest_y[i]).index(closest_y_class)

                y_pred[i] = closest_y[i][closest_y_class_idx]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
