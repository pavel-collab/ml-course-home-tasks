import numpy as np

import random

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            # Your Code Here
            # набор всевозможных индексов объектов
            possible_indexes = list(range(data_length))
            # генерируем наборы случайных индексов
            self.indices_list.append([random.choice(possible_indexes) for  _ in range(data_length)])
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            # Для каждой модели формируем выборки объектов
            # выбираем объекты с индексами в соответствующем indices_list
            data_bag, target_bag = [data[idx] for idx in self.indices_list[bag]], [target[idx] for idx in self.indices_list[bag]] # Your Code Here
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        # Your code here
        prediction = np.zeros(len(self.data))
        for idx, obj in enumerate(self.data):
            # Пробегаем по всем объектам исходной выборки. 
            # Затем пробегаемся по всем моделям и смотрим, какую метку для данного объекта (obj) предсказывает каждая модель.
            object_prediction_list = [model.predict([obj]) for model in self.models_list]
            # Сохраняем для данного объекта предсказания всех моделей. Итоговое предсказание для данного объекта -- среднее от предсказаний всех моделей.
            prediction[idx] = np.mean(object_prediction_list)
        return prediction
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        # Your Code Here
        for i in range(len(self.data)):
            # Пробегаем по всем объектам исходной выборки. Для выбранного объекта пробегаем по всем бэгам.
            for idx, bag in enumerate(self.indices_list):
                # если в текущем бэге данного объекта нет, значит соответствующая можель на нем не обучалась.
                # Делаем предсказание для этого объекта на модели соответсвуюзего бэга.
                if i not in bag:
                    list_of_predictions_lists[i].append(self.models_list[idx].predict( [self.data[i]] ).item())
        # Таким образом, для каждого объекта мы получаем множество предсказаний тех моделей, которые его не видели.
        
        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        # Таким образом, для каждого объекта мы получаем множество предсказаний тех моделей, которые его не видели.
        # Теперь для каждого объекта усредняем данные предсказания, получая, таким образом, среднее предсказание.
        #! В идеальном мире, для каждого объекта найдется такая модель, которая на нем не обучалась.
        #! Но, на практике, у существуют объекты, которые попали в тренировочный сет КАЖОЙ модели.
        #! Для таких объектов список предсказаний на этом моменте программы будет пуст. Усреднение от пустого списка дает NaN.
        self.oob_predictions = [np.mean(np.array(pred)) for pred in self.list_of_predictions_lists] # Your Code Here
        
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        # Для каждого объекта мы получили среднее предсказание от всех моделей. 
        # Теперь, поскольку мы знаем реальные метки этих объектов, просто посчитаем ошибку.
        #! Поскольку при усреднении предсказаний для каждого объекта были объекты, которые попали в обучающие выборки всех моделей
        #! в self.oob_predictions могут быть элементы NaN. Эти элементы не учитываются при усреднении, поэтому в генератор закладываем if-фильтр.
        return np.mean([item for item in (self.target - self.oob_predictions)**2 if not np.isnan(item)]) # Your Code Here