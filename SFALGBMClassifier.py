import numpy as np
import lightgbm as lgb
import warnings
from SFAClassifier import SFAClassifier
from tqdm import tqdm
from sklearn.metrics import roc_auc_score as auc


class SFALGBMClassifier(SFAClassifier):

    def __init__(self, ds_name, seed):
        super().__init__(ds_name, seed)
        self.model_name = 'lgbm'
        self.device = 'cpu'  # 強制使用 CPU
        self.n_trials = 1   # 只進行 1 次試驗
        self.sample_ratio = 0.5  # 使用 50% 的數據進行訓練

    def get_default_params(self):
        params = {
            'objective': self.get_task(),
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.1, # 增大學習率以加快收斂
            'num_leaves': 31, # 減少葉子節點數
            'max_depth': 6, # 減少樹的深度
            'min_data_in_leaf': 50, # 增加葉子節點最小樣本數
            'feature_fraction': 0.8, # 特徵抽樣比例
            'bagging_fraction': 0.8, # 數據抽樣比例
            'bagging_freq': 5,     # 每5次迭代進行一次bagging
            'verbosity': -1,
            'num_classes': self.get_num_classes(),
            'seed': self.seed,
            'device': 'cpu',
            'force_row_wise': True,
            'max_bin': 255,       # 減少特徵分桶數
            'subsample': 0.5,      # 使用50%的數據進行訓練
            'subsample_freq': 1    # 每次迭代都進行子採樣
        }
        # 添加 GPU 相關參數（如果使用 GPU）
        # if self.device == 'gpu':
        #     params.update({
        #         'gpu_platform_id': 0,
        #         'gpu_device_id': 0,
        #         'gpu_use_dp': False,
        #     })
        return params

    def objective(self, trial):
        """
        Hyperparameters optimization
        :param trial: the current trial
        :return: the auc score achieved in the trial
        """
        try:
            train_x, train_y = self.get_train_data()
            valid_x, valid_y = self.get_test_data()
            
            print("\n=== 數據統計 ===")
            print(f"訓練集大小: {len(train_y)}, 正樣本: {sum(train_y)}, 負樣本: {len(train_y)-sum(train_y)}")
            print(f"驗證集大小: {len(valid_y)}, 正樣本: {sum(valid_y)}, 負樣本: {len(valid_y)-sum(valid_y)}")
            
            # 使用更高效的數據加載方式
            dtrain = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
            dvalid = lgb.Dataset(valid_x, label=valid_y, reference=dtrain, free_raw_data=False)
            valid_y_np = self.get_y_np(valid_y)

            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)
            num_leaves = trial.suggest_int('num_leaves', 15, 63)
            max_depth = trial.suggest_int('max_depth', 2, 10)
            max_bin = trial.suggest_int('max_bin', 127, 255)

            # 使用更快的參數設置
            params = self.get_default_params()
            params['learning_rate'] = learning_rate
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth
            params['max_bin'] = max_bin
            
            # 訓練模型，減少迭代次數
            bst = lgb.train(
                params,
                dtrain,
                num_boost_round=100,  # 減少最大迭代次數
                valid_sets=[dvalid],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=5),  # 提前停止輪數減少
                    lgb.log_evaluation(10),  # 每10輪輸出日誌
                    lgb.callback.reset_parameter(
                        learning_rate=lambda current_round: params['learning_rate'] * (0.99 ** current_round)
                    )  # 學習率衰減
                ]
            )
            
            # 預測並計算 AUC
            probas = bst.predict(valid_x, num_iteration=bst.best_iteration)
            auc_score = auc(valid_y_np, probas)
            
            print(f"試驗 {trial.number} 完成, AUC: {auc_score:.4f}")
            
            # 釋放記憶體
            del bst, dtrain, dvalid
            return auc_score
            
        except Exception as e:
            print(f"\n!!! 試驗出錯: {str(e)}")
            raise

    def train(self, x_train, y_train):
        """
        Initialize LGBM classifier and train it
        :param x_train: train features
        :param y_train: train target
        :return: the trained classifier
        """
        # 獲取超參數，如果沒有設置則使用默認值
        params = self.get_default_params()
        hyper_params = self.get_hyper_params()

        for k, v in hyper_params.items():
            params[k] = v

        # 準備數據集
        dtrain = lgb.Dataset(x_train, label=y_train, categorical_feature=self.categories) \
                     if self.categories is not None else lgb.Dataset(x_train, label=y_train)
        
        # 計算迭代次數
        # num_boost_round = min(1000, max(100, int((10 / (0.01 + params["learning_rate"]) ** 2) / 5)))
        num_boost_round = 1000

        # 訓練模型
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dtrain],
                valid_names=['train'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=True),
                    lgb.log_evaluation(50)
                ]
            )
        return model

    def predict_proba(self, clf, val_data):
        """
        Return the predicted probability for the given classifier.
        :param clf: LGBM classifier
        :param val_data: data
        :return: val_data's predicted probability
        """
        x_val = val_data[0]
        probs = clf.predict(x_val)
        if self.get_n_classes() == 2:
            probs = np.array([np.array([1 - i, i]) for i in probs])
        return probs

    def get_task(self):
        """
        Return the task based on the amount of classed in the data
        :return: binary if there are two classed and 'multiclass' otherwise
        """
        return 'binary' if self.get_n_classes() == 2 else 'multiclass'

    def save_model(self, clf, path):
        """
        Saved the model in .model format
        :param clf: LGBM classifier
        :param path: path to save the model in
        """
        clf.save_model(path+'.model')

    def get_num_classes(self):
        """Return the number of classes"""
        return 1 if self.get_n_classes() == 2 else self.get_n_classes()

    def load_model(self, path):
        """
        Load the LGBM classifier from the given path
        :param path: path
        :return: LGBM classifier
        """
        booster = lgb.Booster(model_file=path + '.model')
        booster.params['objective'] = self.get_task()
        return booster
