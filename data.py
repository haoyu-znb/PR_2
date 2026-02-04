import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import scipy.sparse as sp

class Multi_view_data(Dataset):

    def __init__(self, data_path, name=None):

        super(Multi_view_data, self).__init__()

        self.data_name = name
        dataset = sio.loadmat(data_path)

        self.X = dict()
        view_number = 0

        # ==========================================================
        #  数据集适配逻辑
        # ==========================================================
        
        if name == 'Fashion':
            view_number = 3
            self.Y = dataset['Y'].reshape(-1).astype(int) 
            for v_num in range(view_number):
                X = dataset['X' + str(v_num + 1)]
                self.X[v_num] = self.normalize(X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]))
        
        elif name == 'Leaves' or name == 'LandUse' or name == 'NUSOBJ':
            self.Y = dataset['Y'].reshape(-1).astype(int)
            view_number = dataset['X'].shape[1]
            for v_num in range(view_number):
                x = dataset['X'][0][v_num]
                if sp.issparse(x):
                    x = x.toarray()
                self.X[v_num] = self.normalize(x)
        
        elif name == 'MSRC':
            self.Y = dataset['gt'].reshape(-1).astype(int) 
            for v_num in range(int((len(dataset) - 4))): 
                self.X[v_num] = self.normalize(dataset['x' + str(v_num + 1) ]) 
            view_number = len(self.X)

        # -----------------------------------------------------------
        # [关键修复] 智能转置 + 智能标签查找
        # -----------------------------------------------------------
        elif name == 'PIE' or name == 'Scene':        
            # 1. 找标签
            if 'gt' in dataset:
                self.Y = np.squeeze(dataset['gt'])
            elif 'Y' in dataset:
                self.Y = np.squeeze(dataset['Y'])
            elif 'label' in dataset:
                self.Y = np.squeeze(dataset['label'])
            elif 'gnd' in dataset:
                self.Y = np.squeeze(dataset['gnd'])
            elif 'labels' in dataset:
                self.Y = np.squeeze(dataset['labels'])
            else:
                print(f"Error: Keys found: {dataset.keys()}")
                raise KeyError(f"No label key found in {data_path}")

            # 2. 找数据
            if 'X' in dataset:
                data_X = dataset['X'][0]
            elif 'data' in dataset:
                data_X = dataset['data'][0]
            else:
                raise KeyError("No data key 'X' or 'data' found")

            # 3. [修复点] 智能转置：确保形状是 (Sample, Feature)
            # 逻辑：如果 行数 < 列数，说明大概率反了 (通常 N_sample >> N_feature)
            processed_views = []
            for v in range(len(data_X)):
                mat = data_X[v]
                shape = mat.shape
                # 如果行数明显小于列数（且不是Fashion这种图像数据），大概率是 D x N，需要转置回 N x D
                # PIE: 680 samples, dims ~480 -> 680 > 480 (N > D) -> 不转置? 等等，PIE原数据可能是 D x N
                # Scene: 4485 samples, dims ~20 -> 4485 > 20 (N > D) -> 必须保证 N 在第一维
                
                # 简单粗暴规则：取较大的那一维作为样本数 (因为 Scene 和 PIE 的样本数都大于特征维数)
                if shape[0] < shape[1]:
                    # 当前是 (Feature, Sample)，转置为 (Sample, Feature)
                    processed_views.append(mat.T)
                else:
                    # 当前已经是 (Sample, Feature)，不动
                    processed_views.append(mat)
            
            view_number = len(processed_views)
            self.X = dict()
            for v in range(view_number):
                self.X[v] = self.normalize(processed_views[v])
        # -----------------------------------------------------------

        elif name == 'HandWritten':
            self.Y = np.squeeze(dataset['Y'])
            data_X = dataset['X'][0]    
            view_number = data_X.shape[0]
            self.X = dict()
            for v in range(view_number):
                self.X[v] = self.normalize(data_X[v])
                
        if self.Y.min() == 1:
            self.Y = self.Y - 1
            
        self.Y = self.Y.astype(dtype=np.int64)
        self.num_classes = len(np.unique(self.Y))
        self.num_views = view_number
        self.dims = self.get_dims()

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)  
        target = self.Y[index]  
        return data, target, index

    def __len__(self):
        return len(self.X[0])
    
    def get_dims(self):
        dims = []
        for view in range(self.num_views):
            dims.append([self.X[view].shape[1]])

        return np.array(dims)
    
    @staticmethod
    def normalize(x, min=0):
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:  
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def postprocessing(self, index, addNoise=False, sigma=0, ratio_noise=0.5, addConflict=False, ratio_conflict=0.5):
        if addNoise:
            self.addNoise(index, ratio_noise, sigma=sigma)
            print("addNoise")
        if addConflict:
            self.addConflict(index, ratio_conflict)
            print("addConflict")
        pass

    def addNoise(self, index, ratio, sigma):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            views = np.random.choice(np.array(self.num_views), size=np.random.randint(self.num_views), replace=False)
            for v in views:
                self.X[v][i] = np.random.normal(self.X[v][i], sigma)

    def addConflict(self, index, ratio):
        records = dict()
        for c in range(self.num_classes):
            i = np.where(self.Y == c)[0][0] 
            temp = dict()
            for v in range(self.num_views): 
                temp[v] = self.X[v][i]
            records[c] = temp
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            v = np.random.randint(self.num_views)
            self.X[v][i] = records[(self.Y[i] + 1) % self.num_classes][v]
            
def HandWritten():
    data_path = "datasets/handwritten.mat"
    return Multi_view_data(data_path, name='HandWritten')

def Scene_15(): 
    data_path = "datasets/scene15_mtv.mat" 
    return Multi_view_data(data_path, name='Scene')

def Scene():
    data_path = "datasets/scene15_mtv.mat"
    return Multi_view_data(data_path, name='Scene')

def PIE():
    data_path = "datasets/PIE_face_10.mat"
    return Multi_view_data(data_path, name='PIE')

def MSRC():
    data_path = "datasets/MSRCV1_6views.mat"
    return Multi_view_data(data_path, name='MSRC')

def Leaves():
    data_path = "datasets/100Leaves.mat"
    return Multi_view_data(data_path, name='Leaves')

def Fashion():
    data_path = "datasets/3V_Fashion_MV.mat"
    return Multi_view_data(data_path, name="Fashion")

def NUSOBJ():
    data_path = "datasets/NUSWIDEOBJ.mat"
    return Multi_view_data(data_path, name="NUSOBJ")

def LandUse_21(): 
    data_path = "datasets/LandUse_21.mat"
    return Multi_view_data(data_path, name="LandUse")

def LandUse():
    data_path = "datasets/LandUse_21.mat"
    return Multi_view_data(data_path, name="LandUse")