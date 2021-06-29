import pandas as pd
import numpy as np
import torch
from torch.nn import utils as nn_utils
import pickle

def csv_to_fill_mimic3():
    csv_data = pd.read_csv("dataset/lb_mimic3.csv", low_memory=False)
    csv_data = pd.concat([csv_data[['reporttime', 'hadm_id', 'gender', 'hospital_expire_flag']],
                          csv_data.select_dtypes(include=['float']).apply(pd.to_numeric, downcast='float')], axis=1)

    filter_feature = ['reporttime', 'hospital_expire_flag']
    features = []
    for x in csv_data.columns:
        if x not in filter_feature:
            features.append(x)
    train_data_x = csv_data[features]
    #train_data_y = csv_data['hospital_expire_flag']
    #train_data_y.to_csv('lb_mimic3_y.csv')
    """
    features_mode = {}
    for f in features:
        features_mode[f] = list(train_data_x[f].dropna().mode().values)[0]
    train_data_x.fillna(features_mode, inplace=True)
    """
    #print(train_data_x[:500])
    #print(pd.isna(train_data_x[:500].iloc[0]['heig']))
    M = pd.DataFrame(columns=train_data_x.columns.to_list())
    columnsList = train_data_x.columns.to_list()
    print(columnsList)
    print(len(columnsList))
    for i in range(train_data_x.index.size):
        appendRow = []
        for column in columnsList:
            if pd.isna(train_data_x.iloc[i][column]):
                appendRow.append(0)
            else:
                appendRow.append(1)
        M.loc[i] = appendRow

    #train_data_x.fillna(0)
    print(M)
    print(train_data_x)
    train_data_x.to_csv('lb_mimic3_x_fillbynan.csv')
    M.to_csv('lb_mimic3_x_M.csv')


#csv_to_fill_mimic3()

#def test():


#test()

def save_data_mimic3():
    data = pd.read_csv("dataset/lb_mimic3_x_fillbynan.csv",index_col=0)
    dataM = pd.read_csv("dataset/lb_mimic3_x_M.csv",index_col=0)
    datay = pd.read_csv("dataset/lb_mimic3_y.csv",index_col=0)
    z_scaler = lambda x: (x - np.mean(x)) / np.std(x)
    data2 = data[['anchor_age',
       'heig', 'hr', 'rr', 'spo2', 'nbpm', 'nbps', 'nbpd', 'abpm', 'abps',
       'abpd', 'fio2', 'temperaturef', 'cvp', 'peepset', 'meanairwaypressure',
       'tidalvolumeobserved', 'mvalarmhigh', 'mvalarmlow', 'apneainterval',
       'pawhigh', 'peakinsppressure', 'respiratoryratespontaneous',
       'minutevolume', 'vtihigh', 'respiratoryratetotal',
       'tidalvolumespontaneous', 'glucosefingerstick', 'it',
       'respiratoryrateset', 'hralarmlow', 'hralarmhigh', 'hematocrit',
       'potassium', 'sodium', 'creatinine', 'chloride', 'ureanitrogen',
       'bicarbonate', 'plateletcount', 'aniongap', 'whitebloodcells',
       'hemoglobin', 'glucose', 'mchc', 'redbloodcells', 'mch', 'mcv', 'rdw',
       'magnesium', 'calciumtotal', 'phosphate', 'ph', 'baseexcess',
       'calculatedtotalco2', 'po2', 'pco2', 'ptt', 'inr', 'pt',
       'bilirubintotal', 'freecalcium']].apply(z_scaler)
    data1 = pd.concat([data[['hadm_id','gender']],data2], axis=1)
    data1.fillna(0,inplace=True)
    print(data1)#419746*64
    print(dataM)#419746*64
    x = []
    x1 = []
    m = []
    m1 = []
    delta = []
    delta1 = []
    y = []
    hadmid = data1.iloc[0]['hadm_id']
    time = 0
    timeflag = 0
    for i in range(data1.index.size):

        if i%1000 == 0:
            print(i)

        if hadmid != data1.iloc[i]['hadm_id']:
            hadmid = data1.iloc[i]['hadm_id']

        x2 = []
        x2.append(data1.iloc[i]['anchor_age'])
        x2.append(data1.iloc[i]['heig'])
        x2.append(data1.iloc[i]['hr'])
        x2.append(data1.iloc[i]['rr'])
        x2.append(data1.iloc[i]['spo2'])
        x2.append(data1.iloc[i]['nbpm'])
        x2.append(data1.iloc[i]['nbps'])
        x2.append(data1.iloc[i]['nbpd'])
        x2.append(data1.iloc[i]['abpm'])
        x2.append(data1.iloc[i]['abps'])
        x2.append(data1.iloc[i]['abpd'])
        x2.append(data1.iloc[i]['fio2'])
        x2.append(data1.iloc[i]['temperaturef'])
        x2.append(data1.iloc[i]['cvp'])
        x2.append(data1.iloc[i]['peepset'])
        x2.append(data1.iloc[i]['meanairwaypressure'])
        x2.append(data1.iloc[i]['tidalvolumeobserved'])
        x2.append(data1.iloc[i]['mvalarmhigh'])
        x2.append(data1.iloc[i]['mvalarmlow'])
        x2.append(data1.iloc[i]['apneainterval'])
        x2.append(data1.iloc[i]['pawhigh'])
        x2.append(data1.iloc[i]['peakinsppressure'])
        x2.append(data1.iloc[i]['respiratoryratespontaneous'])
        x2.append(data1.iloc[i]['minutevolume'])
        x2.append(data1.iloc[i]['vtihigh'])
        x2.append(data1.iloc[i]['respiratoryratetotal'])
        x2.append(data1.iloc[i]['tidalvolumespontaneous'])
        x2.append(data1.iloc[i]['glucosefingerstick'])
        x2.append(data1.iloc[i]['it'])
        x2.append(data1.iloc[i]['respiratoryrateset'])
        x2.append(data1.iloc[i]['hralarmlow'])
        x2.append(data1.iloc[i]['hralarmhigh'])
        x2.append(data1.iloc[i]['hematocrit'])
        x2.append(data1.iloc[i]['potassium'])
        x2.append(data1.iloc[i]['sodium'])
        x2.append(data1.iloc[i]['creatinine'])
        x2.append(data1.iloc[i]['chloride'])
        x2.append(data1.iloc[i]['ureanitrogen'])
        x2.append(data1.iloc[i]['bicarbonate'])
        x2.append(data1.iloc[i]['plateletcount'])
        x2.append(data1.iloc[i]['aniongap'])
        x2.append(data1.iloc[i]['whitebloodcells'])
        x2.append(data1.iloc[i]['hemoglobin'])
        x2.append(data1.iloc[i]['glucose'])
        x2.append(data1.iloc[i]['mchc'])
        x2.append(data1.iloc[i]['redbloodcells'])
        x2.append(data1.iloc[i]['mch'])
        x2.append(data1.iloc[i]['mcv'])
        x2.append(data1.iloc[i]['rdw'])
        x2.append(data1.iloc[i]['magnesium'])
        x2.append(data1.iloc[i]['calciumtotal'])
        x2.append(data1.iloc[i]['phosphate'])
        x2.append(data1.iloc[i]['ph'])
        x2.append(data1.iloc[i]['baseexcess'])
        x2.append(data1.iloc[i]['calculatedtotalco2'])
        x2.append(data1.iloc[i]['po2'])
        x2.append(data1.iloc[i]['pco2'])
        x2.append(data1.iloc[i]['ptt'])
        x2.append(data1.iloc[i]['inr'])
        x2.append(data1.iloc[i]['pt'])
        x2.append(data1.iloc[i]['bilirubintotal'])
        x2.append(data1.iloc[i]['freecalcium'])

        m2 = []
        m2.append(dataM.iloc[i]['anchor_age'])
        m2.append(dataM.iloc[i]['heig'])
        m2.append(dataM.iloc[i]['hr'])
        m2.append(dataM.iloc[i]['rr'])
        m2.append(dataM.iloc[i]['spo2'])
        m2.append(dataM.iloc[i]['nbpm'])
        m2.append(dataM.iloc[i]['nbps'])
        m2.append(dataM.iloc[i]['nbpd'])
        m2.append(dataM.iloc[i]['abpm'])
        m2.append(dataM.iloc[i]['abps'])
        m2.append(dataM.iloc[i]['abpd'])
        m2.append(dataM.iloc[i]['fio2'])
        m2.append(dataM.iloc[i]['temperaturef'])
        m2.append(dataM.iloc[i]['cvp'])
        m2.append(dataM.iloc[i]['peepset'])
        m2.append(dataM.iloc[i]['meanairwaypressure'])
        m2.append(dataM.iloc[i]['tidalvolumeobserved'])
        m2.append(dataM.iloc[i]['mvalarmhigh'])
        m2.append(dataM.iloc[i]['mvalarmlow'])
        m2.append(dataM.iloc[i]['apneainterval'])
        m2.append(dataM.iloc[i]['pawhigh'])
        m2.append(dataM.iloc[i]['peakinsppressure'])
        m2.append(dataM.iloc[i]['respiratoryratespontaneous'])
        m2.append(dataM.iloc[i]['minutevolume'])
        m2.append(dataM.iloc[i]['vtihigh'])
        m2.append(dataM.iloc[i]['respiratoryratetotal'])
        m2.append(dataM.iloc[i]['tidalvolumespontaneous'])
        m2.append(dataM.iloc[i]['glucosefingerstick'])
        m2.append(dataM.iloc[i]['it'])
        m2.append(dataM.iloc[i]['respiratoryrateset'])
        m2.append(dataM.iloc[i]['hralarmlow'])
        m2.append(dataM.iloc[i]['hralarmhigh'])
        m2.append(dataM.iloc[i]['hematocrit'])
        m2.append(dataM.iloc[i]['potassium'])
        m2.append(dataM.iloc[i]['sodium'])
        m2.append(dataM.iloc[i]['creatinine'])
        m2.append(dataM.iloc[i]['chloride'])
        m2.append(dataM.iloc[i]['ureanitrogen'])
        m2.append(dataM.iloc[i]['bicarbonate'])
        m2.append(dataM.iloc[i]['plateletcount'])
        m2.append(dataM.iloc[i]['aniongap'])
        m2.append(dataM.iloc[i]['whitebloodcells'])
        m2.append(dataM.iloc[i]['hemoglobin'])
        m2.append(dataM.iloc[i]['glucose'])
        m2.append(dataM.iloc[i]['mchc'])
        m2.append(dataM.iloc[i]['redbloodcells'])
        m2.append(dataM.iloc[i]['mch'])
        m2.append(dataM.iloc[i]['mcv'])
        m2.append(dataM.iloc[i]['rdw'])
        m2.append(dataM.iloc[i]['magnesium'])
        m2.append(dataM.iloc[i]['calciumtotal'])
        m2.append(dataM.iloc[i]['phosphate'])
        m2.append(dataM.iloc[i]['ph'])
        m2.append(dataM.iloc[i]['baseexcess'])
        m2.append(dataM.iloc[i]['calculatedtotalco2'])
        m2.append(dataM.iloc[i]['po2'])
        m2.append(dataM.iloc[i]['pco2'])
        m2.append(dataM.iloc[i]['ptt'])
        m2.append(dataM.iloc[i]['inr'])
        m2.append(dataM.iloc[i]['pt'])
        m2.append(dataM.iloc[i]['bilirubintotal'])
        m2.append(dataM.iloc[i]['freecalcium'])
        if data1.iloc[i]['gender'] == 'f' or data1.iloc[i]['gender'] == 'F':
            x2.append(0)
        else:
            x2.append(1)
        m2.append(dataM.iloc[i]['gender'])
        x1.append(x2)
        m1.append(m2)

        delta2 = []

        #t=1 i=1
        if time == 0:
            for k in range(63):
                delta2.append(0)
        else:
        #t>1
            for j in range(63):
                # mt-1=0
                #print(m1)
                if m1[time - 1][j] == 0:
                    delta2.append(1 + delta1[time - 1][j])
                # mt-1=1
                elif m1[time - 1][j] == 1:
                    delta2.append(1)


        delta1.append(delta2)
        """
        print("i=" + str(i))
        print(x1)
        print(m1)
        print(delta1)
        """
        #print("i=" + str(i))
        #print(x2)
        #print(m2)
        #print(delta2)
        x2 = []
        m2 = []
        delta2 = []
        time = time + 1
        if (i + 1 < data1.index.size and hadmid != data1.iloc[i + 1]['hadm_id']) or i == data1.index.size - 1:
            x.append(x1)
            m.append(m1)
            delta.append(delta1)
            y.append(datay.iloc[i]['hospital_expire_flag'])
            x1 = []
            m1 = []
            delta1 = []
            time = 0

    #print(x)
    #print(m)
    #print(delta)
    x = list(map(lambda i: torch.tensor(i), x))
    m = list(map(lambda i: torch.tensor(i), m))
    delta = list(map(lambda i: torch.tensor(i), delta))
    #print(x)
    #lens = list(map(len, x))
    #lens = np.array(lens)
    #lens = torch.from_numpy(lens)

    #padded_sequence = nn_utils.rnn.pad_sequence(x, batch_first=True)

    padded_sequence_x = nn_utils.rnn.pad_sequence(x, batch_first=True)
    pickle.dump(padded_sequence_x, open('lb_mimic3_x_for_missingvalue.p', 'wb'))
    print(padded_sequence_x)
    print("--------------------save x success------------------")
    padded_sequence_m = nn_utils.rnn.pad_sequence(m, batch_first=True)
    pickle.dump(padded_sequence_m, open('lb_mimic3_m_for_missingvalue.p', 'wb'))
    print(padded_sequence_m)
    print("--------------------save m success------------------")
    padded_sequence_delta = nn_utils.rnn.pad_sequence(delta, batch_first=True)
    pickle.dump(padded_sequence_delta, open('lb_mimic3_delta_for_missingvalue.p', 'wb'))
    print(padded_sequence_delta)
    print("--------------------save delta success------------------")


    #y_array = np.array(y)
    #y_torch = torch.from_numpy(y_array)

#save_data_mimic3()

def test():
    x = pickle.load(open('dataset/lb_mimic3_x_for_missingvalue.p', 'rb'))

    x_lens = pickle.load(open('dataset/lb_mimic3_len.p', 'rb'))
    Mask = torch.zeros_like(x)
    for i in range(x.size(0)):
        if i % 1000 == 0:
            print(i)
        for j in range(x_lens[i]):
            for k in range(x.size(2)):
                Mask[i][j][k] = 1

    print(Mask.shape)
    print(Mask)
    pickle.dump(Mask, open('dataset/lb_mimic3_mask_for_missingvalue.p', 'wb'))



#test()

def csv_to_fill_mimic4():
    csv_data = pd.read_csv("lb_mimic4.csv", low_memory=False)
    csv_data = pd.concat([csv_data[['reporttime', 'hadm_id','anchor_age','gender', 'hospital_expire_flag']],
                          csv_data.select_dtypes(include=['float']).apply(pd.to_numeric, downcast='float')], axis=1)

    filter_feature = ['reporttime', 'hospital_expire_flag']
    features = []
    for x in csv_data.columns:
        if x not in filter_feature:
            features.append(x)
    train_data_x = csv_data[features]
    # train_data_y = csv_data['hospital_expire_flag']
    # train_data_y.to_csv('lb_mimic3_y.csv')
    """
    features_mode = {}
    for f in features:
        features_mode[f] = list(train_data_x[f].dropna().mode().values)[0]
    train_data_x.fillna(features_mode, inplace=True)
    """
    # print(train_data_x[:500])
    # print(pd.isna(train_data_x[:500].iloc[0]['heig']))

    columnsList = train_data_x.columns.to_list()
    #M = pd.DataFrame(columns=columnsList)
    M = np.zeros((1725642,64))
    totallens = train_data_x.index.size
    print(columnsList)
    print(len(columnsList))
    print(train_data_x.index.size)#1725642*63
    print(train_data_x)

    """
    train_data_x.to_csv('lb_mimic4_x_fillbynan.csv')
    print("----------------save X success-----------------")
    #del train_data_x
    """
    train_data_x = train_data_x.values

    for i in range(totallens):
        if i % 10000 == 0:
            print(i)

        for j in range(64):
            if not pd.isna(train_data_x[i][j]):
                M[i][j] = 1


    del train_data_x
    MDF = pd.DataFrame(M)
    del M
    MDF.columns=columnsList
    # train_data_x.fillna(0)
    print(MDF)
    MDF.to_csv('lb_mimic4_x_M.csv')
    print("----------------save M success-----------------")
    del MDF







#csv_to_fill_mimic4()

def save_data_mimic4():
    data = pd.read_csv("dataset/lb_mimic4_x_fillbynan.csv", index_col=0)
    dataM = pd.read_csv("dataset/lb_mimic4_x_M.csv", index_col=0)
    #datay = pd.read_csv("lb_mimic4_y.csv", index_col=0)
    z_scaler = lambda x: (x - np.mean(x)) / np.std(x)
    print(data.columns)
    data2 = data[['anchor_age',
       'heig', 'hr', 'rr', 'spo2', 'nbps', 'nbpd', 'nbpm', 'abpm', 'abps',
       'abpd', 'temperaturef', 'fio2', 'peepset', 'tidalvolume',
       'minutevolume', 'meanairwaypressure', 'peakinsppressure', 'mvalarmlow',
       'mvalarmhigh', 'apneainterval', 'pawhigh', 'respiratoryratespontaneous',
       'vtihigh', 'respiratoryratetotal', 'fspnhigh', 'cvp', 'glucosefs',
       'flowrate', 'hralarmlow', 'hralarmhigh', 'spo2alarmlow', 'hematocrit',
       'creatinine', 'plateletcount', 'hemoglobin', 'whitebloodcells',
       'ureanitrogen', 'mchc', 'redbloodcells', 'mcv', 'mch', 'rdw',
       'potassium', 'sodium', 'chloride', 'bicarbonate', 'aniongap', 'glucose',
       'calciumtotal', 'magnesium', 'phosphate', 'inr', 'pt',
       'alanineaminotransferase', 'asparateaminotransferase', 'ptt',
       'bilirubintotal', 'neutrophils', 'lymphocytes', 'monocytes',
       'eosinophils']].apply(z_scaler)
    data1 = pd.concat([data[['hadm_id', 'gender']], data2], axis=1)
    # print(data1)#419746*64
    del data2
    del data
    data1.fillna(0, inplace=True)
    print(data1)  # 419746*64
    print(dataM)  # 419746*64
    x = []
    x1 = []
    m = []
    m1 = []
    delta = []
    delta1 = []
    y = []
    hadmid = data1.iloc[0]['hadm_id']
    time = 0
    timeflag = 0
    for i in range(data1.index.size):

        if i % 1000 == 0:
            print(i)

        if hadmid != data1.iloc[i]['hadm_id']:
            hadmid = data1.iloc[i]['hadm_id']

        x2 = []
        x2.append(data1.iloc[i]['anchor_age'])
        x2.append(data1.iloc[i]['heig'])
        x2.append(data1.iloc[i]['hr'])
        x2.append(data1.iloc[i]['rr'])
        x2.append(data1.iloc[i]['spo2'])
        x2.append(data1.iloc[i]['nbpm'])
        x2.append(data1.iloc[i]['nbps'])
        x2.append(data1.iloc[i]['nbpd'])
        x2.append(data1.iloc[i]['abpm'])
        x2.append(data1.iloc[i]['abps'])
        x2.append(data1.iloc[i]['abpd'])
        x2.append(data1.iloc[i]['fio2'])
        x2.append(data1.iloc[i]['temperaturef'])
        x2.append(data1.iloc[i]['cvp'])
        x2.append(data1.iloc[i]['peepset'])
        x2.append(data1.iloc[i]['tidalvolume'])
        x2.append(data1.iloc[i]['minutevolume'])
        x2.append(data1.iloc[i]['meanairwaypressure'])
        x2.append(data1.iloc[i]['peakinsppressure'])
        x2.append(data1.iloc[i]['mvalarmlow'])
        x2.append(data1.iloc[i]['mvalarmhigh'])
        x2.append(data1.iloc[i]['apneainterval'])
        x2.append(data1.iloc[i]['pawhigh'])
        x2.append(data1.iloc[i]['respiratoryratespontaneous'])
        x2.append(data1.iloc[i]['vtihigh'])
        x2.append(data1.iloc[i]['respiratoryratetotal'])
        x2.append(data1.iloc[i]['fspnhigh'])
        x2.append(data1.iloc[i]['glucosefs'])
        x2.append(data1.iloc[i]['flowrate'])
        x2.append(data1.iloc[i]['hralarmlow'])
        x2.append(data1.iloc[i]['hralarmhigh'])
        x2.append(data1.iloc[i]['spo2alarmlow'])
        x2.append(data1.iloc[i]['hematocrit'])
        x2.append(data1.iloc[i]['creatinine'])
        x2.append(data1.iloc[i]['plateletcount'])
        x2.append(data1.iloc[i]['hemoglobin'])
        x2.append(data1.iloc[i]['whitebloodcells'])
        x2.append(data1.iloc[i]['ureanitrogen'])
        x2.append(data1.iloc[i]['mchc'])
        x2.append(data1.iloc[i]['redbloodcells'])
        x2.append(data1.iloc[i]['mcv'])
        x2.append(data1.iloc[i]['mch'])
        x2.append(data1.iloc[i]['rdw'])
        x2.append(data1.iloc[i]['potassium'])
        x2.append(data1.iloc[i]['sodium'])
        x2.append(data1.iloc[i]['chloride'])
        x2.append(data1.iloc[i]['bicarbonate'])
        x2.append(data1.iloc[i]['aniongap'])
        x2.append(data1.iloc[i]['glucose'])
        x2.append(data1.iloc[i]['calciumtotal'])
        x2.append(data1.iloc[i]['magnesium'])
        x2.append(data1.iloc[i]['phosphate'])
        x2.append(data1.iloc[i]['inr'])
        x2.append(data1.iloc[i]['pt'])
        x2.append(data1.iloc[i]['alanineaminotransferase'])
        x2.append(data1.iloc[i]['asparateaminotransferase'])
        x2.append(data1.iloc[i]['ptt'])
        x2.append(data1.iloc[i]['bilirubintotal'])
        x2.append(data1.iloc[i]['neutrophils'])
        x2.append(data1.iloc[i]['lymphocytes'])
        x2.append(data1.iloc[i]['monocytes'])
        x2.append(data1.iloc[i]['eosinophils'])

        m2 = []
        m2.append(dataM.iloc[i]['anchor_age'])
        m2.append(dataM.iloc[i]['heig'])
        m2.append(dataM.iloc[i]['hr'])
        m2.append(dataM.iloc[i]['rr'])
        m2.append(dataM.iloc[i]['spo2'])
        m2.append(dataM.iloc[i]['nbpm'])
        m2.append(dataM.iloc[i]['nbps'])
        m2.append(dataM.iloc[i]['nbpd'])
        m2.append(dataM.iloc[i]['abpm'])
        m2.append(dataM.iloc[i]['abps'])
        m2.append(dataM.iloc[i]['abpd'])
        m2.append(dataM.iloc[i]['fio2'])
        m2.append(dataM.iloc[i]['temperaturef'])
        m2.append(dataM.iloc[i]['cvp'])
        m2.append(dataM.iloc[i]['peepset'])
        m2.append(dataM.iloc[i]['tidalvolume'])
        m2.append(dataM.iloc[i]['minutevolume'])
        m2.append(dataM.iloc[i]['meanairwaypressure'])
        m2.append(dataM.iloc[i]['peakinsppressure'])
        m2.append(dataM.iloc[i]['mvalarmlow'])
        m2.append(dataM.iloc[i]['mvalarmhigh'])
        m2.append(dataM.iloc[i]['apneainterval'])
        m2.append(dataM.iloc[i]['pawhigh'])
        m2.append(dataM.iloc[i]['respiratoryratespontaneous'])
        m2.append(dataM.iloc[i]['vtihigh'])
        m2.append(dataM.iloc[i]['respiratoryratetotal'])
        m2.append(dataM.iloc[i]['fspnhigh'])
        m2.append(dataM.iloc[i]['glucosefs'])
        m2.append(dataM.iloc[i]['flowrate'])
        m2.append(dataM.iloc[i]['hralarmlow'])
        m2.append(dataM.iloc[i]['hralarmhigh'])
        m2.append(dataM.iloc[i]['spo2alarmlow'])
        m2.append(dataM.iloc[i]['hematocrit'])
        m2.append(dataM.iloc[i]['creatinine'])
        m2.append(dataM.iloc[i]['plateletcount'])
        m2.append(dataM.iloc[i]['hemoglobin'])
        m2.append(dataM.iloc[i]['whitebloodcells'])
        m2.append(dataM.iloc[i]['ureanitrogen'])
        m2.append(dataM.iloc[i]['mchc'])
        m2.append(dataM.iloc[i]['redbloodcells'])
        m2.append(dataM.iloc[i]['mcv'])
        m2.append(dataM.iloc[i]['mch'])
        m2.append(dataM.iloc[i]['rdw'])
        m2.append(dataM.iloc[i]['potassium'])
        m2.append(dataM.iloc[i]['sodium'])
        m2.append(dataM.iloc[i]['chloride'])
        m2.append(dataM.iloc[i]['bicarbonate'])
        m2.append(dataM.iloc[i]['aniongap'])
        m2.append(dataM.iloc[i]['glucose'])
        m2.append(dataM.iloc[i]['calciumtotal'])
        m2.append(dataM.iloc[i]['magnesium'])
        m2.append(dataM.iloc[i]['phosphate'])
        m2.append(dataM.iloc[i]['inr'])
        m2.append(dataM.iloc[i]['pt'])
        m2.append(dataM.iloc[i]['alanineaminotransferase'])
        m2.append(dataM.iloc[i]['asparateaminotransferase'])
        m2.append(dataM.iloc[i]['ptt'])
        m2.append(dataM.iloc[i]['bilirubintotal'])
        m2.append(dataM.iloc[i]['neutrophils'])
        m2.append(dataM.iloc[i]['lymphocytes'])
        m2.append(dataM.iloc[i]['monocytes'])
        m2.append(dataM.iloc[i]['eosinophils'])
        if data1.iloc[i]['gender'] == 'f' or data1.iloc[i]['gender'] == 'F':
            x2.append(0)
        else:
            x2.append(1)
        m2.append(dataM.iloc[i]['gender'])
        x1.append(x2)
        m1.append(m2)

        delta2 = []

        # t=1 i=1
        if time == 0:
            for k in range(63):
                delta2.append(0)
        else:
            # t>1
            for j in range(63):
                # mt-1=0
                # print(m1)
                if m1[time - 1][j] == 0:
                    delta2.append(1 + delta1[time - 1][j])
                # mt-1=1
                elif m1[time - 1][j] == 1:
                    delta2.append(1)

        delta1.append(delta2)
        """
        print("i=" + str(i))
        print(x1)
        print(m1)
        print(delta1)
        """
        # print("i=" + str(i))
        # print(x2)
        # print(m2)
        # print(delta2)
        x2 = []
        m2 = []
        delta2 = []
        time = time + 1
        if (i + 1 < data1.index.size and hadmid != data1.iloc[i + 1]['hadm_id']) or i == data1.index.size - 1:
            x.append(x1)
            m.append(m1)
            delta.append(delta1)
            #y.append(datay.iloc[i]['hospital_expire_flag'])
            x1 = []
            m1 = []
            delta1 = []
            time = 0

    del dataM
    del data1
    #del datay
    # print(x)
    # print(m)
    # print(delta)
    # print(x)
    # lens = list(map(len, x))
    # lens = np.array(lens)
    # lens = torch.from_numpy(lens)

    # padded_sequence = nn_utils.rnn.pad_sequence(x, batch_first=True)
    x = list(map(lambda i: torch.tensor(i), x))
    padded_sequence_x = nn_utils.rnn.pad_sequence(x, batch_first=True)
    del x
    pickle.dump(padded_sequence_x, open('lb_mimic4_x_for_missingvalue.p', 'wb'), protocol=4)
    print(padded_sequence_x)
    print("--------------------save x success------------------")
    del padded_sequence_x

    m = list(map(lambda i: torch.tensor(i), m))
    padded_sequence_m = nn_utils.rnn.pad_sequence(m, batch_first=True)
    del m
    pickle.dump(padded_sequence_m, open('lb_mimic4_m_for_missingvalue.p', 'wb'), protocol=4)
    print(padded_sequence_m)
    print("--------------------save m success------------------")
    del padded_sequence_m

    delta = list(map(lambda i: torch.tensor(i), delta))
    padded_sequence_delta = nn_utils.rnn.pad_sequence(delta, batch_first=True)
    del delta
    pickle.dump(padded_sequence_delta, open('lb_mimic4_delta_for_missingvalue.p', 'wb'), protocol=4)
    print(padded_sequence_delta)
    print("--------------------save delta success------------------")
    del padded_sequence_delta

#save_data_mimic3()
#save_data_mimic4()

def save_data_mimic3_for_tra():
    data = pd.read_csv("lb_mimic3_x_fill.csv",index_col=0)
    datay = pd.read_csv("lb_mimic3_y.csv",index_col=0)
    z_scaler = lambda x: (x - np.mean(x)) / np.std(x)
    data2 = data[['anchor_age',
       'heig', 'hr', 'rr', 'spo2', 'nbpm', 'nbps', 'nbpd', 'abpm', 'abps',
       'abpd', 'fio2', 'temperaturef', 'cvp', 'peepset', 'meanairwaypressure',
       'tidalvolumeobserved', 'mvalarmhigh', 'mvalarmlow', 'apneainterval',
       'pawhigh', 'peakinsppressure', 'respiratoryratespontaneous',
       'minutevolume', 'vtihigh', 'respiratoryratetotal',
       'tidalvolumespontaneous', 'glucosefingerstick', 'it',
       'respiratoryrateset', 'hralarmlow', 'hralarmhigh', 'hematocrit',
       'potassium', 'sodium', 'creatinine', 'chloride', 'ureanitrogen',
       'bicarbonate', 'plateletcount', 'aniongap', 'whitebloodcells',
       'hemoglobin', 'glucose', 'mchc', 'redbloodcells', 'mch', 'mcv', 'rdw',
       'magnesium', 'calciumtotal', 'phosphate', 'ph', 'baseexcess',
       'calculatedtotalco2', 'po2', 'pco2', 'ptt', 'inr', 'pt',
       'bilirubintotal', 'freecalcium']].apply(z_scaler)
    data1 = pd.concat([data[['hadm_id','gender']],data2], axis=1)
    x = []
    y = []
    hadmid = data1.iloc[0]['hadm_id']
    for i in range(data.index.size):
        if i%1000 == 0:
            print(i)

        if hadmid != data1.iloc[i]['hadm_id']:
            hadmid = data1.iloc[i]['hadm_id']
        x2 = []
        x2.append(data1.iloc[i]['anchor_age'])
        x2.append(data1.iloc[i]['heig'])
        x2.append(data1.iloc[i]['hr'])
        x2.append(data1.iloc[i]['rr'])
        x2.append(data1.iloc[i]['spo2'])
        x2.append(data1.iloc[i]['nbpm'])
        x2.append(data1.iloc[i]['nbps'])
        x2.append(data1.iloc[i]['nbpd'])
        x2.append(data1.iloc[i]['abpm'])
        x2.append(data1.iloc[i]['abps'])
        x2.append(data1.iloc[i]['abpd'])
        x2.append(data1.iloc[i]['fio2'])
        x2.append(data1.iloc[i]['temperaturef'])
        x2.append(data1.iloc[i]['cvp'])
        x2.append(data1.iloc[i]['peepset'])
        x2.append(data1.iloc[i]['meanairwaypressure'])
        x2.append(data1.iloc[i]['tidalvolumeobserved'])
        x2.append(data1.iloc[i]['mvalarmhigh'])
        x2.append(data1.iloc[i]['mvalarmlow'])
        x2.append(data1.iloc[i]['apneainterval'])
        x2.append(data1.iloc[i]['pawhigh'])
        x2.append(data1.iloc[i]['peakinsppressure'])
        x2.append(data1.iloc[i]['respiratoryratespontaneous'])
        x2.append(data1.iloc[i]['minutevolume'])
        x2.append(data1.iloc[i]['vtihigh'])
        x2.append(data1.iloc[i]['respiratoryratetotal'])
        x2.append(data1.iloc[i]['tidalvolumespontaneous'])
        x2.append(data1.iloc[i]['glucosefingerstick'])
        x2.append(data1.iloc[i]['it'])
        x2.append(data1.iloc[i]['respiratoryrateset'])
        x2.append(data1.iloc[i]['hralarmlow'])
        x2.append(data1.iloc[i]['hralarmhigh'])
        x2.append(data1.iloc[i]['hematocrit'])
        x2.append(data1.iloc[i]['potassium'])
        x2.append(data1.iloc[i]['sodium'])
        x2.append(data1.iloc[i]['creatinine'])
        x2.append(data1.iloc[i]['chloride'])
        x2.append(data1.iloc[i]['ureanitrogen'])
        x2.append(data1.iloc[i]['bicarbonate'])
        x2.append(data1.iloc[i]['plateletcount'])
        x2.append(data1.iloc[i]['aniongap'])
        x2.append(data1.iloc[i]['whitebloodcells'])
        x2.append(data1.iloc[i]['hemoglobin'])
        x2.append(data1.iloc[i]['glucose'])
        x2.append(data1.iloc[i]['mchc'])
        x2.append(data1.iloc[i]['redbloodcells'])
        x2.append(data1.iloc[i]['mch'])
        x2.append(data1.iloc[i]['mcv'])
        x2.append(data1.iloc[i]['rdw'])
        x2.append(data1.iloc[i]['magnesium'])
        x2.append(data1.iloc[i]['calciumtotal'])
        x2.append(data1.iloc[i]['phosphate'])
        x2.append(data1.iloc[i]['ph'])
        x2.append(data1.iloc[i]['baseexcess'])
        x2.append(data1.iloc[i]['calculatedtotalco2'])
        x2.append(data1.iloc[i]['po2'])
        x2.append(data1.iloc[i]['pco2'])
        x2.append(data1.iloc[i]['ptt'])
        x2.append(data1.iloc[i]['inr'])
        x2.append(data1.iloc[i]['pt'])
        x2.append(data1.iloc[i]['bilirubintotal'])
        x2.append(data1.iloc[i]['freecalcium'])
        if data1.iloc[i]['gender'] == 'f' or data1.iloc[i]['gender'] == 'F':
            x2.append(0)
        else:
            x2.append(1)
        x.append(x2)
        y.append(datay.iloc[i]['hospital_expire_flag'])
        x2 = []

    y_array = np.array(y)
    y_torch = torch.from_numpy(y_array)
    pickle.dump(y_torch, open('lb_mimic3_y_for_tra.p', 'wb'))

    x = torch.Tensor(x)
    pickle.dump(x, open('lb_mimic3_x_for_tra.p', 'wb'))

def save_data_mimic4_for_tra():
    data = pd.read_csv("lb_mimic4_x_fill.csv",index_col=0)
    datay = pd.read_csv("lb_mimic4_y.csv",index_col=0)
    z_scaler = lambda x: (x - np.mean(x)) / np.std(x)
    data2 = data[['anchor_age',
       'heig', 'hr', 'rr', 'spo2', 'nbps', 'nbpd', 'nbpm', 'abpm', 'abps',
       'abpd', 'temperaturef', 'fio2', 'peepset', 'tidalvolume',
       'minutevolume', 'meanairwaypressure', 'peakinsppressure', 'mvalarmlow',
       'mvalarmhigh', 'apneainterval', 'pawhigh', 'respiratoryratespontaneous',
       'vtihigh', 'respiratoryratetotal', 'fspnhigh', 'cvp', 'glucosefs',
       'flowrate', 'hralarmlow', 'hralarmhigh', 'spo2alarmlow', 'hematocrit',
       'creatinine', 'plateletcount', 'hemoglobin', 'whitebloodcells',
       'ureanitrogen', 'mchc', 'redbloodcells', 'mcv', 'mch', 'rdw',
       'potassium', 'sodium', 'chloride', 'bicarbonate', 'aniongap', 'glucose',
       'calciumtotal', 'magnesium', 'phosphate', 'inr', 'pt',
       'alanineaminotransferase', 'asparateaminotransferase', 'ptt',
       'bilirubintotal', 'neutrophils', 'lymphocytes', 'monocytes',
       'eosinophils']].apply(z_scaler)
    data1 = pd.concat([data[['hadm_id','gender']],data2], axis=1)
    x = []
    y = []
    hadmid = data1.iloc[0]['hadm_id']
    for i in range(data.index.size):
        if i%1000 == 0:
            print(i)

        if hadmid != data1.iloc[i]['hadm_id']:
            hadmid = data1.iloc[i]['hadm_id']
        x2 = []
        x2.append(data1.iloc[i]['anchor_age'])
        x2.append(data1.iloc[i]['heig'])
        x2.append(data1.iloc[i]['hr'])
        x2.append(data1.iloc[i]['rr'])
        x2.append(data1.iloc[i]['spo2'])
        x2.append(data1.iloc[i]['nbpm'])
        x2.append(data1.iloc[i]['nbps'])
        x2.append(data1.iloc[i]['nbpd'])
        x2.append(data1.iloc[i]['abpm'])
        x2.append(data1.iloc[i]['abps'])
        x2.append(data1.iloc[i]['abpd'])
        x2.append(data1.iloc[i]['fio2'])
        x2.append(data1.iloc[i]['temperaturef'])
        x2.append(data1.iloc[i]['cvp'])
        x2.append(data1.iloc[i]['peepset'])
        x2.append(data1.iloc[i]['tidalvolume'])
        x2.append(data1.iloc[i]['minutevolume'])
        x2.append(data1.iloc[i]['meanairwaypressure'])
        x2.append(data1.iloc[i]['peakinsppressure'])
        x2.append(data1.iloc[i]['mvalarmlow'])
        x2.append(data1.iloc[i]['mvalarmhigh'])
        x2.append(data1.iloc[i]['apneainterval'])
        x2.append(data1.iloc[i]['pawhigh'])
        x2.append(data1.iloc[i]['respiratoryratespontaneous'])
        x2.append(data1.iloc[i]['vtihigh'])
        x2.append(data1.iloc[i]['respiratoryratetotal'])
        x2.append(data1.iloc[i]['fspnhigh'])
        x2.append(data1.iloc[i]['glucosefs'])
        x2.append(data1.iloc[i]['flowrate'])
        x2.append(data1.iloc[i]['hralarmlow'])
        x2.append(data1.iloc[i]['hralarmhigh'])
        x2.append(data1.iloc[i]['spo2alarmlow'])
        x2.append(data1.iloc[i]['hematocrit'])
        x2.append(data1.iloc[i]['creatinine'])
        x2.append(data1.iloc[i]['plateletcount'])
        x2.append(data1.iloc[i]['hemoglobin'])
        x2.append(data1.iloc[i]['whitebloodcells'])
        x2.append(data1.iloc[i]['ureanitrogen'])
        x2.append(data1.iloc[i]['mchc'])
        x2.append(data1.iloc[i]['redbloodcells'])
        x2.append(data1.iloc[i]['mcv'])
        x2.append(data1.iloc[i]['mch'])
        x2.append(data1.iloc[i]['rdw'])
        x2.append(data1.iloc[i]['potassium'])
        x2.append(data1.iloc[i]['sodium'])
        x2.append(data1.iloc[i]['chloride'])
        x2.append(data1.iloc[i]['bicarbonate'])
        x2.append(data1.iloc[i]['aniongap'])
        x2.append(data1.iloc[i]['glucose'])
        x2.append(data1.iloc[i]['calciumtotal'])
        x2.append(data1.iloc[i]['magnesium'])
        x2.append(data1.iloc[i]['phosphate'])
        x2.append(data1.iloc[i]['inr'])
        x2.append(data1.iloc[i]['pt'])
        x2.append(data1.iloc[i]['alanineaminotransferase'])
        x2.append(data1.iloc[i]['asparateaminotransferase'])
        x2.append(data1.iloc[i]['ptt'])
        x2.append(data1.iloc[i]['bilirubintotal'])
        x2.append(data1.iloc[i]['neutrophils'])
        x2.append(data1.iloc[i]['lymphocytes'])
        x2.append(data1.iloc[i]['monocytes'])
        x2.append(data1.iloc[i]['eosinophils'])
        if data1.iloc[i]['gender'] == 'f' or data1.iloc[i]['gender'] == 'F':
            x2.append(0)
        else:
            x2.append(1)
        x.append(x2)
        y.append(datay.iloc[i]['hospital_expire_flag'])
        x2 = []

    del data
    del data1
    del data2
    del datay

    y_array = np.array(y)
    y_torch = torch.from_numpy(y_array)
    pickle.dump(y_torch, open('lb_mimic4_y_for_tra.p', 'wb'), protocol=4)
    del y_torch
    del y_array
    del y

    x = torch.Tensor(x)
    pickle.dump(x, open('lb_mimic4_x_for_tra.p', 'wb'), protocol=4)
    del x



def randm():
    datatype = "mimic3"

    m_torch = pickle.load(open('dataset/lb_' + datatype + '_m_for_missingvalue.p', 'rb'))
    x_lens = pickle.load(open('dataset/lb_' + datatype + '_len.p', 'rb'))
    N = len(m_torch)
    training_randm = torch.randint_like(m_torch[: int(0.8 * N)],low=1, high=3)
    training_randm = training_randm - torch.ones_like(training_randm)  # 除了0就是2
